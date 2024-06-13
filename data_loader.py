import os
import gc
from functools import partial
import gzip
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler
import torch
from sklearn.model_selection import KFold
from datasets import load_dataset
import threading
from multiprocessing import Process, Queue, current_process


def load_one_gz_file(file_path, field):
    df = pd.read_csv(file_path, sep='\t', compression='gzip', index_col=False, names=field, header=None, 
                                na_values='NULL')
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data
    return df


def load_df(source_dir, field, is_train, num_folds, fold_id):
        # in case we want dataframe data
        files = os.listdir(source_dir)
        
        indices = np.arange(len(files))
        skf = KFold(n_splits=num_folds, shuffle=False)
        for i, (train_indices, test_indices) in enumerate(skf.split(indices)):
            if i == fold_id:
                break
        indices = train_indices if is_train else test_indices
                
        data = []
        for i in indices:   
            data.append(load_one_gz_file(os.path.join(source_dir, files[i]), field))
        data = pd.concat(data, ignore_index=True)
        data.reset_index(drop=True, inplace=True)
        return data


def collate_fn_tensor(batch: torch.tensor, target_treatment: list, target_task: list,):
    # target_treatment: target treatment column idx
    # target_task: target task column idx
    features = torch.stack([sample[:536] for sample in batch])
    assert min(min(target_treatment), min(target_task)) > 535, "check the indices for treatments or tasks"
    treatments = torch.stack([sample[target_treatment] for sample in batch])
    tasks = torch.stack([sample[target_task] for sample in batch])
    return features, treatments, tasks


def collate_fn_dict(batch, target_treatment, target_task):
    assert min(min(target_treatment), min(target_task)) > 535, "check the indices for treatments or tasks"
    treatments = torch.stack([sample[target_treatment] for sample in batch])
    tasks = torch.stack([sample[target_task] for sample in batch])
    features = torch.stack([list(sample.values())[:536] for sample in batch])
    return features, treatments, tasks


def collate_fn_ndarray(batch: np.ndarray, target_treatment: list, target_task: list):
    features = [sample[:536] for sample in batch]
    assert min(min(target_treatment), min(target_task)) > 535, "check the indices for treatments or tasks"
    treatments = [sample[target_treatment] for sample in batch]
    tasks = [sample[target_task] for sample in batch]
    features = torch.tensor(features).to(torch.float32)
    treatments = torch.tensor(treatments).to(torch.float32).squeeze()
    tasks = torch.tensor(tasks).to(torch.float32).squeeze()
    return features, treatments, tasks


def create_folds(file_paths, n_folds=5):
    sub_datasets = [(file_paths[i], file_paths[i+1]) for i in range(0, len(file_paths), 2)]
    kf = KFold(n_splits=n_folds, shuffle=False)
    
    folds = []
    for train_index, test_index in kf.split(sub_datasets):
        train_files = [file for idx in train_index for file in sub_datasets[idx]]
        test_files = [file for idx in test_index for file in sub_datasets[idx]]
        folds.append((train_files, test_files))
    return folds


def get_data_(source_dir, target_treatment, target_task, fold_id, batch_size, num_folds=5):
    # directly access gz files
    # train_set = CustomDataset(source_dir, is_train=True, fold_id=fold_id, num_folds=num_folds)
    # test_set  = CustomDataset(source_dir, is_train=False, fold_id=fold_id, num_folds=num_folds) 

    # process parquet files. must use streaming or it may oom
    # dataset = load_dataset('parquet', data_files='data/demo/dataset_demo.parquet', split='train', streaming=True)
    # train_set = dataset.take(100000)
    # test_set = dataset.skip(100000).take(20000)
    
    # split train and test
    dataset = CustomDatasetHdf5('data/demo/dataset_demo.hdf5', None)
    skf = KFold(n_splits=num_folds, shuffle=False)
    for i, (train_indices, test_indices) in enumerate(skf.split(np.arange(len(dataset)))):
        if i == fold_id:
            break
    train_set = CustomDatasetHdf5('data/demo/dataset_demo.hdf5', train_indices)
    test_set  = CustomDatasetHdf5('data/demo/dataset_demo.hdf5', test_indices)
    
    collate_fn = partial(collate_fn_ndarray, target_treatment=target_treatment, target_task=target_task)
    sampler = RandomSampler(train_set)
    train_loader = DataLoader(train_set, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn)
    test_loader  = DataLoader(test_set, shuffle=False, batch_size=batch_size, collate_fn=collate_fn)
    return train_loader, test_loader


def collate_fn(batch, feature_index, treatment_index, task_index):
    # Using list comprehensions for efficiency
    features = [sample[feature_index] for sample in batch]
    treatments = [sample[treatment_index] for sample in batch]
    tasks = [sample[task_index] for sample in batch]

    # Convert to tensors
    features_tensor = torch.as_tensor(np.array(features), dtype=torch.float32)
    treatments_tensor = torch.as_tensor(np.array(treatments), dtype=torch.float32).squeeze()
    tasks_tensor = torch.as_tensor(np.array(tasks), dtype=torch.float32).squeeze()

    return features_tensor, treatments_tensor, tasks_tensor
    

def get_data(train_files, test_files, target_treatment, target_task, batch_size, dist=False, feature_group=None, addition_feat=None):
    with open('data/OUT_COLUMN', 'r') as f:
        labels = f.readlines()
    labels = [x.strip('\n') for x in labels]
    prefixes = [x.split('_')[0] for x in labels]
    
    if feature_group != None:
        ...
    else:
        feature_index = [idx for idx, elem in enumerate(prefixes) if elem[:3] == 'fea']
    treatment_index = [idx for idx, elem in enumerate(labels) if elem == 'treatment_next_iswarm']
    task_index = [idx for idx, elem in enumerate(labels) if elem == 'label_nextday_login']
    
    if addition_feat is not None:
        feature_index.extend(addition_feat)
    
    train_set = CustomDatasetHdf5MultiChunk(train_files, chunk_size=3840 * 16)
    test_set  = CustomDatasetHdf5MultiChunk(test_files,  chunk_size=3840 * 16)
    
    collate = partial(collate_fn, feature_index=feature_index, treatment_index=treatment_index, task_index=task_index)
    
    if dist:
        sampler = DistributedSampler(train_set, shuffle=False)
    
    # we cannot shuffle the dataset since hdf5 stores the data in chunks
    train_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size, collate_fn=collate, pin_memory=True, num_workers=16)
    test_loader  = DataLoader(test_set, shuffle=False, batch_size=batch_size, collate_fn=collate, pin_memory=True, num_workers=16)
    return train_loader, test_loader


class FakeData(Dataset):
    def __init__(self, x):
        self.data = np.random.randn(10000, 685)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
   

class CustomDatasetHdf5Multi(Dataset):
    # hdf5 custom dataset that handles multiple files
    def __init__(self, source_files: list):
        self.source_files = sorted(source_files)
        self.h5_file_handle = {}
        self.lengths = []
        
        for i, f in enumerate(source_files):
            with h5py.File(f, 'r') as file:
                self.lengths.append(len(file['data']))
        
    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.lengths[file_idx]:
            idx -= self.lengths[file_idx]
            file_idx += 1
        source_file = self.source_files[file_idx]
        
        if source_file not in self.h5_file_handle:
            self.h5_file_handle[source_file] = h5py.File(source_file, 'r')
        return self.h5_file_handle[source_file]['data'][idx].astype(np.float32)

    def __del__(self):
        for k in self.h5_file_handle:
            self.h5_file_handle[k].close()


class CustomDatasetHdf5MultiPreload(Dataset):
    """hdf5 custom dataset that handles multiple files and load the entire file into memory"""
    def __init__(self, source_files: list):
        self.source_files = sorted(source_files)
        self.data = {}
        self.lengths = []
        
        for i, f in enumerate(source_files):
            with h5py.File(f, 'r') as file:
                self.lengths.append(len(file['data']))
        
    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.lengths[file_idx]:
            if self.source_files[file_idx] in self.data and self.data[self.source_files[file_idx]] is not None:
                self.data[self.source_files[file_idx]] = None
                gc.collect()
            
            idx -= self.lengths[file_idx]
            file_idx += 1
        source_file = self.source_files[file_idx]
        
        if source_file not in self.data or self.data[source_file] is None:
            with h5py.File(source_file, 'r') as f:
                self.data[source_file] = f['data'][:]
            
        return self.data[source_file][idx].astype(np.float32)


class CustomDatasetHdf5MultiChunk(Dataset):
    """hdf5 dataset that non-blockingly pre-loads data in chunks"""
    def __init__(self, source_files: list, chunk_size: int = 3840*10):
        self.source_files = sorted(source_files)
        self.chunk_size = chunk_size
        self.h5_file_handle = {}
        self.lengths = []
        self.data = {}
        self.current_chunk = {}
        self.lock = threading.Lock()
        self.chunk_load_threads = {}

        for i, f in enumerate(source_files):
            with h5py.File(f, 'r') as file:
                self.lengths.append(len(file['data']))

    def __len__(self):
        return sum(self.lengths)

    def load_chunk(self, source_file, chunk_start, chunk_end):
        with self.lock:
            if source_file not in self.h5_file_handle:
                self.h5_file_handle[source_file] = h5py.File(source_file, 'r')
            self.data[source_file] = self.h5_file_handle[source_file]['data'][chunk_start:chunk_end]
            self.current_chunk[source_file] = (chunk_start, chunk_end)

    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.lengths[file_idx]:
            idx -= self.lengths[file_idx]
            file_idx += 1
        source_file = self.source_files[file_idx]

        chunk_start = (idx // self.chunk_size) * self.chunk_size
        chunk_end = min(chunk_start + self.chunk_size, self.lengths[file_idx])

        if source_file not in self.current_chunk or not (self.current_chunk[source_file][0] <= idx and idx  < self.current_chunk[source_file][1]):
            if source_file in self.chunk_load_threads:
                self.chunk_load_threads[source_file].join()

            self.chunk_load_threads[source_file] = threading.Thread(
                target=self.load_chunk, args=(source_file, chunk_start, chunk_end)
            )
            self.chunk_load_threads[source_file].start()
            self.chunk_load_threads[source_file].join()  # Ensure the chunk is loaded before accessing it

        data_idx = idx - self.current_chunk[source_file][0]
        return self.data[source_file][data_idx].astype(np.float32)
    
    def __del__(self):
        for k in self.h5_file_handle:
            self.h5_file_handle[k].close()


class CustomDatasetHdf5(Dataset):
    def __init__(self, source_dir: str, indices: list=None):
        self.source_dir = source_dir
        self.indices = indices  # if we need to split train/val in a single data file
        self.h5_file = None
        if self.indices is not None:
            self.length = len(indices)
        else:
            with h5py.File(self.source_dir, 'r') as file:
                self.dataset = file['data']
                self.length = len(self.dataset)
                self.indices = np.arange(self.length)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.source_dir, 'r')
            self.dataset = self.h5_file['data']
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx].astype(np.float32)
    
    def __del__(self):
        if self.h5_file:
            self.h5_file.close()
        

class CustomDataset(Dataset):
    def __init__(self, source_dir: str, is_train: bool, fold_id: int, num_folds: int, field_path='data/demo/OUT_COLUMN'):
        with open(field_path, 'r') as f:
            field = f.readlines()
        self.field = [x.strip('\n') for x in field]
        
        self.source_dir = source_dir
        self.is_train = is_train
        self.fold_id = fold_id
        self.num_folds = num_folds
        
        self.files = os.listdir(source_dir)
        self.files = sorted([f for f in self.files if f.startswith('part')])
        indices = np.arange(len(self.files))
        skf = KFold(n_splits=num_folds, shuffle=False)
        for i, (train_indices, test_indices) in enumerate(skf.split(indices)):
            if i == fold_id:
                break
        self.indices = train_indices if is_train else test_indices  # get indice of the file instead of the sample
        
        if os.path.isfile('num_samples.txt'):
            with open('num_samples.txt', 'r') as f:
                sample_sizes = [int(line.strip()) for line in f]
            self.sample_sizes = {i: sample_sizes[i] for i in self.indices}
            self.total_length = sum(list(self.sample_sizes.values()))
        else:
            self.sample_sizes, self.total_length = self._cache_sample_sizes()
        
        self.all_df = {}  # if memory can hold all data
        for i in self.indices:
            self.all_df[i] = load_one_gz_file(os.path.join(self.source_dir, self.files[i]), self.field)

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        current_idx = idx
        for i in self.indices:
            sample_size = self.sample_sizes[i]
            if current_idx < sample_size:
                if not i in self.all_df:
                    df = load_one_gz_file(os.path.join(self.source_dir, self.files[i]), self.field)
                    self.all_df[i] = df
                else:
                    df = self.all_df[i]
                row = df.iloc[current_idx]
                row = torch.tensor(row.values, dtype=torch.float32)         
                return row
            
            current_idx -= sample_size
        raise IndexError("Index out of range")

    def _cache_sample_sizes(self):
        lengths = {}
        total_length = 0
        for i in self.indices:
            file_path = os.path.join(self.source_dir, self.files[i])
            with gzip.open(file_path, 'rt') as f:
                lengths[i] = sum(1 for line in f)
            total_length += lengths[i]
        return lengths, total_length


class CustomDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn if collate_fn else self.default_collate
        self.queue = Queue(maxsize=num_workers * 2)
        self.processes = []

    def _worker(self, indices, queue):
        worker_name = current_process().name
        for idx in indices:
            queue.put(self.dataset[idx])
        queue.put(None)  # End-of-data signal

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        if self.num_workers > 0:
            chunk_size = (len(indices) + self.num_workers - 1) // self.num_workers
            for i in range(self.num_workers):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(indices))
                process = Process(target=self._worker, args=(indices[start_idx:end_idx], self.queue))
                process.start()
                self.processes.append(process)
        else:
            for idx in indices:
                self.queue.put(self.dataset[idx])
            self.queue.put(None)  # End-of-data signal

        return self

    def __next__(self):
        batch = []
        while len(batch) < self.batch_size:
            item = self.queue.get()
            if item is None:  # End-of-data signal
                if len(batch) == 0:
                    self._cleanup()
                    raise StopIteration
                else:
                    break
            batch.append(item)
        return self.collate_fn(batch)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _cleanup(self):
        for process in self.processes:
            process.join()
        self.processes = []

    def __del__(self):
        self._cleanup()

    def default_collate(self, batch):
        return torch.tensor(batch)


def get_all_sizes(source_dir):
    files = os.listdir(source_dir)
    files = sorted([f for f in files if f.startswith('part')])
    lengths = []
    for f in files:
        file_path = os.path.join(source_dir, f)
        with gzip.open(file_path, 'rt') as f:
            lengths.append(str(sum(1 for _ in f)) + '\n')
    with open('num_samples.txt', 'w') as f:
        f.writelines(lengths)
    

if __name__ == '__main__':
    from tqdm import tqdm
    train_loader, test_loader = get_data('data/demo', target_treatment=[536, 537, 538], target_task=[554, 555, 556], 
                                          fold_id=0, batch_size=1024)
    for batch in tqdm(train_loader):
        ...

