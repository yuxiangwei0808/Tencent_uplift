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


def create_folds(file_paths, n_folds=5):
    sub_datasets = [(file_paths[i], file_paths[i+1]) for i in range(0, len(file_paths), 2)]
    kf = KFold(n_splits=n_folds, shuffle=False)
    
    folds = []
    for train_index, test_index in kf.split(sub_datasets):
        train_files = [file for idx in train_index for file in sub_datasets[idx]]
        test_files = [file for idx in test_index for file in sub_datasets[idx]]
        folds.append((train_files, test_files))
    return folds


def collate_fn(batch, feature_index, treatment_index, task_index, group_discrete=False, pad=False):
    # Using list comprehensions for efficiency
    batch = np.stack(batch, axis=0)
    features = batch[:, feature_index]
    treatments = batch[:, treatment_index]
    tasks = batch[:, task_index]

    # Convert to tensors
    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    treatments_tensor = torch.as_tensor(treatments, dtype=torch.float32).squeeze()
    tasks_tensor = torch.as_tensor(tasks, dtype=torch.float32).squeeze()
    
    if group_discrete:
        if pad:
            grouped_disc_feature = [batch[:, indices] for indices in feature_groups.groups.values()]
            # pad differnt groups features to the same length according to group_area
            grouped_disc_feature = [np.pad(f, ((0, 0), (0, 35 - f.shape[-1])), 'constant', constant_values=0)[:, None, :] for f in grouped_disc_feature]
            grouped_disc_feature = np.concatenate(grouped_disc_feature, axis=1)  # B 9 35
            grouped_disc_feature = torch.as_tensor(grouped_disc_feature, dtype=torch.float32)
        else:
            grouped_disc_feature = torch.as_tensor(batch[:, feature_groups.indices])
        
        return (features_tensor, grouped_disc_feature), treatments_tensor, tasks_tensor

    return features_tensor, treatments_tensor, tasks_tensor


class feature_groups:
    groups = {
        'group_weekday': list(range(685, 692)),
        'group_time': list(range(692, 700)),
        'group_gender': list(range(700, 704)),
        'group_camp': [704, 705],
        'group_grade': list(range(706, 717)),
        'group_lane': list(range(717, 722)),
        # TODO encode district by nn.Embedding
        'group_district': list(range(722, 726)),
        'group_area': list(range(726, 761)),
        'group_r': list(range(761, 770)),
    }
    indices = list(range(685, 770))


def get_data(train_files, test_files, target_treatment=None, target_task=None, batch_size=3840, dist=False, feature_group=None, addition_feat=None, pad=False):
    with open('data/train_test_data/OUT_COLUMN_new', 'r') as f:
        labels = f.readlines()
    labels = [x.strip('\n') for x in labels]
        
    treatment_index = [idx for idx, elem in enumerate(labels) if elem == 'treatment_next_iswarm']
    task_index = [idx for idx, elem in enumerate(labels) if elem == 'label_nextday_login']
    
    feature_index = [idx for idx, elem in enumerate(labels) if elem[:3] == 'fea']
    if feature_group != None:
        feature_index = list(filter(lambda i: i not in list(range(537, 607)) + feature_groups.indices, feature_index))
        collate = partial(collate_fn, feature_index=feature_index, treatment_index=treatment_index, task_index=task_index, group_discrete=True, pad=pad)
    else:
        # feature_index = [idx for idx, elem in enumerate(labels) if elem[:3] == 'fea']
        feature_index = list(filter(lambda i: i not in list(range(537, 607)), feature_index))  # filter out columns that are used for AI accompany
        collate = partial(collate_fn, feature_index=feature_index, treatment_index=treatment_index, task_index=task_index)
    
    if addition_feat is not None:  # return additional features that can be used for further analysis
        feature_index.extend(addition_feat)
    
    train_set = CustomDatasetHdf5MultiChunk(train_files, chunk_size=3840 * 16)
    test_set  = CustomDatasetHdf5MultiChunk(test_files,  chunk_size=3840 * 16)
    
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

