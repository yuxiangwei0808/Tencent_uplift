import os
import gc
from functools import partial
import gzip
import h5py
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, DistributedSampler, Subset
import torch
from sklearn.model_selection import KFold, StratifiedKFold
import threading
from multiprocessing import Process, Queue, current_process


def create_folds(file_paths, n_folds=5):
    if n_folds == 1:
        return [(file_paths, file_paths)]
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

    if len(task_index) == 3:  # multi-task, where the second should be loginday_diff
        # binarize
        tasks[:, 1][tasks[:, 1] > 0] = 1.
        tasks[:, 1][tasks[:, 1] <= 0] = 0.

    # Convert to tensors
    features_tensor = torch.as_tensor(features, dtype=torch.float32)
    treatments_tensor = torch.as_tensor(treatments, dtype=torch.float32).squeeze()
    tasks_tensor = torch.as_tensor(tasks, dtype=torch.float32)
    if group_discrete:
        if pad:
            grouped_disc_feature = [batch[:, indices] for indices in feature_groups.groups.values()]
            # pad differnt groups features to the same length according to group_area
            grouped_disc_feature = [np.pad(f, ((0, 0), (0, 35 - f.shape[-1])), 'constant', constant_values=0)[:, None, :] for f in grouped_disc_feature]
            grouped_disc_feature = np.concatenate(grouped_disc_feature, axis=1)  # B 9 35
            grouped_disc_feature = torch.as_tensor(grouped_disc_feature, dtype=torch.float)
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
        'group_district': list(range(722, 726)),
        'group_area': list(range(726, 761)),
        'group_r': list(range(761, 770)),
        # 'group_rank5': list(range(121, 184)),
    }
    indices = list(range(685, 770))
    # indices += list(range(121, 184))

def get_data(train_files, test_files, target_treatment, target_task, batch_size=3840, dist=False, feature_group=None, addition_feat=None):
    with open('data/train_test_data/OUT_COLUMN_new', 'r') as f:
        labels = f.readlines()
    labels = [x.strip('\n') for x in labels]
    
    treatment_index = [idx for idx, elem in enumerate(labels) if elem in target_treatment]
    task_index = [idx for idx, elem in enumerate(labels) if elem in target_task]
    # task_index.append(371)  # pre30_login_days, used for additional regularization of MTMT
    
    feature_index = [idx for idx, elem in enumerate(labels) if elem[:3] == 'fea' and 'login_days' not in elem]  # also exclude other login_days features
    # feature_index = [idx for idx, elem in enumerate(labels) if elem[:3] == 'fea']
    if feature_group != None:
        pad = True if 'pad' in feature_group else False
        feature_index = list(filter(lambda i: i not in list(range(537, 607)) + feature_groups.indices, feature_index))
        collate = partial(collate_fn, feature_index=feature_index, treatment_index=treatment_index, task_index=task_index, group_discrete=True, pad=pad)
    else:
        # feature_index = [idx for idx, elem in enumerate(labels) if elem[:3] == 'fea']
        feature_index = list(filter(lambda i: i not in list(range(537, 607)), feature_index))  # filter out columns that are used for AI accompany
        collate = partial(collate_fn, feature_index=feature_index, treatment_index=treatment_index, task_index=task_index)
    
    if addition_feat is not None:
        feature_index.extend(addition_feat)
    
    train_set = CustomDatasetHdf5MultiChunk(train_files, chunk_size=3840 * 16)
    test_set  = CustomDatasetHdf5MultiChunk(test_files,  chunk_size=3840 * 16)
    
    if dist:
        sampler = DistributedSampler(train_set, shuffle=False)
    
    # we cannot shuffle the dataset since hdf5 stores the data in chunks
    train_loader = DataLoader(train_set, shuffle=False, batch_size=batch_size, collate_fn=collate, pin_memory=True, num_workers=16)
    test_loader  = DataLoader(test_set, shuffle=False, batch_size=batch_size, collate_fn=collate, pin_memory=True, num_workers=16)
    return train_loader, test_loader


def get_data_public(batch_size, fold_idx, file_path, target_idx=None, treat_idx=None):
    dataset = DatasetPublic(file_path, target_idx, treat_idx)
    
    skf = StratifiedKFold(n_splits=5)
    indices = np.arange(len(dataset))
    treat = dataset.treat[:, 0] if dataset.treat.dim() > 1 and dataset.treat.shape[1] > 1 else dataset.treat
    for i, (train_indices, test_indices) in enumerate(skf.split(indices, treat)):
        if i == fold_idx:
            train_set = Subset(dataset, train_indices)
            test_set  = Subset(dataset, test_indices)

            train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=16, drop_last=True)
            test_loader  = DataLoader(test_set, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=16, drop_last=True)
            return train_loader, test_loader


class DatasetPublic(Dataset):
    def __init__(self, file_path, target_idx=None, treat_idx=None,):
        data_file = np.load(file_path, allow_pickle=True)
        self.data = data_file['data'].astype(np.float32)
        self.treat = data_file['treat'].astype(np.float32)
        self.target = data_file['target'].astype(np.float32)

        self.data = torch.as_tensor(self.data)
        self.treat =  torch.as_tensor(self.treat)
        self.target = torch.as_tensor(self.target).unsqueeze(-1)

        self.target = self.target[:, target_idx] if target_idx != None else self.target
        self.treat = self.treat[:, target_idx] if treat_idx != None else self.treat
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.treat[idx], self.target[idx]


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

