import os
import h5py
import numpy as np
import gzip
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from multiprocessing import Pool


def convert_to_parquet(input_dir, output_file):
    field_path = 'data/demo/OUT_COLUMN'
    with open(field_path, 'r') as f:
            field = f.readlines()
    field = [x.strip('\n') for x in field]

    # Create an empty list to store dataframes
    dataframes = []
    writer = None

    # Iterate over all files in the directory
    filenames = sorted(os.listdir(input_dir))

    for filename in tqdm(filenames):
        if filename.endswith(".gz"):
            file_path = os.path.join(input_dir, filename)
            # df = pd.read_csv(file_path, sep='\t', compression='gzip', names=field, index_col=False, header=None,
            #                   na_values='NULL')
            reader = pd.read_csv(file_path, sep='\t', compression='gzip', names=field, index_col=False, header=None, na_values='NULL', chunksize=1000)
            for chunk in reader:
                table = pa.Table.from_pandas(chunk)
                if writer is None:
                       writer = pq.ParquetWriter(output_file, table.schema)
                writer.write_table(table)
    if writer:
        writer.close()
        
    
def group_files_into_bins(source_dir, num_bins=10):
    # Get file sizes
    filenames = [f for f in os.listdir(source_dir) if f.endswith('.gz')]
    file_paths = [os.path.join(source_dir, f) for f in filenames]
    files_with_sizes = [(file, os.path.getsize(file)) for file in file_paths]
    
    # Sort files by size (largest first)
    files_with_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize bins
    bins = [[] for _ in range(num_bins)]
    bin_sizes = [0] * num_bins
    
    # Distribute files into bins using a greedy approach
    for file, size in files_with_sizes:
        # Find the bin with the smallest current size
        smallest_bin_index = bin_sizes.index(min(bin_sizes))
        bins[smallest_bin_index].append(file)
        bin_sizes[smallest_bin_index] += size
    
    return bins


def downsample_untreated_login(df, col1=537, col2=555):
    # Downsample the untreated but nextday login samples (majority)
    # to the number of treated by no nextday login (minority)
    combination_counts = df.groupby([col1, col2]).size()
    min_count = combination_counts.min()

    downsampled_dfs = []
    for combination, count in combination_counts.items():
        filtered_df = df[(df[col1] == combination[0]) & (df[col2] == combination[1])]
        downsampled_df = filtered_df.sample(n=min_count, random_state=42)
        downsampled_dfs.append(downsampled_df)
    result_df = pd.concat(downsampled_dfs)
    return result_df


def process_bin(bin, output_file, k, downsample=None):
    with h5py.File(output_file + f'{k}.hdf5', 'w') as h5f:
        first_file = True
        dset = None
        for file_path in tqdm(bin):
            df = pd.read_csv(file_path, sep='\t', compression='gzip', index_col=False, header=None, na_values='NULL')
            if downsample is not None:
                df = downsample(df)
            df = df.sample(frac=1, random_state=42)
            df = np.array(df)
            if first_file:
                dset = h5f.create_dataset('data', data=df, maxshape=(None, df.shape[1]), chunks=(3840, 685))
                first_file = False
            else:
                dset.resize(dset.shape[0] + df.shape[0], axis=0)
                dset[-df.shape[0]:] = df


def convert_to_h5py(input_dir, output_file, bins=None, downsample=None, num_processes=2):
    if bins is not None:
        filenames = bins
    else:
        filenames = sorted(os.listdir(input_dir))
        files_per_fold = len(filenames // 10)
    
    # Prepare arguments for multiprocessing
    args = [(bin, output_file, k, downsample) for k, bin in enumerate(filenames)]
    
    # Use multiprocessing to process each bin
    with Pool(num_processes) as pool:
        pool.starmap(process_bin, args)
        
    
def stratify_login_days(input_dir):
    # splite the test data accoridng to the login days (597th row)
    sampled_data = {'backflow': [], 'lowactive': [], 'midactive': [], 'highactive': []}
    with h5py.File(input_dir + 'dataset_0.hdf5', 'r') as file:
        data_size = len(file['data'])
        
        for i in range(0, data_size, 3840):
            sample = file['data'][i: i+3840]
            sampled_data['backflow'].append(sample[sample[:, 597] == 0])
            sampled_data['lowactive'].append(sample[(1 <= sample[:, 597]) & (sample[:, 597] <= 4)])
            sampled_data['midactive'].append(sample[(5 <= sample[:, 597]) & (sample[:, 597] <= 9)])
            sampled_data['highactive'].append(sample[(10 <= sample[:, 597]) & (sample[:, 597] <= 14)])
            
    for k in sampled_data:
        with h5py.File(input_dir + f'dataset_{k}_0.hdf5', 'w') as h5f:
            dataset = h5f.create_dataset('data', data=np.concatenate(sampled_data[k], axis=0), chunks=(3840, 685))
        

if __name__ == '__main__':
    input_dir = 'data/testdata_240412_240528_minmax/'
    out_file = 'data/traindata_240119_240411_minmax/dataset_'
    # bins = group_files_into_bins(input_dir)
    # convert_to_h5py(input_dir, out_file, bins, downsample=downsample_untreated_login)
    stratify_login_days(input_dir)
