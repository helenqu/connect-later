import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

def split_dataset_into_files(dataset_path, pattern, train_size, fraction=1.0, required_paths=[]):
    dataset_full = pd.DataFrame()
    print(f"Reading files from {dataset_path} matching {pattern}")
    data_paths = list(Path(dataset_path).glob(pattern))
    print(len(data_paths), int(len(data_paths)*fraction))
    if fraction < 1.0:
        data_paths = np.random.choice(list(data_paths), int(len(list(data_paths))*fraction), replace=False)
        if len(required_paths) > 0:
            print(f"Adding {len(required_paths)} required paths")
            data_paths = np.concatenate([data_paths, required_paths])
        print(f"taking {fraction*100}% of the data, {len(data_paths)} files")
    for path in tqdm(data_paths):
        dataset = pd.read_csv(path)
        dataset_full = pd.concat([dataset_full, dataset])

    # print(dataset_full)
    num_ids = len(dataset_full['object_id'].unique())
    train_ids = np.random.choice(dataset_full['object_id'].unique(), size=int(num_ids*train_size), replace=False)

    remaining_ids = np.setdiff1d(dataset_full['object_id'].unique(), train_ids)
    val_test_size = 1 - train_size
    val_ids = np.random.choice(remaining_ids, size=int(num_ids*(val_test_size/2)), replace=False)
    test_ids = np.setdiff1d(remaining_ids, val_ids)

    print(f"selected {len(train_ids)} train ids, {len(val_ids)} val ids, {len(test_ids)} test ids")

    train = dataset_full[dataset_full['object_id'].isin(train_ids)]
    val = dataset_full[dataset_full['object_id'].isin(val_ids)]
    test = dataset_full[dataset_full['object_id'].isin(test_ids)]

    print(f"writing train.csv, val.csv, test.csv to {dataset_path}")

    train.to_csv(Path(dataset_path) / 'train.csv', index=False)
    val.to_csv(Path(dataset_path) / 'val.csv', index=False)
    test.to_csv(Path(dataset_path) / 'test.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
    parser.add_argument('--dataset_path', type=str, help='absolute or relative path to your dataset directory')
    parser.add_argument('--pattern', type=str, help='pattern to match relevant files, i.e. "lc_*.csv"')
    parser.add_argument('--train_size', type=float, help='percentage to allocate to training')
    args = parser.parse_args()

    split_dataset_into_files(**vars(args))

