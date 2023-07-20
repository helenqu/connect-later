import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import jsonlines
import pdb
from collections import Counter
import json

def split_jsonl_dataset_into_files(metadata_path, dataset_path, pattern, train_size, fraction=1.0, required_paths=[]):
    metadata = pd.read_csv(metadata_path)

    data_paths = list(Path(dataset_path).glob(pattern))
    data_paths += required_paths
    # if fraction < 1.0:
    #     data_paths = np.random.choice(list(data_paths), int(len(list(data_paths))*fraction), replace=False)
    #     if len(required_paths) > 0:
    #         print(f"Adding {len(required_paths)} required paths")
    #         data_paths = np.concatenate([data_paths, required_paths])
    #     print(f"taking {fraction*100}% of the data, {len(data_paths)} files")

    # num_ids = len(metadata['object_id'].unique())
    # train_ids = np.random.choice(metadata['object_id'].unique(), size=int(num_ids*train_size), replace=False)

    # remaining_ids = np.setdiff1d(metadata['object_id'].unique(), train_ids)
    # val_test_size = 1 - train_size
    # val_ids = np.random.choice(remaining_ids, size=int(num_ids*(val_test_size/2)), replace=False)
    # test_ids = np.setdiff1d(remaining_ids, val_ids)

    # print(f"selected {len(train_ids)} train ids, {len(val_ids)} val ids, {len(test_ids)} test ids")

    train = pd.DataFrame()
    val = pd.DataFrame()
    test = pd.DataFrame()

    data = pd.DataFrame()
    for path in data_paths:
        with open(path, "r") as f:
            lines = f.read().splitlines()
        df = pd.DataFrame(lines)
        df.columns = ['json_element']
        df['json_element'] = df['json_element'].apply(json.loads)
        df = pd.json_normalize(df['json_element'])
        data = pd.concat([data, df])

    if 'label' not in data.columns:
        data['object_id'] = pd.to_numeric(data['object_id'])

        data = data.merge(metadata, on='object_id', how='left')
        data = data[['object_id', 'times_wv', 'lightcurve', 'true_target']]
        data = data.rename(columns={'true_target': 'label'})

    for i in data['label'].unique():
        label_data = data[data['label'] == i]
        idxs_for_train = np.random.choice(range(len(label_data)), size=int(len(label_data)*train_size), replace=False)
        idxs_for_val_test = np.setdiff1d(range(len(label_data)), idxs_for_train)
        idxs_for_val = np.random.choice(idxs_for_val_test, size=int(len(idxs_for_val_test)/2), replace=False)
        idxs_for_test = np.setdiff1d(idxs_for_val_test, idxs_for_val)

        for_train = label_data.iloc[idxs_for_train]
        for_val = label_data.iloc[idxs_for_val]
        for_test = label_data.iloc[idxs_for_test]

        train = pd.concat([train, for_train])
        val = pd.concat([val, for_val])
        test = pd.concat([test, for_test])

    print(Counter(train['label']))
    train.to_json(dataset_path / 'train.jsonl', orient='records', lines=True)
    val.to_json(dataset_path / 'val.jsonl', orient='records', lines=True)
    test.to_json(dataset_path / 'test.jsonl', orient='records', lines=True)
        # with jsonlines.open(path) as reader:
        #     for obj in reader:
        #         if int(obj['object_id']) in train_ids:
        #             train.append(obj)
        #         elif int(obj['object_id']) in val_ids:
        #             val.append(obj)
        #         elif int(obj['object_id']) in test_ids:
        #             test.append(obj)

    # print(len(train), len(val), len(test))
    # print(f"writing train.jsonl, val.jsonl, test.jsonl to {dataset_path}")
    # with jsonlines.open(dataset_path / 'train.jsonl', 'w') as writer:
        # writer.write_all(train)
    # with jsonlines.open(dataset_path / 'val.jsonl', 'w') as writer:
        # writer.write_all(val)
    # with jsonlines.open(dataset_path / 'test.jsonl', 'w') as writer:
        # writer.write_all(test)

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

