from datasets import load_dataset, concatenate_datasets
import torch
from torch.utils.data import Dataset
import pdb
import sys
import os
import numpy as np

# use local copy of wilds
sys.path.insert(1, os.path.join(sys.path[0], '/path/to/targeted_augs')) # replace with path to clone of https://github.com/i-gao/targeted-augs
import wilds
from wilds.datasets.wilds_dataset import WILDSSubset

from targeted_augs.examples.data_augmentation.transforms import initialize_transform

CACHE_DIR = "/path/to/cache/dir"

def get_astro_dataset():
    test = load_dataset("/path/to/astroclassification/test", cache_dir=CACHE_DIR)['train']
    train_dataset = load_dataset("/path/to/astroclassification/train", cache_dir=CACHE_DIR)
    train = concatenate_datasets([train_dataset['train'], train_dataset['validation'], train_dataset['test']])
    return train, test

def get_wilds_dataset(dataset_name):
    return wilds.get_dataset(
        dataset=dataset_name,
        root_dir=f"/path/to/wilds/datasets/{dataset_name}",
    )

def get_wilds_subset(dataset, indices, config, grouper):
    train_transform = initialize_transform(
        base_transforms=config.train_base_transforms,
        additional_transforms=config.train_additional_transforms,
        transform_p=config.transform_p,
        config=config,
        dataset=dataset,
        grouper=grouper,
        is_training=True)

    return WILDSSubset(dataset, indices, train_transform)

def get_across_domain_dataset(ID_ds, OOD_ds, ID_class, OOD_class, config=None, grouper=None):
    # across domain: ID and OOD class are the same
    if hasattr(ID_ds, 'column_names'):
        ID_ds_for_class = ID_ds.filter(lambda x: x["label"] == ID_class)
        OOD_ds_for_class = OOD_ds.filter(lambda x: x["label"] == OOD_class)
        num_ID_samples = len(ID_ds_for_class)
        num_OOD_samples = len(OOD_ds_for_class)
        print(f"astro dataset has {num_ID_samples} train samples of class {ID_class} and {num_OOD_samples} test samples of class {OOD_class}, balancing now...")

        # oversample the smaller class
        num_indices_to_sample = max(len(ID_ds_for_class), len(OOD_ds_for_class))
        ID_indices = np.random.choice(len(ID_ds_for_class), num_indices_to_sample, replace=True)
        OOD_indices = np.random.choice(len(OOD_ds_for_class), num_indices_to_sample, replace=True)
        print(f"sampled {num_indices_to_sample} indices from each dataset")
        ID_ds_for_class = ID_ds_for_class.select(ID_indices)
        OOD_ds_for_class = OOD_ds_for_class.select(OOD_indices)
    else: # passed in full dataset as ds1
        # get train and test splits manually
        ID_indices = np.where(ID_ds.split_array == ID_ds.split_dict['train'])[0]
        OOD_indices = np.where(ID_ds.split_array == ID_ds.split_dict['test'])[0]
        ID_class_indices = np.argwhere(ID_ds.y_array == ID_class)
        OOD_class_indices = np.argwhere(ID_ds.y_array == OOD_class)
        # get intersection of class indices and train/test (domain) indices
        ID_indices = np.intersect1d(ID_indices, ID_class_indices)
        OOD_indices = np.intersect1d(OOD_indices, OOD_class_indices)
        num_ID_samples = len(ID_indices)
        num_OOD_samples = len(OOD_indices)

        print(f"iwildcam dataset has {num_ID_samples} train samples of class {ID_class} and {num_OOD_samples} test samples of class {OOD_class}, balancing now...")
        # oversample the smaller class
        num_indices_to_sample = max(len(ID_indices), len(OOD_indices))
        ID_indices = np.random.choice(ID_indices, num_indices_to_sample, replace=True)
        OOD_indices = np.random.choice(OOD_indices, num_indices_to_sample, replace=True)
        ID_ds_for_class = get_wilds_subset(ID_ds, ID_indices, config, grouper)
        OOD_ds_for_class = get_wilds_subset(ID_ds, OOD_indices, config, grouper)

    return num_ID_samples, num_OOD_samples, ConnectivityDataset(ID_ds_for_class, OOD_ds_for_class)

def get_across_class_dataset(ds, _, class_1, class_2, config=None, grouper=None):
    if hasattr(ds, 'column_names'):
        ds_for_class_1 = ds.filter(lambda x: x["label"] == class_1)
        ds_for_class_2 = ds.filter(lambda x: x["label"] == class_2)
        num_class_1_samples = len(ds_for_class_1)
        num_class_2_samples = len(ds_for_class_2)
        print(f"astro dataset has {num_class_1_samples} samples of class {class_1} and {num_class_2_samples} samples of class {class_2}, balancing now...")

        # oversample the smaller class
        num_indices_to_sample = max(len(ds_for_class_1), len(ds_for_class_2))
        class_1_indices = np.random.choice(len(ds_for_class_1), num_indices_to_sample, replace=True)
        class_2_indices = np.random.choice(len(ds_for_class_2), num_indices_to_sample, replace=True)
        ds_for_class_1 = ds_for_class_1.select(class_1_indices)
        ds_for_class_2 = ds_for_class_2.select(class_2_indices)
    else: # passed in full dataset as ds1
        # get train and test splits manually
        ID_indices = np.where(ds.split_array == ds.split_dict['train'])[0]
        class_1_indices = np.argwhere(ds.y_array == class_1)
        class_2_indices = np.argwhere(ds.y_array == class_2)
        # get intersection of class indices and domain indices
        class_1_indices = np.intersect1d(ID_indices, class_1_indices)
        class_2_indices = np.intersect1d(ID_indices, class_2_indices)
        num_class_1_samples = len(class_1_indices)
        num_class_2_samples = len(class_2_indices)

        print(f"iwildcam dataset has {num_class_1_samples} samples of class {class_1} and {num_class_2_samples} samples of class {class_2}, balancing now...")
        # oversample the smaller class
        num_indices_to_sample = max(len(class_1_indices), len(class_2_indices))
        class_1_indices = np.random.choice(class_1_indices, num_indices_to_sample, replace=True)
        class_2_indices = np.random.choice(class_2_indices, num_indices_to_sample, replace=True)
        ds_for_class_1 = get_wilds_subset(ds, class_1_indices, config, grouper)
        ds_for_class_2 = get_wilds_subset(ds, class_2_indices, config, grouper)

    return num_class_1_samples, num_class_2_samples, ConnectivityDataset(ds_for_class_1, ds_for_class_2)

# define custom dataset following https://github.com/p-lambda/unlabeled_extrapolation/blob/main/unlabeled_extrapolation/datasets/connectivity_utils/data.py#L18, which labels 0 or 1 based on what you input
class ConnectivityDataset(Dataset):
    def __init__(self, ID_ds, OOD_ds):
        if hasattr(ID_ds, "features"):
            self.dataset = 'astro'
        else:
            self.dataset = 'iwildcam'

        if self.dataset == 'astro':
            self.ID_samples = {"target": ID_ds["target"], "times_wv": ID_ds["times_wv"], "label": [0] * len(ID_ds)}
            self.OOD_samples = {"target": OOD_ds["target"], "times_wv": OOD_ds["times_wv"], "label": [1] * len(OOD_ds)}
            self.samples = {
                "target": self.ID_samples['target'] + self.OOD_samples['target'],
                "times_wv": self.ID_samples['times_wv'] + self.OOD_samples['times_wv'],
                "label": self.ID_samples['label'] + self.OOD_samples['label']
            }
            print(f"dataset has {len(self.ID_samples['label'])} ID samples and {len(self.OOD_samples['label'])} OOD samples")
        else:
            self.ID_samples = [(x[0], 0) for x in ID_ds]
            self.OOD_samples = [(x[0], 1) for x in OOD_ds]
            self.samples = self.ID_samples + self.OOD_samples
            print(f"dataset has {len(self.ID_samples)} ID samples and {len(self.OOD_samples)} OOD samples")

    def __len__(self):
        return len(self.samples) if self.dataset != 'astro' else len(self.samples['target'])

    def __getitem__(self, idx):
        return self.samples[idx] if self.dataset != 'astro' else {key: value[idx] for key, value in self.samples.items()}
