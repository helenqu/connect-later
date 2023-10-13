import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path

class LightcurveDataset(Dataset):
    def __init__(self, path_to_data, transform=None, target_transform=None):
        labels_paths = Path(path_to_data).glob('labels*.csv')
        lc_paths = Path(path_to_data).glob('preprocessed*.csv')

        self.labels = pd.DataFrame()
        self.lightcurves = pd.DataFrame()
        for labels_path, lc_path in zip(labels_paths, lc_paths):
            labels = pd.read_csv(labels_path)
            self.labels = pd.concat([self.labels, labels], ignore_index=True)
            lightcurves = pd.read_csv(lc_path)
            self.lightcurves = pd.concat([self.lightcurves, lightcurves], ignore_index=True)

        #TODO: maybe group by objectid then wavelength once
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sn = self.labels.iloc[idx]
        label = sn['label']
        lc = self.lightcurves[self.lightcurves['object_id'] == sn['object_id']]

        lc_grouped = lc.groupby('wavelength')
        groups = list(lc_grouped.groups.items())
        lc_shape = (len(groups), len(groups[0][1]))
        lc_tensor = np.zeros(lc_shape)

        lc_grouped_sorted = lc_grouped.apply(lambda x: x.sort_values('mjd').reset_index(drop=True))

        for i in range(len(np.unique(lc_grouped_sorted['wavelength']))):#, group in enumerate(lc_grouped.groups.keys()):
            # lc_tensor[i] = lc_grouped.get_group(group).flux.values
            lc_tensor[i] = lc_grouped_sorted.iloc[i*lc_shape[1]:(i+1)*lc_shape[1]].flux.values

        lc_tensor = torch.from_numpy(lc_tensor)

        if self.transform:
            lc_tensor = self.transform(lc_tensor)
        if self.target_transform:
            label = self.target_transform(label)
        return lc_tensor, label
