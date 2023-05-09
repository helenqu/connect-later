import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from datasets import load_dataset
from pathlib import Path
from torch.utils.data import DataLoader
import pdb

def plot_batch_examples(
    dataloader: DataLoader,
    labels_path: str = None, # TODO: add support for multiple label paths or reorganize the labels files
    num_examples: int = 10
):
    """Plot examples from a batch of data from a dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): Dataloader to plot examples from.
        n_examples (int, optional): Number of examples to plot. Defaults to 5."""
    batch = next(iter(dataloader))
    if labels_path is not None:
        type_labels = pd.read_csv(labels_path)

    cols = num_examples // 2
    fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(20, 8))
    cmap = plt.get_cmap('Set2')
    colors = cmap(np.linspace(0,1,8))
    bands = ['u', 'g', 'r', 'i', 'z', 'Y']
    for i in range(num_examples):
        ax = axes[i // cols, i % cols]
        for band_idx, band in enumerate(bands):
            past_row = batch['past_values'][i,:,band_idx]
            future_row = batch['future_values'][i,:,band_idx]
            ax.scatter(range(len(past_row)), past_row, label=band, color=colors[band_idx])
            ax.scatter(range(len(past_row), len(past_row)+len(future_row)), future_row, label=band, color=colors[band_idx], alpha=0.5)
        if labels_path is not None:
            objid = int(batch['objid'][i])
            label = type_labels[type_labels['object_id'] == objid].label.values[0]
            ax.set_title(f"objid: {objid}, type: {label}")
        if i == 0:
            ax.legend()
    plt.show()
    fig.savefig("/global/homes/h/helenqu/time_series_transformer/figures/batch_examples_with_padding.pdf", bbox_inches="tight")

def plot_gp_interp_examples(
    gp_interp_path: str,
    labels_path: str,
    raw_lcdata_path: str,
    num_examples: int = 10
):
    dataset = pd.read_csv(gp_interp_path)
    type_labels = pd.read_csv(labels_path)
    raw_lcdata = pd.read_csv(raw_lcdata_path)
    ids = np.random.choice(np.unique(type_labels['object_id']), size=num_examples, replace=False)

    type_label_mapping = _get_type_label_mapping()

    """Plots a few examples from the dataset."""
    cols = num_examples // 2
    fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(20, 8))
    cmap = plt.get_cmap('Set2')
    colors = cmap(np.linspace(0,1,8))

    for i in range(len(ids)):
        ax = axes[i // cols, i % cols]
        data = dataset[dataset['object_id'] == ids[i]]
        lcdata = raw_lcdata[raw_lcdata['object_id'] == ids[i]]
        bands = np.sort(np.unique(lcdata['passband']))
        for band_idx, band in enumerate(bands): # passbands are numbers here
            band_data = lcdata[lcdata['passband'] == band]
            ax.errorbar(band_data.mjd, band_data.flux, yerr=band_data.flux_err, label=band, fmt='o', color=colors[band_idx])
        for band_idx, wv in enumerate(np.sort(np.unique(data['wavelength']))):
            wv_data = data[data['wavelength'] == wv]
            ax.errorbar(wv_data.mjd, wv_data.flux, yerr=wv_data.flux_err, elinewidth=0.5, errorevery=10, label=bands[band_idx], color=colors[band_idx], alpha=0.8)
        ax.set_title(f"type: {type_label_mapping[type_labels[type_labels['object_id'] == ids[i]].label.values[0]]}")
        if i == 0:
            ax.legend()
    plt.show()

def _get_type_label_mapping():
    with open(Path(__file__).parent / 'type_labels.yml') as f:
        type_label_mapping = yaml.load(f, Loader=yaml.FullLoader)
    return type_label_mapping['PLASTICC_CLASS_MAPPING']

if __name__ == '__main__':
    ROOT_DIR = Path("/pscratch/sd/h/helenqu/plasticc/")
    plot_gp_interp_examples(ROOT_DIR / "plasticc_train_gp_interp" / 'preprocessed_0.csv', ROOT_DIR / "plasticc_train_gp_interp" / 'labels_0.csv', ROOT_DIR / 'plasticc_train_lightcurves.csv.gz')
