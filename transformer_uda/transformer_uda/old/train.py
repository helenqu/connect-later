import yaml
import argparse
from torch.utils.data import DataLoader

from transformer_uda.dataset import LightcurveDataset

def load_config(config_path):
    with open(config_path, "r") as cfgfile:
        config = yaml.load(cfgfile, Loader=yaml.Loader)
    return config

def get_args():
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
    parser.add_argument('--config_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config_path)

    dataset = LightcurveDataset(config['output_path'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    print(next(iter(dataloader)))
