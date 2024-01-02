from pathlib import Path
import numpy as np
import argparse
import yaml
import pdb
from datasets import Dataset
import torch
import sys
import os

from connectivity_utils import get_across_domain_dataset, get_across_class_dataset, get_astro_dataset, get_wilds_dataset
from connect_later.huggingface_informer import setup_model_config
from targeted_augs.examples.data_augmentation.transforms import _parse_transform_str
from train import train, train_astro
from wilds.common.grouper import CombinatorialGrouper

parser = argparse.ArgumentParser(description='Compute connectivity across domains')
parser.add_argument('--across_domain', action='store_true', help='Compute connectivity across domain')
parser.add_argument('--across_class', action='store_true', help='Compute connectivity across class')
parser.add_argument('--dataset', type=str, help='Dataset name (one of: [astro, camelyon17, iwildcam])')
parser.add_argument('--resume', type=int, help='Resume from a particular class-domain pair index')
parser.add_argument('--append', action='store_true', help='Append to existing class-domain pair list')
# defaults
parser.add_argument('--redshift', action='store_true', help='Use redshift for classification')
parser.add_argument('--mask_probability', type=float, default=0.6, help='masking percentage for MAE pretraining')
parser.add_argument('--seed', type=float, default=1, help='random seed')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--num_steps', type=int, default=3000, help='max number of training steps')
parser.add_argument('--num_pairs', type=int, default=5, help='number of class-domain pairs to compute connectivity for')
# args for targeted augs interfacing
parser.add_argument('--groupby_fields', nargs='+', default=['location'])
parser.add_argument('--train_base_transforms',  type=_parse_transform_str, default='image_base')
parser.add_argument('--eval_base_transforms',  type=_parse_transform_str, default='image_base')
parser.add_argument('--train_additional_transforms', type=_parse_transform_str, default='swav')
parser.add_argument('--transform_p', type=float, default=0.5)
args = parser.parse_args()

# iwildcam code expects to access config fields with dot notation
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

ROOT_DIR = Path(".")
DATASETS_DIR = Path("/path/to/datasets")
config_yml = "/path/to/config.yml"

with open(config_yml, "r") as f:
    config = yaml.safe_load(f)
model_config = setup_model_config(args, config)
model_config.update({
    "has_labels": True,
    "num_labels": 2,
})
config['batch_size'] = 32
config['to_tensor'] = True
config['target_resolution'] = (448, 448)
config['dataset'] = args.dataset
for arg in vars(args):
    config[arg] = getattr(args, arg)

if args.dataset != 'astro':
    config = dotdict(config)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

connectivity_type = 'across_domain' if args.across_domain else 'across_class'
if args.across_domain and args.across_class:
    connectivity_type = 'across_both'
classes_file = f'{connectivity_type}_classes_{args.dataset}.txt'

if args.dataset == 'iwildcam':
    # classes with more than 1 thing in both train and test
    avail_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 52, 59, 61, 65, 67, 68, 69, 70, 71, 72, 73, 74, 77, 79, 83, 85, 87, 88, 89, 90, 93, 95, 96, 97, 98, 100, 102, 106, 107, 108, 112, 113, 114, 115, 116, 117, 118, 119, 120, 126, 127, 133, 135, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 155, 158, 171, 172, 173, 176, 177]
    config.model = 'resnet50'
    config.model_kwargs = {}
    config.lr = 1.2352813497608926e-05
    config.weight_decay = 0.05
elif args.dataset == 'astro':
    avail_classes = range(14)
elif args.dataset == 'camelyon17':
    avail_classes = range(2)
    config.groupby_fields = ['hospital']
    config.model = 'densenet121'
    config.model_kwargs = {'pretrained': False}
    config.lr = 0.003898422290069297
    config.weight_decay = 0.01
    config.target_resolution = (96, 96)
    config.batch_size = 128
else:
    raise ValueError('Invalid dataset')

if args.dataset != 'camelyon17':
    # select classes in each domain to measure connectivity for
    prng = np.random.RandomState()
    first_classes = prng.choice(avail_classes, size=min(len(avail_classes), args.num_pairs), replace=False)
    if args.across_class:
        second_classes = prng.choice(avail_classes, size=min(len(avail_classes), args.num_pairs), replace=False)
        classes = [f"{first} {second}" for first, second in zip(first_classes, second_classes) if first != second]
    else:
        second_classes = first_classes
        classes = [f"{first} {second}" for first, second in zip(first_classes, second_classes)]
else:
    # camelyon only has two classes
    classes = ['0 1'] if args.across_class else ['0 0', '1 1']

if args.append:
    assert (ROOT_DIR / classes_file).exists()
    with open(ROOT_DIR / classes_file, 'a') as f:
        f.write('\n')
        f.write('\n'.join(classes))
elif (ROOT_DIR / classes_file).exists():
    # ignore generated classes if classes_file exists and we're not adding more seeds (args.append)
    with open((ROOT_DIR / classes_file), "r") as f:
        classes = f.read().splitlines()
else:
    # write space-separated list of classes to file
    with open(ROOT_DIR / classes_file, "w") as f:
        f.write("\n".join(classes))
    print(f"Saved classes to {classes_file}")

if args.resume:
    classes = classes[args.resume:]
print("Classes:", classes)

set_seed(args.seed)

# get dataset
if args.dataset != 'astro':
    print(f"Loading {args.dataset} dataset")
    dataset = get_wilds_dataset(args.dataset)
    print(config.groupby_fields)
    grouper = CombinatorialGrouper(
        dataset=dataset,
        groupby_fields=config.groupby_fields
    )
else:
    domain_1, domain_2 = get_astro_dataset()

for class_idxs in classes:
    first_class, second_class = class_idxs.split()
    first_class = int(first_class)
    second_class = int(second_class)
    config['class1'] = first_class
    config['class2'] = second_class

    if args.across_domain:
        connectivity_dataset_fn = get_across_domain_dataset
    elif args.across_class:
        connectivity_dataset_fn = get_across_class_dataset
    else:
        raise ValueError('at least one of across_domain or across_class must be true')

    if args.dataset != 'astro':
        print(f"getting {args.dataset} dataset for classes {first_class} and {second_class}")
        num_class_1_samples, num_class_2_samples, connectivity_dataset = connectivity_dataset_fn(
            dataset,
            None,
            first_class,
            second_class,
            config=config,
            grouper=grouper
        )

        print(f"splitting {args.dataset} dataset")
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            connectivity_dataset,
            [
                int(0.8 * len(connectivity_dataset)),
                int(0.1 * len(connectivity_dataset)),
                len(connectivity_dataset) - int(0.8 * len(connectivity_dataset)) - int(0.1 * len(connectivity_dataset))
            ],
            generator=generator
        )

        print("launching training")
        config['num_class_1_samples'] = num_class_1_samples
        config['num_class_2_samples'] = num_class_2_samples
        train(config, train_ds, val_ds, test_ds, connectivity_type)

    else:
        num_class_1_samples, num_class_2_samples, dataset = connectivity_dataset_fn(domain_1, domain_2, first_class, second_class)
        dataset = Dataset.from_list(dataset)
        dataset = dataset.shuffle(seed=config['seed']).flatten_indices()
        train_test_dict = dataset.train_test_split(test_size=0.1)
        train_val_dict = train_test_dict['train'].train_test_split(test_size=0.1)

        print("launching training")
        config['num_class_1_samples'] = num_class_1_samples
        config['num_class_2_samples'] = num_class_2_samples
        train_astro(config, model_config, train_val_dict['train'], train_val_dict['test'], train_test_dict['test'], connectivity_type)

