import yaml
import argparse
import torch
import pdb
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from torch.nn import MSELoss

from transformer_uda.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw
from transformer_uda.informer_models import InformerForSequenceClassification
from transformer_uda.huggingface_informer import get_dataset, setup_model_config
from transformer_uda.constants import INT_LABELS

config_yml = "/global/homes/h/helenqu/time_series_transformer/transformer_uda/configs/bigger_model_hyperparameters.yml"
with open(config_yml, "r") as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--model_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
parser.add_argument('--redshift_prediction', action='store_true', default=False, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
parser.add_argument('--redshift', action='store_true', default=False, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
parser.add_argument('--mask_probability', type=float, default=0., help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
args = parser.parse_args()

model_config = setup_model_config(args, config)
finetune_config = {
    "has_labels": False,
    "num_labels": 14 if not args.redshift_prediction else 1,
    "regression": args.redshift_prediction,
    "classifier_dropout": 0.2,
    "fourier_pe": True,
    # "balance": config['balance'],
    "mask": True,
}
model_config.update(finetune_config)
print(model_config)

model = InformerForSequenceClassification.from_pretrained(args.model_path, config=model_config, ignore_mismatched_sizes=True)

test_dataset = get_dataset('/pscratch/sd/h/helenqu/plasticc/raw/plasticc_raw_examples')['train']
# test_dataset = get_dataset('/pscratch/sd/h/helenqu/plasticc/raw_test_with_labels')['train']
# test_dataset = test_dataset.filter(lambda x: x['label'] < 14)
test_dataloader = create_test_dataloader_raw(
    config=model_config,
    dataset=test_dataset,
    batch_size=256,
    compute_loss=True,
    allow_padding=False,
    add_objid=True
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
model.eval()

labels = pd.read_csv('/pscratch/sd/h/helenqu/plasticc/raw/plasticc_test_metadata.csv.gz')
labels = pd.concat([labels, pd.read_csv('/pscratch/sd/h/helenqu/plasticc/raw/plasticc_train_metadata.csv.gz')], ignore_index=True)
if args.redshift_prediction:
    labels = labels[['object_id', 'true_z']].rename(columns={'true_z': 'label'})
else:
    labels.loc[labels['true_target'] > 990, 'true_target'] = 99
    labels = labels[['object_id', 'true_target']].rename(columns={'true_target': 'label'})
labels = labels.set_index('object_id')

pred_labels = []
objids = []
for i, batch in enumerate(tqdm(test_dataloader)):
    input_batch = {k: v.to(device) for k, v in batch.items() if k != "objid"}
    with torch.no_grad():
        outputs = model(**input_batch)
        predictions_for_batch = torch.argmax(outputs.logits, dim=-1) if not args.redshift_prediction else outputs.logits.squeeze()
        # predictions = accelerator.gather(predictions_for_batch)
        # labels = accelerator.gather(batch["labels"])
        if i % 100 == 0:
            labels_for_batch = labels.loc[batch['objid']]['label'].values
            if args.redshift_prediction:
                labels_for_batch = torch.tensor(labels_for_batch).to(device)
                print(f"loss: {MSELoss()(predictions_for_batch, labels_for_batch)}")
            else:
                int_labels = [INT_LABELS.index(label) for label in labels_for_batch]
                print(f"accuracy: {np.sum(predictions_for_batch.flatten().cpu().numpy() == int_labels) / len(predictions_for_batch)}", flush=True)
            # print(f"accuracy: {np.sum(predictions_for_batch.cpu().numpy() == batch['labels'].cpu().numpy()) / len(predictions_for_batch)}")
        pred_labels.extend(predictions_for_batch.cpu().numpy().flatten())
        objids.extend(batch['objid'])

df = pd.DataFrame({'objid': objids, 'pred_label': pred_labels})
df.to_csv(f'/pscratch/sd/h/helenqu/plasticc/raw/{"redshift_" if args.redshift_prediction else ""}self_training_labels_{Path(args.model_path).parent.stem}.csv', index=False)
