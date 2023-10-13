import yaml
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.nn import MSELoss

from connect_later.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw
from connect_later.informer_models import InformerForSequenceClassification
from connect_later.pretrain import get_dataset, setup_model_config
from connect_later.constants import INT_LABELS

parser = argparse.ArgumentParser(description='Self-training baseline for ConnectLater')
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument('--model_path', type=str, help='path to saved model to use for pseudolabeling')
parser.add_argument('--labels_path', type=str, help='path to true labels for test set')
parser.add_argument('--output_path', type=str, help='path to save pseudolabels to')
parser.add_argument('--redshift_prediction', action='store_true', default=False, help='set if doing redshift prediction')
parser.add_argument('--mask_probability', type=float, default=0.)
args = parser.parse_args()

with open(args.config_path, "r") as f:
    config = yaml.safe_load(f)

model_config = setup_model_config(args, config)
finetune_config = {
    "has_labels": False,
    "num_labels": 14 if not args.redshift_prediction else 1,
    "regression": args.redshift_prediction,
    "classifier_dropout": 0.2,
    "fourier_pe": True,
    "mask": True,
}
model_config.update(finetune_config)
print(model_config)

model = InformerForSequenceClassification.from_pretrained(args.model_path, config=model_config, ignore_mismatched_sizes=True)

test_dataset = get_dataset(args.dataset_path)['train']
test_dataloader = create_test_dataloader_raw(
    config=model_config,
    dataset=test_dataset,
    batch_size=256,
    compute_loss=True,
    add_objid=True
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
model.eval()

labels = pd.read_csv(args.labels_path)
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
df.to_csv(Path(args.output_path) / "self_training_labels_{Path(args.model_path).parent.stem}.csv", index=False)
