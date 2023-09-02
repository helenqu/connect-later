import yaml
import argparse
import torch
import pandas as pd
from sklearn.metrics import f1_score
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator

from transformer_uda.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw
from transformer_uda.informer_models import InformerForSequenceClassification
from transformer_uda.huggingface_informer import get_dataset, setup_model_config

config_yml = "/global/homes/h/helenqu/time_series_transformer/transformer_uda/configs/bigger_model_hyperparameters.yml"
with open(config_yml, "r") as f:
    config = yaml.safe_load(f)

parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--model_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
parser.add_argument('--redshift', action='store_true', default=False, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
parser.add_argument('--mask_probability', type=float, default=0.6, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
args = parser.parse_args()

model_config = setup_model_config(args, config)
finetune_config = {
    "has_labels": False,
    "num_labels": 14,
    "regression": False,
    "classifier_dropout": 0.2,
    "fourier_pe": True,
    # "balance": config['balance'],
    "mask": True
}
model_config.update(finetune_config)

model = InformerForSequenceClassification.from_pretrained(args.model_path, config=model_config, ignore_mismatched_sizes=True)

test_dataset = get_dataset('/pscratch/sd/h/helenqu/plasticc/raw/plasticc_raw_examples')['train']
test_dataloader = create_test_dataloader_raw(
    config=model_config,
    dataset=test_dataset,
    batch_size=256,
    compute_loss=True,
    allow_padding=False,
    add_objid=True
)

accelerator = Accelerator(mixed_precision='bf16')
device = accelerator.device

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model, test_dataloader = accelerator.prepare(
    model,
    test_dataloader,
)

model.to(device)
model.eval()
pred_labels = []
objids = []
for i, batch in enumerate(tqdm(test_dataloader)):
    input_batch = {k: v.to(device) for k, v in batch.items() if k != "objid"}
    with torch.no_grad():
        outputs = model(**input_batch)
        predictions_for_batch = torch.argmax(outputs.logits, dim=-1)
        predictions = accelerator.gather(predictions_for_batch)
        objids_for_batch = accelerator.gather(batch['objid'])
        pred_labels.extend(predictions.cpu().numpy().flatten())
        objids.extend(objids_for_batch.cpu().numpy())

df = pd.DataFrame({'objid': objids, 'pred_label': pred_labels})
df.to_csv(f'/pscratch/sd/h/helenqu/plasticc/raw/plasticc_raw_examples/self_training_labels_{Path(args.model_path).parent.stem}.csv', index=False)
