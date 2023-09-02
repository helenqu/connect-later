import yaml
import argparse
import torch
from sklearn.metrics import f1_score

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
    "has_labels": True,
    "num_labels": 1,
    "regression": False,
    "classifier_dropout": 0.2,
    "fourier_pe": True,
    # "balance": config['balance'],
    "mask": True
}
model_config.update(finetune_config)

model = InformerForSequenceClassification.from_pretrained(args.model_path, config=model_config, ignore_mismatched_sizes=True)

test_dataset = get_dataset('/pscratch/sd/h/helenqu/sdss/dataset')['test']
test_dataloader = create_test_dataloader_raw(
    config=model_config,
    dataset=test_dataset,
    batch_size=256,
    compute_loss=True,
    allow_padding=False,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)
model.eval()
y_true = []
y_pred = []
for i, batch in enumerate(test_dataloader):
    print(i)
    batch = {k: v.to(device) for k, v in batch.items() if k != "objid"}
    with torch.no_grad():
        outputs = model(**batch)
    y_true.extend(batch['labels'].cpu().numpy())
    y_pred.extend(torch.round(torch.sigmoid(outputs.logits)).squeeze().cpu().numpy())
    if i == 0:
        print(y_true, y_pred)
    if i % 10 == 0:
        print(f"batch {i}")

print(f1_score(y_true, y_pred))
