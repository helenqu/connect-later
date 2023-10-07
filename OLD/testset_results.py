#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import InformerConfig, InformerForPrediction, AdamW
from datasets import load_metric
from accelerate import Accelerator
import torch
from torch.distributions import StudentT

from gluonts.time_feature import month_of_year

from transformer_uda.huggingface_informer import get_dataset
from transformer_uda.dataset_preprocess import create_test_dataloader, create_train_dataloader
from transformer_uda.informer_for_classification import InformerForSequenceClassification

import yaml
import numpy as np
import pdb
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import argparse

RESULTS_DIR = Path("/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/test_results")

# In[3]:
parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
parser.add_argument('--pretrained_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"', required=True)
parser.add_argument('--model_config', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"', required=True)
args = parser.parse_args()

accelerator = Accelerator()
device = accelerator.device

num_variates = 6

model = InformerForSequenceClassification.from_pretrained(args.pretrained_model)
model.to(device)
model.eval()

test_dataset = get_dataset('/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples')#, data_subset_file='/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples/single_test_file.txt') # random file from plasticc test dataset

with open(args.model_config, 'r') as f:
    config = yaml.safe_load(f)

model_config = InformerConfig(
    input_size=num_variates,
    prediction_length=10,#config['prediction_length'],
    context_length=170,#config['context_length'],
    lags_sequence=[0],
    num_time_features=len(config['time_features']) + 1,
    dropout=config['dropout_rate'],
    encoder_layers=config['num_encoder_layers'],
    decoder_layers=config['num_decoder_layers'],
    d_model=config['d_model'],
    has_labels=True
)

labels = pd.DataFrame()
for path in Path("/pscratch/sd/h/helenqu/plasticc/plasticc_test_gp_interp/").glob("labels*"):
    labels_tmp = pd.read_csv(path)
    labels = pd.concat([labels, labels_tmp])

labels = pd.concat([labels, pd.read_csv("/pscratch/sd/h/helenqu/plasticc/plasticc_train_gp_interp/labels_0.csv")])
#labels = pd.read_csv("/pscratch/sd/h/helenqu/plasticc/plasticc_test_gp_interp/labels_7.csv")
labels.loc[labels['label'] > 990, 'label'] = 99
# test_dataset['train'].add_column('labels', [1.0] * len(test_dataset['train']))

with open("/global/homes/h/helenqu/time_series_transformer/transformer_uda/transformer_uda/type_labels.yml") as f:
    label_str_map = yaml.safe_load(f)
    label_str_map = label_str_map['PLASTICC_CLASS_MAPPING']
label_str_map[99] = "extra"

INT_LABELS = [90, 67, 52, 42, 62, 95, 15, 64, 88, 92, 65, 16, 53, 6, 99]
unencoded_labels = labels[labels['object_id'].isin(test_dataset['train']['objid'])]['label'].values
encoded_labels = [INT_LABELS.index(i) for i in unencoded_labels]
test_dataset['train'] = test_dataset['train'].add_column('label', encoded_labels)

test_dataloader = create_test_dataloader(
    config=model_config,
    dataset=test_dataset['train'],
    time_features=[month_of_year if x == 'month_of_year' else None for x in config['time_features']],
    batch_size=config["batch_size"],
    #num_batches_per_epoch=1000,
    add_objid=True,
    compute_loss=True
    # num_workers=1,
)

pred = []
true = []

metric = load_metric('accuracy')
progress_bar = tqdm(range(len(test_dataset) // config["batch_size"]))

for i, batch in enumerate(test_dataloader):
        # outputs = model.generate(
    with torch.no_grad():
        #batch['labels'] = []
        ##TODO: probably need to add these labels to the dataset instead of here
        #for objid in batch['objid']:
        #    batch['labels'].append(INT_LABELS.index(labels[labels['object_id'] == int(objid)]['label'].values[0]))
        #batch['labels'] = torch.tensor(batch['labels'])
        outputs = model(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if model_config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if model_config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            future_values=batch["future_values"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
            future_observed_mask=batch["future_observed_mask"].to(device),
            labels=batch['labels'].to(device)
        )
        print(batch['labels'])

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch['labels'])
        pred.extend(predictions.cpu().numpy())
        true.extend(batch['labels'].cpu().numpy())
        progress_bar.update(1)

results = pd.DataFrame({'pred': pred, 'true': true})
results.to_csv(RESULTS_DIR / f"{Path(args.pretrained_model).stem}_results.csv")
