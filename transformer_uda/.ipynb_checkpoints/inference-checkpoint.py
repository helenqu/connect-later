from transformers import InformerConfig, InformerForPrediction, AdamW
from accelerate import Accelerator
import torch
from torch.distributions import StudentT

from gluonts.time_feature import month_of_year

from transformer_uda.huggingface_informer import get_dataset
from transformer_uda.dataset_preprocess import create_test_dataloader, create_train_dataloader

import yaml
import numpy as np
import pdb
import matplotlib.pyplot as plt

accelerator = Accelerator()
device = accelerator.device

num_variates = 6

model = InformerForPrediction.from_pretrained('/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_500k_context_100_pred_50_no_pad/hf_model')
model.to(device)
model.eval()

test_dataset = get_dataset('/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples', data_subset_file='/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples/single_test_file.txt') # random file from plasticc test dataset

with open("configs/500k_best_hyperparams.yml", 'r') as f:
    config = yaml.safe_load(f)

model_config = InformerConfig(
    input_size=num_variates,
    has_labels=False,
    prediction_length=50,#config['prediction_length'],
    context_length=100,#config['context_length'],
    lags_sequence=[0],
    num_time_features=len(config['time_features']) + 1,
    dropout=config['dropout_rate'],
    encoder_layers=config['num_encoder_layers'],
    decoder_layers=config['num_decoder_layers'],
    d_model=config['d_model']
)

test_dataloader = create_train_dataloader(
    config=model_config,
    dataset=test_dataset['train'],
    time_features=[month_of_year if x == 'month_of_year' else None for x in config['time_features']],
    batch_size=config["batch_size"],
    num_batches_per_epoch=100,
    # num_workers=1,
)

forecasts = []

for batch in test_dataloader:
    # outputs = model.generate(
    with torch.no_grad():
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
        )
    dist = StudentT(*outputs.params)
    mean = dist.mean * outputs.scale + outputs.loc
    variance = dist.variance * outputs.scale ** 2

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, 6)]
    for i in range(6):
        plt.plot(range(0, 100), batch['past_values'][0, :, i].cpu().numpy(), color=colors[i])
        plt.plot(range(100, 150), batch['future_values'][0, :, i].cpu().numpy(), label=f'true {i}', color=colors[i])
        plt.plot(range(100, 150), mean[0, :, i].cpu().numpy(), color=colors[i], linestyle='--')
        stdev = np.sqrt(variance[0, :, i].cpu().numpy())
        plt.fill_between(range(100, 150), mean[0, :, i].cpu().numpy()-stdev, mean[0, :, i].cpu().numpy()+stdev, alpha=0.2, color=colors[i])
    plt.legend()
    plt.show()

    # forecasts.append(outputs.sequences.cpu().numpy())

forecasts = np.concatenate(forecasts, axis=0)
print(forecasts.shape)
