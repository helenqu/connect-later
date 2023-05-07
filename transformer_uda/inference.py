from transformers import InformerConfig, InformerForPrediction, AdamW
from accelerate import Accelerator

from gluonts.time_feature import month_of_year

from transformer_uda.huggingface_informer import get_dataset
from transformer_uda.dataset_preprocess import create_test_dataloader

import yaml

accelerator = Accelerator()
device = accelerator.device

num_variates = 6

model = InformerForPrediction.from_pretrained('/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/model_500k_pretrained')
model.to(device)
model.eval()

test_dataset = get_dataset('/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples', data_subset_file='/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples/single_test_file.txt') # random file from plasticc test dataset

with open("configs/500k_best_hyperparams.yml", 'r') as f:
    config = yaml.safe_load(f)

model_config = InformerConfig(
    input_size=num_variates,
    has_labels=False,
    prediction_length=config['prediction_length'],
    context_length=config['context_length'],
    lags_sequence=[0],
    num_time_features=len(config['time_features']) + 1,
    dropout=config['dropout_rate'],
    encoder_layers=config['num_encoder_layers'],
    decoder_layers=config['num_decoder_layers'],
    d_model=config['d_model']
)

test_dataloader = create_test_dataloader(
    config=model_config,
    dataset=test_dataset,
    time_features=[month_of_year if x == 'month_of_year' else None for x in config['time_features']],
    batch_size=config["batch_size"],
    # num_workers=1,
)

forecasts = []

for batch in test_dataloader:
    outputs = model.generate(
        static_categorical_features=batch["static_categorical_features"].to(device)
        if model_config.num_static_categorical_features > 0
        else None,
        static_real_features=batch["static_real_features"].to(device)
        if model_config.num_static_real_features > 0
        else None,
        past_time_features=batch["past_time_features"].to(device),
        past_values=batch["past_values"].to(device),
        future_time_features=batch["future_time_features"].to(device),
        past_observed_mask=batch["past_observed_mask"].to(device),
    )

    forecasts.append(outputs.sequences.cpu().numpy())

forecasts = np.concatenate(forecasts, axis=0)
print(forecasts.shape)
