from datasets import load_dataset, load_metric
from transformers import InformerConfig, InformerForPrediction, PretrainedConfig, AdamW, get_scheduler#, Trainer, TrainingArguments

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from pathlib import Path
import wandb
from tqdm.auto import tqdm
from functools import partial
import pdb
import argparse

from gluonts.time_feature import month_of_year

from transformer_uda.dataset_preprocess import create_train_dataloader, create_transformation, convert_to_pandas_period, transform_start_field
from transformer_uda.informer_for_classification import InformerForSequenceClassification


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def normalize_data(example):
    # normalize data
    example["target"] = example["target"] / np.max(example["target"])
    return example

def train(args, config=None):
    if not args.dry_run:
        wandb.init(config=config, name='finetune-classification-500k-pretrained-lr_1e-4-context-170', project='sn-transformer')
        # config = wandb.config

    model_config = InformerConfig(
        # in the multivariate setting, input_size is the number of variates in the time series per time step
        input_size=6,
        # prediction length:
        prediction_length=config['prediction_length'],
        # context length:
        context_length=config['context_length'],
        # lags value copied from 1 week before:
        lags_sequence=config['lags'],
        time_features=config['time_features'],
        num_time_features=len(config['time_features']) + 1,
        dropout=config['dropout_rate'],
        encoder_layers=config['num_encoder_layers'],
        decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        has_labels=True,
        num_labels=15,
        classifier_dropout=0.1,
    )

    dataset = load_dataset(str(config['dataset_path']))#, download_mode='force_redownload')

    train_dataloader = create_train_dataloader(
        config=model_config,
        dataset=dataset['train'],
        time_features=config['time_features'],
        batch_size=config["batch_size"],
        num_batches_per_epoch=config["num_batches_per_epoch"],
        shuffle_buffer_length=1_000_000,
    )
    val_dataloader = create_train_dataloader(
        config=model_config,
        dataset=dataset['validation'],
        time_features=config['time_features'],
        batch_size=config["batch_size"],
        num_batches_per_epoch=config["num_batches_per_epoch"],
        shuffle_buffer_length=1_000_000,
    )

    model = InformerForSequenceClassification.from_pretrained(config['model_path'], config=model_config)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config['lr'])

    if args.load_finetuned_model:
        ckpt = torch.load(args.load_finetuned_model)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    num_training_steps = config['num_epochs'] * config['num_batches_per_epoch']
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))
    metric= load_metric("accuracy")
    save_model_path = Path(args.save_model)

    model.train()
    for epoch in range(config['num_epochs']):
        cumulative_loss = 0
        correct = 0
        for idx, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            cumulative_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions.flatten() == batch['labels']).sum().item()

        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_model_path.parent / "torch_checkpoints" / f"ckpt_{epoch}.pt") # load via torch.load

        if not args.dry_run:
            print(correct)
            accuracy = (100 * correct) / (config['num_batches_per_epoch'] * config['batch_size'])
            wandb.log({
                'loss': cumulative_loss / (idx+1),
                'accuracy': accuracy,
            })

    if not save_model_path.exists():
        save_model_path.mkdir(parents=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, save_model_path / "model.pt") # load via torch.load

    model.eval()
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    print(metric.compute())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
    parser.add_argument('--num_epochs', type=int, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"', required=True)
    parser.add_argument('--load_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"', required=True)
    parser.add_argument('--load_finetuned_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--save_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"', required=True)
    parser.add_argument('--dry_run', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    args = parser.parse_args()

    DATASET_PATH = Path('/pscratch/sd/h/helenqu/plasticc/train_with_labels')
    train(args, config={
        'prediction_length': 10,
        'context_length': 20,
        'time_features': [month_of_year],
        'dataset_path': str(DATASET_PATH),
        'model_path': args.load_model,
        'batch_size': 32,
        'num_batches_per_epoch': 1000,
        'num_epochs': args.num_epochs,
        "dropout_rate": 0.2,
        "num_encoder_layers": 11,
        "num_decoder_layers": 7,
        "lags": [0],
        "d_model": 128,
        "freq": "1M",
        'lr': 1e-4,
        })
