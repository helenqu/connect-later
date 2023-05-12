from datasets import load_dataset
from transformers import InformerConfig, InformerForPrediction, PretrainedConfig
from accelerate import Accelerator

import torch
from torch.optim import AdamW

from gluonts.time_feature import month_of_year

import pandas as pd
import numpy as np
import pdb
import wandb
from tqdm.auto import tqdm
import argparse
import yaml
from pathlib import Path
import shutil
from datetime import datetime

from transformer_uda.dataset_preprocess import create_train_dataloader
from transformer_uda.plotting_utils import plot_batch_examples

WANDB_DIR = "/pscratch/sd/h/helenqu/sn_transformer/wandb"
CACHE_DIR = "/pscratch/sd/h/helenqu/huggingface_datasets_cache"
CHECKPOINT_DIR = "/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/checkpoints"

def get_dataset(data_dir, data_subset_file=None):
    if data_subset_file is not None:
        with open(data_subset_file) as f:
            data_subset = [x.strip() for x in f.readlines()]
    else:
        data_subset = Path(data_dir).glob("*.jsonl")
    dataset = load_dataset(data_dir, data_files={"train": data_subset}, cache_dir=CACHE_DIR)#, download_mode='force_redownload')

    return dataset

def save_model(model, optimizer, output_dir):
    print(f"Saving model to {output_dir}")
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    torch_model_dir = Path(output_dir) / "torch_model"
    hf_model_dir = Path(output_dir) / "hf_model"

    print(f"overwriting torch model at {torch_model_dir}")
    if torch_model_dir.exists():
        shutil.rmtree(torch_model_dir)
    torch_model_dir.mkdir(parents=True)

    print(f"overwriting hf model at {hf_model_dir}")
    if hf_model_dir.exists():
        shutil.rmtree(hf_model_dir)
    hf_model_dir.mkdir(parents=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, torch_model_dir / "model.pt")

    model.save_pretrained(hf_model_dir)

def train(args, config=None):
    if not args.dry_run:
        wandb.init(project="informer", config=config, dir=WANDB_DIR)

    dataset = get_dataset(args.data_dir, data_subset_file=config['data_subset_file'])

    model_config = InformerConfig(
        # in the multivariate setting, input_size is the number of variates in the time series per time step
        input_size=6,
        # prediction length:
        prediction_length=config['prediction_length'],
        # context length:
        context_length=config['context_length'],
        # lags value copied from 1 week before:
        lags_sequence=[0],
        # we'll add 5 time features ("hour_of_day", ..., and "age"):
        num_time_features=len(config['time_features']) + 1,

        # informer params:
        dropout=config['dropout_rate'],
        encoder_layers=config['num_encoder_layers'],
        decoder_layers=config['num_decoder_layers'],
        # project input from num_of_variates*len(lags_sequence)+num_time_features to:
        d_model=config['d_model'],
        has_labels=False
    )

    accelerator = Accelerator()
    device = accelerator.device

    model = InformerForPrediction(model_config)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(config['lr']),
        betas=(0.9, 0.95),
        weight_decay=float(config['weight_decay'])
    )

    if args.load_model:
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    train_dataloader = create_train_dataloader(
        config=model_config,
        dataset=dataset['train'],
        time_features=[month_of_year] if 'month_of_year' in config['time_features'] else None,
        batch_size=config['batch_size'],
        num_batches_per_epoch=config['num_batches_per_epoch'],
        shuffle_buffer_length=1_000_000,
        allow_padding=config['allow_padding'],
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )
    if args.load_checkpoint:
        accelerator.load_state(args.load_checkpoint)

    num_training_steps = config['num_epochs'] * config['num_batches_per_epoch']
    progress_bar = tqdm(range(num_training_steps))

    start_time = datetime.now()
    model.train()
    for epoch in range(config['num_epochs']):
        cumulative_loss = 0
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
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
            loss = outputs.loss

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            progress_bar.update(1)
            cumulative_loss += loss.item()

        print(f"epoch {epoch}: loss = {cumulative_loss / idx}")

        if epoch % 100 == 0:
            ckpt_dir = Path(CHECKPOINT_DIR) / f"checkpoint_{start_time.strftime('%Y-%m-%d_%H:%M:%S')}_epoch_{epoch}"
            print(f"saving ckpt at {ckpt_dir}")
            accelerator.save_state(output_dir=ckpt_dir)
        if not args.dry_run:
            wandb.log({"loss": cumulative_loss / idx})

    if args.save_model:
        model = accelerator.unwrap_model(model)
        save_model(model, optimizer, args.save_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--save_model", type=str)
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--load_checkpoint", type=str)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    with open("/global/homes/h/helenqu/time_series_transformer/transformer_uda/configs/500k_best_hyperparams.yml") as f:
        config = yaml.safe_load(f)
    config["data_subset_file"] = "/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples/15_percent_filepaths.txt"
    config["context_length"] = 100
    config["prediction_length"] = 50
    config["num_epochs"] = args.num_epochs
    config["allow_padding"] = False

    train(args, config=config)
