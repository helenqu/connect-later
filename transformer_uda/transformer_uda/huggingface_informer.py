import datasets
import transformers
from datasets import load_dataset
from transformers import InformerConfig, InformerForPrediction, PretrainedConfig
from accelerate import Accelerator, DistributedDataParallelKwargs

import torch
from torch.optim import AdamW
from torchinfo import summary

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
from functools import partial

from transformer_uda.informer_models import InformerFourierPEForPrediction, MaskedInformerFourierPE
from transformer_uda.dataset_preprocess import create_train_dataloader
from transformer_uda.dataset_preprocess_raw import create_train_dataloader_raw
from transformer_uda.plotting_utils import plot_batch_examples

WANDB_DIR = "/pscratch/sd/h/helenqu/sn_transformer/wandb"
CACHE_DIR = "/pscratch/sd/h/helenqu/huggingface_datasets_cache"
CHECKPOINT_DIR = "/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/checkpoints"

def get_dataset(data_dir, data_subset_file=None):
    print(f"data subset file: {data_subset_file}")
    if data_subset_file is not None:
        print("data subset file not none")
        with open(data_subset_file) as f:
            data_subset = [x.strip() for x in f.readlines()]
    else:
        data_subset = [str(x) for x in Path(data_dir).glob("*.jsonl")] # this includes original training set
    print(f"using data subset: {data_subset}")
    dataset = load_dataset(data_dir, data_files={"train": data_subset}, cache_dir=CACHE_DIR)#, download_mode="force_redownload")
    print(f"loading dataset {'from file ' if data_subset_file is not None else ''}with {len(dataset['train'])} examples")

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

def prepare_model_input(batch, device, config, mask):
    model_inputs = {
            "past_time_features": batch['past_time_features'].to(device),
            "past_values": batch["past_values"].to(device),
            "past_observed_mask": batch["past_observed_mask"].to(device),
    }
    if config.num_static_categorical_features > 0:
        model_inputs["static_categorical_features"] = batch["static_categorical_features"].to(device)
    if config.num_static_real_features > 0:
        model_inputs["static_real_features"] = batch["static_real_features"].to(device)
    if not mask:
        model_inputs["future_time_features"] = batch["future_time_features"].to(device)
        model_inputs["future_observed_mask"] = batch["future_observed_mask"].to(device)
        model_inputs["future_values"] = batch["future_values"].to(device)
    else:
        model_inputs["labels"] = batch["mask_label"].to(device)

    return model_inputs

def train(args, base_config, add_config=None):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # accelerator = Accelerator()

    device = accelerator.device

    if args.log_level:
        print(f"setting log level to {args.log_level}")
        log_levels = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}
        transformers.logging.set_verbosity(log_levels[args.log_level])
        datasets.logging.set_verbosity(log_levels[args.log_level])
        datasets.logging.enable_propagation()

    if not args.dry_run and accelerator.is_main_process:
        print("initializing wandb")
        wandb.init(project="informer", name="masked", config=base_config, dir=WANDB_DIR) #mode="offline")
        add_config = wandb.config
    print(add_config)
    config = base_config
    if add_config is not None:
        config.update(add_config)
    print(config)

    dataset = get_dataset(args.data_dir, data_subset_file=config['data_subset_file'])

    model_config = InformerConfig(
        # in the multivariate setting, input_size is the number of variates in the time series per time step
        input_size=6 if not args.fourier_pe else 2,
        # prediction length:
        prediction_length=config['prediction_length'],
        # context length:
        context_length=config['context_length'],
        # lags value copied from 1 week before:
        lags_sequence=[0],
        # we'll add 5 time features ("hour_of_day", ..., and "age"):
        num_time_features=len(config['time_features']) + 1 if not args.fourier_pe else 2, #wavelength + time

        # informer params:
        dropout=config['dropout_rate'],
        encoder_layers=config['num_encoder_layers'],
        decoder_layers=config['num_decoder_layers'],
        # project input from num_of_variates*len(lags_sequence)+num_time_features to:
        d_model=config['d_model'],
        scaling=config['scaling'],
        has_labels=False,
        mask=args.mask,
        mask_probability=config['mask_probability'],
    )

    addl_config = {}
    # additional encoder/decoder hyperparams:
    if 'encoder_attention_heads' in config:
        addl_config['encoder_attention_heads'] = config['encoder_attention_heads']
    if 'decoder_attention_heads' in config:
        addl_config['decoder_attention_heads'] = config['decoder_attention_heads']
    if 'encoder_ffn_dim' in config:
        addl_config['encoder_ffn_dim'] = config['encoder_ffn_dim']
    if 'decoder_ffn_dim' in config:
        addl_config['decoder_ffn_dim'] = config['decoder_ffn_dim']
    # additional hyperparams for learnable fourier PE:
    if 'fourier_dim' in config:
        addl_config['fourier_dim'] = config['fourier_dim']
    if 'PE_hidden_dim' in config:
        addl_config['PE_hidden_dim'] = config['PE_hidden_dim']

    model_config.update(addl_config)

    if args.fourier_pe and args.mask:
        print("instantiating model with fourier PE and masking")
        model = MaskedInformerFourierPE(model_config)
        dataloader_fn = create_train_dataloader_raw
    elif args.fourier_pe:
        print("instantiating model with fourier PE")
        model = InformerFourierPEForPrediction(model_config)
        dataloader_fn = create_train_dataloader_raw
    else:
        print("instantiating model with GP-interpolated inputs")
        model = InformerForPrediction(model_config)
        dataloader_fn = create_train_dataloader
    print(model)
    print(f"num total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"num trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
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

    print(f"allow padding: {config['allow_padding']}, {type(config['allow_padding'])}")
    train_dataloader = dataloader_fn(
        config=model_config,
        dataset=dataset['train'],
        time_features=[month_of_year] if 'month_of_year' in config['time_features'] else None, # not used in raw version
        batch_size=config['batch_size'],
        num_batches_per_epoch=config['num_batches_per_epoch'],
        shuffle_buffer_length=1_000_000,
        allow_padding=config['allow_padding'],
        cache_data=False
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
            if idx == 0:
                print(batch['past_values'][0])
                print(batch['labels'][0])
            optimizer.zero_grad()

            outputs = model(**prepare_model_input(batch, device, model_config, args.mask))
            loss = outputs.loss

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            progress_bar.update(1)
            cumulative_loss += loss.item()

        print(f"epoch {epoch}: loss = {cumulative_loss / idx}")

        if epoch % 10 == 0:
            ckpt_dir = Path(CHECKPOINT_DIR) / f"checkpoint_{start_time.strftime('%Y-%m-%d_%H:%M:%S')}_epoch_{epoch}"
            print(f"saving ckpt at {ckpt_dir}")
            accelerator.save_state(output_dir=ckpt_dir)
        if not args.dry_run and accelerator.is_main_process:
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
    parser.add_argument("--fourier_pe", action="store_true")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--log_level", type=str)
    parser.add_argument("--lr", type=float)

    args = parser.parse_args()

    with open("/global/homes/h/helenqu/time_series_transformer/transformer_uda/configs/bigger_model_hyperparameters.yml") as f:
        config = yaml.safe_load(f)
    with open("/global/homes/h/helenqu/time_series_transformer/transformer_uda/hyperparameters.yml") as f:
        sweep_config = yaml.safe_load(f)

    config['num_epochs'] = args.num_epochs
    config['weight_decay'] = 0.01
    config['dropout_rate'] = 0.2
    config['lr'] = 0.0001
    config['batch_size'] = 1024
    config["data_subset_file"] = "/pscratch/sd/h/helenqu/plasticc/raw/plasticc_raw_examples/single_test_file.txt"
    # config["data_subset_file"] = "/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/examples/15_percent_filepaths.txt"
    config['scaling'] = None
    # config["context_length"] = 170
    # config["prediction_length"] = 10
    # config["num_epochs"] = args.num_epochs
    config["allow_padding"] = True
    config["mask_probability"] = 0.5

    train(args, config)
    # sweep_id = wandb.sweep(sweep_config, project="pretraining-fourier-sweep")
    # sweep_id = "helenqu/pretraining-20k-sweep/x52pxocd"
    # sweep_id = "helenqu/pretraining-all-sweep/w674xnm8"
    # wandb.agent(sweep_id, partial(train, args, config), count=5)
