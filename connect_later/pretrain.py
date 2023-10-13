from datasets import load_dataset
from transformers import InformerConfig
from accelerate import Accelerator, DistributedDataParallelKwargs

import torch
from torch.optim import AdamW

import wandb
from tqdm.auto import tqdm
import argparse
import yaml
from pathlib import Path
import shutil
from datetime import datetime

from connect_later.informer_models import MaskedInformerFourierPE
from connect_later.dataset_preprocess_raw import create_train_dataloader_raw

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, required=True)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--save_model", type=str)
    parser.add_argument("--load_model", type=str)
    parser.add_argument("--load_checkpoint", type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--mask_probability", default=0.6, type=float)

    return parser.parse_args()

def get_dataset(data_dir, data_subset_file=None, force_redownload=False):
    kwargs = {}
    if data_subset_file is not None:
        with open(data_subset_file) as f:
            data_subset = [x.strip() for x in f.readlines()]
            print(f"using data subset: {data_subset}")

            kwargs["data_files"] = {'train': data_subset}
    if force_redownload:
        kwargs["download_mode"] = "force_redownload"

    dataset = load_dataset(data_dir, **kwargs)
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

def setup_model_config(args, config):
    # model config computes certain properties, can't config.update these
    model_config = InformerConfig(
        input_size=2,
        prediction_length=0,
        context_length=300,
        lags_sequence=[0],
        num_time_features=2, #wavelength + time
        num_static_real_features=0,

        # informer params:
        dropout=config['dropout_rate'],
        encoder_layers=config['num_encoder_layers'],
        decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        scaling=config['scaling'],
        has_labels=False,
        mask=True,
        mask_probability=args.mask_probability,
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

    return model_config

def train():
    args = parse_args()
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    model_config = setup_model_config(args, config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[ddp_kwargs])

    device = accelerator.device

    if not args.dry_run and accelerator.is_main_process:
        wandb.init(project="pretrainig", name=args.wandb_name, config=config)

    dataset = get_dataset(args.dataset_path)

    model = MaskedInformerFourierPE(model_config)
    print(model)
    print(f"num total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"num trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=float(args.lr if args.lr is not None else config['pretrain_lr']),
        betas=(0.9, 0.95),
        weight_decay=float(config['weight_decay'])
    )

    if args.load_model:
        ckpt = torch.load(args.load_model)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    train_dataloader = create_train_dataloader_raw(
        config=model_config,
        dataset=dataset['train'],
        batch_size=config['batch_size'],
    )

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )
    if args.load_checkpoint:
        accelerator.load_state(args.load_checkpoint)

    progress_bar = tqdm(range(config['num_steps']))

    def cycle(dataloader):
        while True:
            for x in dataloader:
                yield x

    start_time = datetime.now()
    model.train()
    for idx, batch in enumerate(cycle(train_dataloader)):
        if idx == config['num_steps']:
            break
        optimizer.zero_grad()

        outputs = model(**prepare_model_input(batch, device, model_config, args.mask))
        loss = outputs.loss

        # Backpropagation
        accelerator.backward(loss)
        optimizer.step()
        progress_bar.update(1)

        if not args.dry_run and accelerator.is_main_process:
            wandb.log({"loss": loss.item()})

        if idx % 1_000 == 0:
            print(f"step {idx}: loss = {loss.item()}")

        if idx % 5_000 == 0:
            ckpt_dir = Path(args.save_model) / f"checkpoint_{args.wandb_name}_{start_time.strftime('%Y-%m-%d_%H:%M:%S')}_step_{idx}"
            print(f"saving ckpt at {ckpt_dir}")
            accelerator.save_state(output_dir=ckpt_dir)

    if args.save_model:
        model = accelerator.unwrap_model(model)
        save_model(model, optimizer, args.save_model)

if __name__ == "__main__":
    train()
