from datasets import load_dataset, concatenate_datasets, load_metric
from transformers import InformerConfig, PretrainedConfig, AdamW, get_scheduler, set_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

import torch

import numpy as np
from pathlib import Path
import wandb
from tqdm.auto import tqdm
import pdb
import argparse
import yaml
import shutil
from collections import Counter

from gluonts.time_feature import month_of_year

from connect_later.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw
from connect_later.informer_models import InformerForSequenceClassification
from connect_later.pretrain import get_dataset, setup_model_config

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.last_validation_loss = 0

    def early_stop(self, validation_loss):
        self.last_validation_loss = validation_loss
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta) and validation_loss > self.last_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
    parser.add_argument('--config_path', type=str, help='path to model config yaml', required=True)
    parser.add_argument('--dataset_path', type=str, help='path to training data', required=True)
    parser.add_argument('--test_set_path', type=str, help='path to testing data', default=None)
    parser.add_argument('--load_model', type=str, help='path to pretrained model')
    parser.add_argument('--save_model', type=str, help='path to save model to')
    parser.add_argument('--dry_run', action='store_true', help='run without logging to wandb')
    parser.add_argument('--random_init', action='store_true', help='randomly initialize model')
    parser.add_argument('--class_weights', action='store_true')
    parser.add_argument('--num_lp_steps', type=int)
    parser.add_argument('--num_ft_steps', type=int)
    parser.add_argument('--lp_lr', type=float)
    parser.add_argument('--ft_lr', type=float)
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--redshift_prediction', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask_probability', type=float, default=0.6)
    parser.add_argument('--self_training', action='store_true')
    parser.add_argument('--orig_dataset_path', type=str)

    return parser.parse_args()

def cycle(dataloader):
    while True:
        for x in dataloader:
            yield x

def train_loop(
        model: InformerForSequenceClassification,
        model_config: InformerConfig,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        num_epochs: int,
        num_training_steps: int,
        val_interval=100,
        log_interval=100,
        early_stopping=False,
        dry_run=False,
        class_weights=None,
        save_model_path=None,
    ):

    progress_bar = tqdm(range(num_training_steps))
    abundances = Counter({k: 0 for k in range(15)})
    if early_stopping:
        early_stopper = EarlyStopper(patience=5, min_delta=0.05)

    best_test_loss = np.inf
    for idx, batch in enumerate(cycle(train_dataloader)):
        model.train()
        metric= load_metric("accuracy")
        if idx == num_training_steps:
            break
        optimizer.zero_grad()

        if class_weights is not None:
            batch['weights'] = torch.tensor(class_weights)
        if not model_config.regression:
            batch['labels'] = batch['labels'].type(torch.int64)
        input_batch = {k: v.to(device) for k, v in batch.items() if k != 'objid'}
        if idx == 0:
            print(f"batch contents: {input_batch.keys()}")
        outputs = model(**input_batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        progress_bar.update(1)

        logits = outputs.logits
        if model_config.regression: # redshift prediction
            predictions = logits.squeeze()
        elif model_config.num_labels == 1:
            predictions = torch.round(torch.sigmoid(logits))
        else:
            predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions.flatten() == input_batch['labels']).sum().div(len(input_batch['labels'])).item()

        if idx % log_interval == 0:
            metrics = {
                'loss': loss.item(),
                'accuracy': accuracy,
                'lr': lr_scheduler.get_last_lr()[0],
            }

            if idx % val_interval == 0:
                model.eval()
                print("validating", flush=True)
                val_loss, val_accuracy = validate(model, model_config, val_dataloader, device)

                if early_stopping and early_stopper.early_stop(val_loss):
                    print("Early stopping")
                    break

                print("testing")
                test_loss, test_accuracy = validate(model, model_config, test_dataloader, device)

                if test_loss < best_test_loss and save_model_path is not None:
                    best_test_loss = test_loss
                    save_model(model, optimizer, save_model_path)

                metrics.update({
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                })

            if not dry_run:
                wandb.log(metrics)

        lr_scheduler.step()

    return model

def run_training_stage(
        stage,
        model,
        model_config,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        config,
        dry_run=False,
        class_weights=None,
        save_model_path=None
    ):

    if class_weights is not None:
        print("Using class weights")
    else:
        print("Not using class weights")

    if stage == 'lp':
        # freeze pretrained weights
        print("LP, freezing pretrained weights")
        for name, param in model.named_parameters():
            if 'encoder' in name or 'decoder' in name:
                param.requires_grad = False
    else:
        # unfreeze all weights
        print("FT, unfreezing pretrained weights")
        for name, param in model.named_parameters():
            param.requires_grad = True

    print(f"weight decay: {config['weight_decay']}")
    print(f"learning rate: {config[f'{stage}_lr']}")
    optimizer = AdamW(model.parameters(), lr=float(config[f'{stage}_lr']), weight_decay=config['weight_decay'])

    # ckpt = torch.load("/pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/finetuned_classification/pretrained_500k_context_170/torch_model/model.pt")
    # model.load_state_dict(ckpt['model_state_dict'])
    # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    num_training_steps = config[f'num_{stage}_steps']
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision='bf16', kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    model.to(device)

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )

    return train_loop(
        model=model,
        model_config=model_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=config[f'{stage}_epochs'],
        num_training_steps=num_training_steps,
        early_stopping=(stage == 'ft'),
        dry_run=dry_run,
        class_weights=class_weights,
        save_model_path=save_model_path,
    ), optimizer

def validate(model, model_config, dataloader, device):
    cumulative_loss = 0
    correct = 0
    metric = load_metric("accuracy")
    abundances = Counter({k: 0 for k in range(15)})

    for idx, batch in enumerate(dataloader):
        # batch['labels'] = batch['labels'].type(torch.int64)
        batch = {k: v.to(device) for k, v in batch.items() if k != 'objid'}
        with torch.no_grad():
            outputs = model(**batch)

        cumulative_loss += outputs.loss.item()
        logits = outputs.logits
        if model_config.regression: # redshift prediction
            predictions = logits.squeeze()
        elif model_config.num_labels == 1:
            predictions = torch.round(torch.sigmoid(logits))
        else:
            predictions = torch.argmax(logits, dim=-1)
        correct += (predictions.flatten() == batch['labels']).sum().item()
        # print(abundances, weights, weighted_correct,(predictions.flatten() == batch['labels']).sum().item(),  (weights[batch['labels'].cpu().numpy()] * (predictions.flatten() == batch['labels'])).sum().item())
        abundances = Counter(batch['labels'].cpu().numpy())
        abundance_values = np.array([abundances[k] for k in range(15)])
        metric.add_batch(predictions=predictions, references=batch["labels"])
        if idx == 0 or idx == 15: # two random samples
            print(f"abundances: {abundances}, predictions: {predictions.flatten()[:30]}, true: {batch['labels'][:30]}")
            print(f"accuracy: {(predictions.flatten() == batch['labels']).sum().item() / len(batch['labels'])}, loss: {outputs.loss.item()}")
        # pdb.set_trace()

    return cumulative_loss / (idx+1), metric.compute()

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

def setup_model_config_finetune(args, config):
    print(config)
    # args.redshift is used inside here
    model_config = setup_model_config(args, config)

    finetune_config = {
        "has_labels": True,
        "num_labels": 1 if args.redshift_prediction else 14,
        "regression": args.redshift_prediction,
        "classifier_dropout": config['classifier_dropout'],
        "fourier_pe": True,
        "mask": True
    }
    model_config.update(finetune_config)

    return model_config

def train():
    args = parse_args()
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
    model_config = setup_model_config_finetune(args, config)

    if not args.dry_run:
        wandb.init(config=config, name=args.wandb_name, project="finetuning")

    if args.seed:
        set_seed(args.seed)

    print(f"loading dataset from {args.dataset_path}")
    dataset = load_dataset(str(args.dataset_path), download_mode='force_redownload')

    if args.self_training:
        # if self-training, add the original labeled data back in
        orig_train = load_dataset(args.orig_dataset_path)
        orig_train['train'] = orig_train['train'].remove_columns(['redshift'])
        print(f"original training dataset size: {len(orig_train['train'])}")
        dataset['train'] = concatenate_datasets([orig_train['train'], dataset['train']])
        dataset['validation'] = orig_train['validation']

    print(f"dataset sizes: {len(dataset['train'])}, {len(dataset['validation'])}")

    abundances = Counter(dataset['train']['label'])
    max_count = max(abundances.values())
    class_weights = [abundances[i] / sum(abundances.values()) for i in range(model_config.num_labels)] if args.class_weights else None # weight upsampled balanced dataset by abundance
    if args.class_weights:
        print(f"Class weights applied: {class_weights}")

    train_dataloader = create_train_dataloader_raw(
        config=model_config,
        dataset=dataset['train'],
        batch_size=config["batch_size"],
        seed=args.seed,
        add_objid=True,
    )
    val_dataloader = create_test_dataloader_raw(
        config=model_config,
        dataset=dataset['validation'],
        batch_size=config["batch_size"],
        compute_loss=True,# no longer optional for encoder-decoder latent space
        seed=args.seed,
    )

    test_dataset = get_dataset(args.test_set_path)['train'] if args.test_set_path is not None else dataset['test']
    no_anomalies_dataset = test_dataset.filter(lambda x: x['label'] < 14)

    test_dataloader = create_test_dataloader_raw(
        config=model_config,
        dataset=no_anomalies_dataset,
        batch_size=config["batch_size"],
        compute_loss=True,
        allow_padding=False,
        seed=args.seed,
    )

    if args.load_model is not None and not args.random_init:
        print(f"Loading model from {args.load_model}")
        model = InformerForSequenceClassification.from_pretrained(args.load_model, config=model_config)
    elif args.random_init:
        print("Randomly initializing model")
        model = InformerForSequenceClassification(model_config)
    print(model)
    print(f"num total parameters: {sum(p.numel() for p in model.parameters())}")

    model.train()
    if args.num_lp_steps > 0:
        model, _ = run_training_stage('lp', model, model_config, train_dataloader, val_dataloader, test_dataloader, config, dry_run=args.dry_run, class_weights=class_weights, save_model_path=args.save_model)
    model, optimizer = run_training_stage('ft', model, model_config, train_dataloader, val_dataloader, test_dataloader, config, dry_run=args.dry_run, class_weights=class_weights, save_model_path=args.save_model)

    if args.save_model:
        save_model(model, optimizer, args.save_model)

if __name__ == "__main__":
    train()
