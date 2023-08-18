from datasets import load_dataset, load_metric, concatenate_datasets
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
import yaml
import shutil
from collections import Counter

from gluonts.time_feature import month_of_year

from transformer_uda.dataset_preprocess import create_train_dataloader, create_test_dataloader
from transformer_uda.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw, create_network_inputs
from transformer_uda.informer_models import InformerForSequenceClassification
from transformer_uda.huggingface_informer import get_dataset, setup_model_config

WANDB_DIR = "/pscratch/sd/h/helenqu/sn_transformer/wandb/finetuning"
CACHE_DIR = "/pscratch/sd/h/helenqu/huggingface_datasets_cache"
CHECKPOINTS_DIR = "/pscratch/sd/h/helenqu/plasticc_all_gp_interp/models/finetuning_checkpoints"

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

def linear_probe_sklearn(model, model_config, train_dataloader, device, batch_size):
    num_training_batches = 1_000
    progress_bar = tqdm(range(num_training_batches))

    from sklearn.linear_model import LogisticRegression
    from sklearn import preprocessing

    X = np.zeros((batch_size*num_training_batches, model_config.d_model))
    y = np.zeros((batch_size*num_training_batches,))

    print("preparing training data for logreg linear probe")
    # for idx, batch in enumerate(train_dataloader):
    #     if idx == num_training_batches:
    #         break

    #     transformer_inputs, loc, scale, static_feat = create_network_inputs(
    #         config=model_config,
    #         past_values=batch['past_values'].to(device),
    #         past_time_features=batch['past_time_features'].to(device),
    #         past_observed_mask=batch['past_observed_mask'].to(device),
    #         static_categorical_features=batch['static_categorical_features'].to(device) if 'static_categorical_features' in batch else None,
    #         static_real_features=batch['static_real_features'].to(device) if 'static_real_features' in batch else None,
    #     )

    #     outputs = model.encoder(inputs_embeds=transformer_inputs)
    #     pooled_output = torch.mean(outputs.last_hidden_state, dim=1, keepdim=True)

    #     X[idx*batch_size:(idx+1)*batch_size] = pooled_output.squeeze().detach().cpu().numpy()
    #     y[idx*batch_size:(idx+1)*batch_size] = batch['labels'].cpu().numpy()

    #     progress_bar.update(1)

    # np.save("/pscratch/sd/h/helenqu/plasticc/linear_probe_X.npy", X)
    # np.save("/pscratch/sd/h/helenqu/plasticc/linear_probe_y.npy", y)
    X = np.load("/pscratch/sd/h/helenqu/plasticc/linear_probe_X.npy")
    y = np.load("/pscratch/sd/h/helenqu/plasticc/linear_probe_y.npy")

    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    for c in [0.01, 0.1, 1, 10, 100]:
        linear_classifier = LogisticRegression(
            multi_class='multinomial',
            random_state=0,
            C=c
        )
        linear_classifier.fit(X_scaled, y)
        print(f"linear probe with C={c} has score {linear_classifier.score(X, y)}")
    return linear_classifier.coef_

def cycle(dataloader):
    while True:
        for x in dataloader:
            yield x

def train_loop(
        model: InformerForSequenceClassification,
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
        # if idx % 100 == 0:
            # print(f"training: {batch['labels']}")
        if class_weights is not None:
            batch['weights'] = torch.tensor(class_weights)
        input_batch = {k: v.to(device) for k, v in batch.items() if k != 'objid'}
        if idx == 0:
            print(f"batch contents: {input_batch.keys()}")
        outputs = model(**input_batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        progress_bar.update(1)

        # cumulative_loss += loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        # correct += (predictions.flatten() == batch['labels']).sum().item()
        # abundances += Counter(batch['labels'].cpu().numpy())

        # accuracy = (100 * correct) / (idx * batch['labels'].shape[0])
        accuracy = (predictions.flatten() == input_batch['labels']).sum().div(len(input_batch['labels'])).item()
        # if save_ckpts and cumulative_loss < best_loss:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss
        #     }, save_model_path / "best_model.pt") # load via torch.load

        if idx % log_interval == 0:
            metrics = {
                'loss': loss.item(),
                # 'accuracy': metric.compute(predictions=predictions, references=batch["labels"]),
                'accuracy': accuracy,
                'lr': lr_scheduler.get_last_lr()[0],
            }

            if idx % val_interval == 0:
                model.eval()
                print("validating", flush=True)
                val_loss, val_accuracy, _ = validate(model, val_dataloader, device)

                if early_stopping and early_stopper.early_stop(val_loss):
                    print("Early stopping")
                    break

                print("testing")
                test_loss, test_accuracy, weighted_test_accuracy = validate(model, test_dataloader, device)

                if test_loss < best_test_loss and save_model_path is not None:
                    best_test_loss = test_loss
                    save_model(model, optimizer, save_model_path)

                metrics.update({
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'weighted_test_accuracy': weighted_test_accuracy,
                })

            if not dry_run:
                wandb.log(metrics)

        lr_scheduler.step()

    return model

def run_training_stage(stage, model, train_dataloader, val_dataloader, test_dataloader, config, device, dry_run=False, class_weights=None, save_model_path=None):
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

    return train_loop(
        model=model,
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

def validate(model, dataloader, device):
    cumulative_loss = 0
    correct = 0
    weighted_correct = 0
    metric = load_metric("accuracy")
    abundances = Counter({k: 0 for k in range(15)})

    for idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items() if k != 'objid'}
        with torch.no_grad():
            outputs = model(**batch)

        cumulative_loss += outputs.loss.item()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions.flatten() == batch['labels']).sum().item()
        # print(abundances, weights, weighted_correct,(predictions.flatten() == batch['labels']).sum().item(),  (weights[batch['labels'].cpu().numpy()] * (predictions.flatten() == batch['labels'])).sum().item())
        abundances = Counter(batch['labels'].cpu().numpy())
        abundance_values = np.array([abundances[k] for k in range(15)])
        weights = torch.Tensor(sum(abundance_values) / abundance_values).to(device)
        weighted_correct += (weights[batch['labels']] * (predictions.flatten() == batch['labels'])).sum().item()
        metric.add_batch(predictions=predictions, references=batch["labels"])
        if idx == 0 or idx == 15: # two random samples
            print(f"abundances: {abundances}, predictions: {predictions.flatten()[:30]}, true: {batch['labels'][:30]}")
            print(f"accuracy: {(predictions.flatten() == batch['labels']).sum().item() / len(batch['labels'])}, loss: {outputs.loss.item()}")
        # pdb.set_trace()

    return cumulative_loss / (idx+1), metric.compute(), weighted_correct

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
    # args.redshift is used inside here
    model_config = setup_model_config(args, config)

    finetune_config = {
        "has_labels": True,
        "num_labels": 14, # TODO: remove the anomalous types
        "classifier_dropout": config['classifier_dropout'],
        "fourier_pe": config['fourier_pe'],
        # "balance": config['balance'],
        "mask": config['mask']
    }
    model_config.update(finetune_config)

    return model_config

def train(args, base_config, config=None):
    if not args.dry_run:
        wandb.init(config=config, name=args.wandb_name, project="finetuning-sweep-fourier-masked" if args.mask else "finetuning-sweep-fourier-pretrained", dir=WANDB_DIR)
        config = wandb.config
    if config:
        config.update(base_config)
    else:
        config = base_config
    print(config)

    model_config = setup_model_config_finetune(args, config)

    print(f"loading dataset from {config['dataset_path']}")
    dataset = load_dataset(str(config['dataset_path']), cache_dir=CACHE_DIR)#, download_mode='force_redownload')
    abundances = Counter(dataset['train']['label'])
    print(abundances)
    max_count = max(abundances.values())
    class_weights = [abundances[i] / sum(abundances) for i in range(model_config.num_labels)]  # weight upsampled balanced dataset by abundance
    if args.class_weights:
        print(f"Class weights applied: {class_weights}")

    print(f"fourier pe? {config['fourier_pe']}, mask? {config['mask']}")

    dataloader_fn = create_train_dataloader_raw if config['fourier_pe'] else create_train_dataloader
    test_dataloader_fn = create_test_dataloader_raw if config['fourier_pe'] else create_test_dataloader

    print(f"redshift: {args.redshift}")
    train_dataloader = dataloader_fn(
        config=model_config,
        dataset=dataset['train'],
        time_features=[month_of_year for x in config['time_features'] if 'month_of_year' in x],
        batch_size=config["batch_size"],
        num_batches_per_epoch=4000, # TODO: this arg not used when masking, may not be necessary in general
        shuffle_buffer_length=1_000_000,
        allow_padding=False,
        # add_objid=True,
    )
    val_dataloader = test_dataloader_fn(
        config=model_config,
        dataset=dataset['validation'],
        time_features=[month_of_year for x in config['time_features'] if 'month_of_year' in x],
        batch_size=config["batch_size"],
        shuffle_buffer_length=1_000_000,
        compute_loss=True,# no longer optional for encoder-decoder latent space
        allow_padding=False,
    )
    test_dataset = get_dataset(config['test_set_path'])#, force_redownload=True)#, data_subset_file=Path(config['test_set_path']) / "single_test_file.txt") # random file from plasticc test dataset
    no_anomalies_dataset = test_dataset['train'].filter(lambda x: x['label'] < 14)

    test_dataloader = test_dataloader_fn(
        config=model_config,
        dataset=no_anomalies_dataset,
        time_features=[month_of_year for x in config['time_features'] if 'month_of_year' in x],
        batch_size=config["batch_size"],
        compute_loss=True,
        allow_padding=False,
    )

    if config.get('model_path') and not args.random_init:
        print(f"Loading model from {config['model_path']}")
        model = InformerForSequenceClassification.from_pretrained(config['model_path'], config=model_config, ignore_mismatched_sizes=True)
    elif args.random_init:
        print("Randomly initializing model")
        model = InformerForSequenceClassification(model_config)
    print(model)
    print(f"num total parameters: {sum(p.numel() for p in model.parameters())}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # if args.load_finetuned_model:
    #     ckpt = torch.load(args.load_finetuned_model)
    #     model.load_state_dict(ckpt['model_state_dict'])
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    model.train()
    if config['num_lp_steps'] > 0:
        model, _ = run_training_stage('lp', model, train_dataloader, val_dataloader, test_dataloader, config, device, dry_run=args.dry_run, class_weights=class_weights if args.class_weights else None, save_model_path=args.save_model)
        # linear_weights = linear_probe_sklearn(model, model_config, train_dataloader, device, config['batch_size'])
        # model.classifier.weight = torch.nn.Parameter(torch.tensor(linear_weights.astype(np.float32)).to(device))
    model, optimizer = run_training_stage('ft', model, train_dataloader, val_dataloader, test_dataloader, config, device, dry_run=args.dry_run, class_weights=class_weights if args.class_weights else None, save_model_path=args.save_model)

    if args.save_model:
        save_model(model, optimizer, args.save_model)

    model.eval()
    metric = load_metric("accuracy")
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
    # parser.add_argument('--num_epochs', type=int, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"', required=True)
    parser.add_argument('--context_length', type=int, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--load_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--load_finetuned_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--save_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--dry_run', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--random_init', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--test_set_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--fourier_pe', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--mask', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--no_balance', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--redshift', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--class_weights', action='store_true')
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--num_lp_steps', type=int)
    parser.add_argument('--num_ft_steps', type=int)
    parser.add_argument('--wandb_name', type=str)

    args = parser.parse_args()

    if args.no_augment:
        if args.no_balance:
            DATASET_PATH = Path('/pscratch/sd/h/helenqu/plasticc/raw_train_with_labels')
        else:
            DATASET_PATH = Path('/pscratch/sd/h/helenqu/plasticc/raw_train_with_labels_balanced')
    else:
        DATASET_PATH = Path('/pscratch/sd/h/helenqu/plasticc/train_augmented_dataset')

    TEST_SET_PATH = Path('/pscratch/sd/h/helenqu/plasticc/raw_test_with_labels')
    with open("/global/homes/h/helenqu/time_series_transformer/transformer_uda/configs/bigger_model_hyperparameters.yml", "r") as f:
        config = yaml.safe_load(f)
    config['dataset_path'] = str(DATASET_PATH)
    config['test_set_path'] = str(TEST_SET_PATH)
    config['model_path'] = args.load_model if args.load_model else None
    config['fourier_pe'] = args.fourier_pe
    config['batch_size'] = 256 # batch size per gpu
    config['balance'] = not args.no_balance
    config['mask'] = args.mask
    config['scaling'] = None

    with open("/global/homes/h/helenqu/time_series_transformer/transformer_uda/configs/bigger_model_best_finetune_hyperparams.yml", 'r') as f:
        ft_config = yaml.safe_load(f)
    ft_config['weight_decay'] = 0.01
    # ft_config['lp_lr'] = 1e-4 # for masked pretraining
    ft_config['lp_lr'] = 1e-5
    ft_config['num_lp_steps'] = args.num_lp_steps if args.num_lp_steps else 0
    ft_config['num_ft_steps'] = args.num_ft_steps if args.num_ft_steps else 80_000
    # ft_config['ft_lr'] = 4e-5# may need to be scaled by num processes
    ft_config['ft_lr'] = 4e-6# may need to be scaled by num processes
    ft_config['classifier_dropout'] = 0.3 if args.mask else 0.7

    train(args, config, ft_config)
