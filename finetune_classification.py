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

from connect_later.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw
from connect_later.informer_models import InformerForSequenceClassification
from connect_later.huggingface_informer import get_dataset, setup_model_config

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
        # if idx % 100 == 0:
            # print(f"training: {batch['labels']}")
        if class_weights is not None:
            batch['weights'] = torch.tensor(class_weights)
        if not model_config.regression:
            batch['labels'] = batch['labels'].type(torch.int64)
        input_batch = {k: v.to(device) for k, v in batch.items() if k != 'objid'}
        if idx == 0:
            print(f"batch contents: {input_batch.keys()}")
            print(input_batch['past_values'])
        outputs = model(**input_batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        progress_bar.update(1)

        # cumulative_loss += loss.item()
        logits = outputs.logits
        if model_config.regression: # redshift prediction
            predictions = logits.squeeze()
        elif model_config.num_labels == 1:
            predictions = torch.round(torch.sigmoid(logits))
        else:
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

def run_training_stage(stage, model, model_config, train_dataloader, val_dataloader, test_dataloader, config, dry_run=False, class_weights=None, save_model_path=None):
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
        if type(outputs.loss) != torch.Tensor:
            pdb.set_trace()

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
    # args.redshift is used inside here
    model_config = setup_model_config(args, config)

    finetune_config = {
        "has_labels": True,
        "num_labels": 1 if args.redshift_prediction or args.sdss else 14, # TODO: remove the anomalous types
        "regression": args.redshift_prediction,
        "classifier_dropout": config['classifier_dropout'],
        "fourier_pe": config['fourier_pe'],
        # "balance": config['balance'],
        "mask": config['mask']
    }
    model_config.update(finetune_config)

    return model_config

def train(args, base_config, config=None):
    if not args.dry_run:
        project_name = "finetuning-redshift" if args.redshift_prediction else "finetuning-sweep-fourier-masked"
        if args.sdss:
            project_name = 'finetuning-sdss'
        wandb.init(config=config, name=args.wandb_name, project=project_name, dir=WANDB_DIR)
        config = wandb.config
    if config:
        config.update(base_config)
    else:
        config = base_config
    print(config)

    model_config = setup_model_config_finetune(args, config)

    if args.seed:
        set_seed(args.seed)

    print(f"loading dataset from {config['dataset_path']}")
    dataset = load_dataset(str(config['dataset_path']), cache_dir=CACHE_DIR)#, download_mode='force_redownload')

    if args.self_training:
        print("TODO: REMOVE - adding original training dataset")
        if args.redshift_prediction:
            orig_train = load_dataset("/pscratch/sd/h/helenqu/plasticc/train_augmented_redshift", cache_dir=CACHE_DIR)
        else:
            orig_train = load_dataset("/pscratch/sd/h/helenqu/plasticc/train_augmented_dataset", cache_dir=CACHE_DIR)
            orig_train['train'] = orig_train['train'].remove_columns(['redshift'])
        print(f"original training dataset size: {len(orig_train['train'])}")
        dataset['train'] = concatenate_datasets([orig_train['train'], dataset['train']])
        dataset['validation'] = orig_train['validation']
    print(f"dataset sizes: {len(dataset['train'])}, {len(dataset['validation'])}")

    abundances = Counter(dataset['train']['label'])
    # print(abundances)
    max_count = max(abundances.values())
    class_weights = [abundances[i] / sum(abundances.values()) for i in range(model_config.num_labels)] if args.class_weights else None # weight upsampled balanced dataset by abundance
    if args.class_weights:
        print(f"Class weights applied: {class_weights}")

    print(f"fourier pe? {config['fourier_pe']}, mask? {config['mask']}")

    test_dataloader_fn = create_test_dataloader_raw if config['fourier_pe'] else create_test_dataloader

    print(f"redshift: {args.redshift}")
    train_dataloader = create_train_dataloader_raw(
        dataset=dataset['train'],
        batch_size=config["batch_size"],
        seed=args.seed,
        add_objid=True,
        has_labels=True,
        masked_ft=args.masked_ft,
        mask_probability=args.mask_probability
    )
    val_dataloader = create_test_dataloader_raw(
        dataset=dataset['validation'],
        batch_size=config["batch_size"],
        compute_loss=True,# no longer optional for encoder-decoder latent space
        seed=args.seed,
        has_labels=True,
    )
    test_dataset = get_dataset(config['test_set_path'])['train'] if config['test_set_path'] is not None else dataset['test']
    #, force_redownload=True)#, data_subset_file=Path(config['test_set_path']) / "single_test_file.txt") # random file from plasticc test dataset
    no_anomalies_dataset = test_dataset.filter(lambda x: x['label'] < 14)

    test_dataloader = test_dataloader_fn(
        dataset=no_anomalies_dataset,
        batch_size=config["batch_size"],
        compute_loss=True,
        seed=args.seed,
        has_labels=True
    )

    if config.get('model_path') and not args.random_init:
        print(f"Loading model from {config['model_path']}")
        model = InformerForSequenceClassification.from_pretrained(config['model_path'], config=model_config, ignore_mismatched_sizes=True)
    elif args.random_init:
        print("Randomly initializing model")
        model = InformerForSequenceClassification(model_config)
    print(model)
    print(f"num total parameters: {sum(p.numel() for p in model.parameters())}")

    model.train()
    if config['num_lp_steps'] > 0:
        model, _ = run_training_stage('lp', model, model_config, train_dataloader, val_dataloader, test_dataloader, config, dry_run=args.dry_run, class_weights=class_weights, save_model_path=args.save_model)
        # linear_weights = linear_probe_sklearn(model, model_config, train_dataloader, device, config['batch_size'])
        # model.classifier.weight = torch.nn.Parameter(torch.tensor(linear_weights.astype(np.float32)).to(device))
    model, optimizer = run_training_stage('ft', model, model_config, train_dataloader, val_dataloader, test_dataloader, config, dry_run=args.dry_run, class_weights=class_weights, save_model_path=args.save_model)

    if args.save_model:
        save_model(model, optimizer, args.save_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')
    parser.add_argument('--dataset_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--test_set_path', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--context_length', type=int, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--load_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--load_finetuned_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--save_model', type=str, help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--dry_run', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--random_init', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--fourier_pe', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--mask', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    # parser.add_argument('--no_balance', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--redshift', action='store_true', help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')
    parser.add_argument('--class_weights', action='store_true')
    parser.add_argument('--no_augment', action='store_true')
    parser.add_argument('--num_lp_steps', type=int)
    parser.add_argument('--num_ft_steps', type=int)
    parser.add_argument('--lp_lr', type=float)
    parser.add_argument('--ft_lr', type=float)
    parser.add_argument('--wandb_name', type=str)
    parser.add_argument('--redshift_prediction', action='store_true')
    parser.add_argument('--sdss', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mask_probability', type=float, default=0.6)
    parser.add_argument('--masked_ft', action='store_true')
    parser.add_argument('--self_training', action='store_true')
    parser.add_argument('--uniformity_loss_weight', type=float, default=0)

    args = parser.parse_args()

    if args.sdss:
        DATASET_PATH = '/pscratch/sd/h/helenqu/sdss/spec_augmented_dataset'
    elif args.redshift_prediction:
        print("running redshift prediction")
        DATASET_PATH = '/pscratch/sd/h/helenqu/plasticc/train_augmented_redshift'
    elif args.no_augment:
        DATASET_PATH = '/pscratch/sd/h/helenqu/plasticc/raw_train_with_labels'
    else:
        DATASET_PATH = '/pscratch/sd/h/helenqu/plasticc/train_augmented_dataset'

    TEST_SET_PATH = Path("/pscratch/sd/h/helenqu/plasticc")
    if args.dataset_path:
        print("using dataset path for test set unless --test_set_path is set")
        TEST_SET_PATH = None
    elif args.sdss:
        TEST_SET_PATH = "/pscratch/sd/h/helenqu/sdss/phot_dataset"
    elif args.redshift_prediction:
        TEST_SET_PATH = str(TEST_SET_PATH / "raw_test_redshift")
    else:
        TEST_SET_PATH = str(TEST_SET_PATH / 'raw_test_with_labels')

    if "69M" in args.wandb_name:
        config_yml = "/global/homes/h/helenqu/time_series_transformer/configs/75M_masked_hyperparameters.yml"
    else:
        config_yml = "/global/homes/h/helenqu/time_series_transformer/configs/bigger_model_hyperparameters.yml"
    with open(config_yml, "r") as f:
        config = yaml.safe_load(f)
    config['dataset_path'] = DATASET_PATH if not args.dataset_path else args.dataset_path
    config['test_set_path'] = TEST_SET_PATH if not args.test_set_path else args.test_set_path
    config['model_path'] = args.load_model if args.load_model else None
    config['fourier_pe'] = args.fourier_pe
    config['batch_size'] = 128 # batch size per gpu
    # config['balance'] = not args.no_balance
    config['mask'] = args.mask
    config['scaling'] = None

    with open("/global/homes/h/helenqu/time_series_transformer/configs/bigger_model_best_finetune_hyperparams.yml", 'r') as f:
        ft_config = yaml.safe_load(f)
    ft_config['weight_decay'] = 0.01
    ft_config['lp_lr'] = 1e-4 if args.lp_lr is None else args.lp_lr # for masked pretraining
    # ft_config['lp_lr'] = 1e-5
    ft_config['num_lp_steps'] = args.num_lp_steps if args.num_lp_steps else 0
    ft_config['num_ft_steps'] = args.num_ft_steps if args.num_ft_steps else 10_000
    ft_config['ft_lr'] = 4e-5 if args.ft_lr is None else args.ft_lr # may need to be scaled by num processes
    ft_config['classifier_dropout'] = 0.3 #if args.mask else 0.7

    train(args, config, ft_config)
