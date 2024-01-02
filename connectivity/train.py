import torch
from tqdm import tqdm
import pdb
from pathlib import Path
import numpy as np

from targeted_augs.examples.models.initializer import initialize_model
from targeted_augs.examples.utils import load
from wilds.common.data_loaders import get_train_loader
from connect_later.dataset_preprocess_raw import create_train_dataloader_raw, create_test_dataloader_raw
from connect_later.informer_models import InformerForSequenceClassification

DATASETS_PATH = Path('/path/to/datasets')

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def train(config, train_ds, val_ds, test_ds, connectivity_type):
    set_seed(config.seed)

    model = initialize_model(config, 2)
    print(model)

    # define dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=4)

    # define optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress_bar = tqdm(range(config.num_steps))

    model.to(device)

    def cycle(dataloader):
        while True:
            for x in dataloader:
                yield x
    # train
    avg_loss = 0
    patience = 0
    best_model = None
    best_val_loss = float('inf')
    for i, (x, y) in enumerate(cycle(train_loader)):
        if i == config['num_steps']:
            break

        model.train()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        avg_loss += loss.item()
        if i % 100 == 0:
            print(f"labels: {y}")
            print(f"predicted: {torch.argmax(logits, dim=1)}")
            print(f"avg loss: {avg_loss / 100}")
            final_loss = avg_loss/100
            avg_loss = 0
            # evaluate and early stop
            model.eval()
            avg_val_loss = 0
            for _, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                avg_val_loss += loss.item()
            print(f'val loss: {avg_val_loss / len(val_loader)}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model
                patience = 0
            else:
                patience += 1
                if patience > 5:
                    print("Early stopping at step", i)
                    break
        loss.backward()
        optimizer.step()
        progress_bar.update(1)

    # evaluate
    best_model.eval()
    total = 0
    correct = 0
    avg_loss = 0
    for _, (x, y) in enumerate(test_loader):
        x, y = x.cuda(), y.cuda()
        logits = best_model(x)
        _, predicted = torch.max(logits.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        avg_loss += criterion(logits, y).item()
    print(f'test accuracy: {correct / total}, test loss: {avg_loss / len(test_loader)}')

    results_path = DATASETS_PATH / f'{config.dataset}_connectivity_results.csv'
    if not results_path.exists():
        with open(results_path, 'w') as f:
            f.write('connectivity_type,seed,num_steps,num_steps_trained,lr,class1,class2,class1_num,class2_num,train_loss,test_accuracy,test_loss\n')
    with open(results_path, 'a') as f:
        #TODO: make type of prediction an arg (predict_domain, predict_class, predict_both)
        f.write(f"{connectivity_type},{config.seed},{config.num_steps},{i},{config.lr},{config['class1']},{config['class2']},{config['num_class_1_samples']},{config['num_class_2_samples']},{final_loss},{correct / total},{avg_loss / len(test_loader)}\n")

def train_astro(config, model_config, train_ds, val_ds, test_ds, connectivity_type):
    set_seed(config['seed'])

    model = InformerForSequenceClassification(model_config)
    train_loader = create_train_dataloader_raw(
        dataset=train_ds,
        batch_size=config["batch_size"],
        seed=config['seed'],
        add_objid=True,
        has_labels=True,
        masked_ft=True, # this makes sure masking happens even though we are doing FT
    )
    val_loader = create_train_dataloader_raw(
        dataset=val_ds,
        batch_size=config["batch_size"],
        seed=config['seed'],
        add_objid=True,
        has_labels=True,
        masked_ft=True, # this makes sure masking happens even though we are doing FT
    )
    test_loader = create_train_dataloader_raw(
        dataset=test_ds,
        batch_size=config["batch_size"],
        seed=config['seed'],
        add_objid=True,
        has_labels=True,
        masked_ft=True, # this makes sure masking happens even though we are doing FT
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    progress_bar = tqdm(range(config['num_steps']))

    model.to(device)

    def cycle(dataloader):
        while True:
            for x in dataloader:
                yield x

    avg_loss = 0
    best_model = None
    best_val_loss = float('inf')
    patience = 0
    for step, batch in enumerate(cycle(train_loader)):
        if step == config['num_steps']:
            break

        model.train()
        optimizer.zero_grad()
        batch['labels'] = batch['labels'].type(torch.int64)
        input_batch = {k: v.to(device) for k, v in batch.items() if k != 'objid'}
        outputs = model(**input_batch)
        loss = outputs.loss
        avg_loss += loss.item()

        if step % 100 == 0:
            print(f"labels: {batch['labels']}")
            print(f"predicted: {torch.argmax(outputs.logits, dim=-1).flatten()}")
            print(f"step {step} avg loss: {avg_loss/100}")
            final_loss = avg_loss/100
            avg_loss = 0
            # evaluate and early stop
            model.eval()
            avg_val_loss = 0
            for _, batch in enumerate(val_loader):
                batch['labels'] = batch['labels'].type(torch.int64)
                input_batch = {k: v.to(device) for k, v in batch.items() if k != 'objid'}
                outputs = model(**input_batch)
                avg_val_loss += outputs.loss.item()
            print(f'val loss: {avg_val_loss / len(val_loader)}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model
                patience = 0
            else:
                patience += 1
                if patience > 5:
                    print("Early stopping at step", step)
                    break

        loss.backward()
        optimizer.step()
        progress_bar.update(1)

    # evaluate
    best_model.eval()
    total = 0
    correct = 0
    avg_loss = 0
    for i, batch in enumerate(test_loader):
        batch['labels'] = batch['labels'].type(torch.int64)
        input_batch = {k: v.to(device) for k, v in batch.items() if k != 'objid'}
        outputs = best_model(**input_batch)
        predicted = torch.argmax(outputs.logits, dim=-1)
        total += input_batch['labels'].size(0)
        correct += (predicted.flatten() == input_batch['labels']).sum().item()
        avg_loss += outputs.loss.item()
    print(f"test accuracy for {config['class1']}, {config['class2']}: {correct / total}, test loss: {avg_loss / len(test_loader)}")

    results_path = DATASETS_PATH / 'astro_connectivity_results.csv'
    if not results_path.exists():
        with open(results_path, 'w') as f:
            f.write('connectivity_type,seed,num_steps,num_steps_trained,lr,class1,class2,class1_num,class2_num,train_loss,test_accuracy,test_loss\n')
    with open(results_path, 'a') as f:
        #TODO: make type of prediction an arg (predict_domain, predict_class, predict_both)
        f.write(f"{connectivity_type},{config['seed']},{config['num_steps']},{step},{config['lr']},{config['class1']},{config['class2']},{config['num_class_1_samples']},{config['num_class_2_samples']},{final_loss},{correct / total},{avg_loss / len(test_loader)}\n")
