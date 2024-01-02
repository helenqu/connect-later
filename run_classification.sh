#!/bin/bash

bash run_finetuning.sh \
    /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/pretrained_all_masked_60%/hf_model \
    /pscratch/sd/h/helenqu/plasticc/models/masked_60%_seed_12345_test \
    /pscratch/sd/h/helenqu/plasticc/train_augmented_dataset \
    test_wandb_name \
    "--class_weights --test_set_path /pscratch/sd/h/helenqu/plasticc/raw_train_with_labels --seed 12345"
