# Connect Later: Improving Fine-tuning for Robustness with Targeted Augmentations

## Astronomical Time-Series Datasets
### Datasets
will be released on the HuggingFace Hub!
### Pretraining
```
bash run_pretrain.sh /path/to/dataset /path/to/save/model wandb_run_name
```
### Fine-tuning
```
bash run_finetuning.sh \
  /path/to/pretrained/model \
  /path/to/save/model \
  /path/to/dataset \
  wandb_run_name \
  "--class_weights --test_set_path /path/to/test/dataset --seed 12345"
```
add `--redshift_prediction` argument and remove `--class_weights` to do redshift prediction instead of classification
### Pseudolabeling to run self-training baseline
```
bash run_pseudolabeling.sh \
  /path/to/dataset \
  /path/to/trained/model \
  /path/to/dataset/labels \
  /path/to/output/dir
```
