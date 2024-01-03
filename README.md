# Connect Later: Improving Fine-tuning for Robustness with Targeted Augmentations

## Astronomical Time-Series Datasets (AstroClassification and Redshifts)
### Datasets
Download the pretraining dataset [here](https://huggingface.co/datasets/helenqu/astro-classification-redshifts-pretrain) and the fine-tuning dataset with the _redshifting_ targeted augmentation applied [here](https://huggingface.co/datasets/helenqu/astro-classification-redshifts-augmented).

### Pretraining
```shell
bash run_pretrain.sh /path/to/pretraining/dataset /path/to/save/model wandb_run_name
```
### Fine-tuning
```shell
bash run_finetuning.sh \
  /path/to/pretrained/model \
  /path/to/save/model \
  /path/to/finetuning/dataset \
  wandb_run_name \
  "--class_weights --test_set_path /path/to/test/dataset --seed 12345"
```
add `--redshift_prediction` argument and remove `--class_weights` to do redshift prediction instead of classification
### Pseudolabeling to run self-training baseline
```shell
bash run_pseudolabeling.sh \
  /path/to/dataset \
  /path/to/trained/model \
  /path/to/dataset/labels \
  /path/to/output/dir
```
