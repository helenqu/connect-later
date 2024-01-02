#!/bin/bash
#SBATCH -A dessn
#SBATCH -C gpu
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 128
#SBATCH -q regular
#SBATCH --gpus=4

export SLURM_CPU_BIND="cores"
export HF_HOME="/pscratch/sd/h/helenqu/huggingface_datasets_cache"
module load python
module load pytorch/1.13.1
source activate pytorch-1.13.1
module load pytorch/1.13.1
# srun python \
accelerate launch --multi_gpu --num_processes=4 \
    --mixed_precision=bf16 \
    /global/homes/h/helenqu/time_series_transformer/connect_later/huggingface_informer.py \
    --data_dir /pscratch/sd/h/helenqu/plasticc/raw/plasticc_raw_examples \
    --save_model /pscratch/sd/h/helenqu/plasticc/models/pretrained_uniformity_cossim_1e-2 \
    --fourier_pe \
    --mask \
    --num_steps 60_000 \
    --uniformity_loss_weight 1e-2
    # --load_checkpoint /pscratch/sd/h/helenqu/plasticc/plasticc_all_gp_interp/models/checkpoints/checkpoint_masked_60%_2023-09-02_12:20:32_step_15000
