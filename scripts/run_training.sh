#!/bin/bash

# CT to MRI Diffusion Model Training Script

echo "Starting CT to MRI diffusion model training..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Training parameters
DATA_DIR="data"
LOG_DIR="logs/ct_mri_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="models/checkpoints"

# Create directories
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# Run training
python scripts/training/train_ct_mri.py \
    --data_dir $DATA_DIR \
    --log_dir $LOG_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --image_size 256 \
    --batch_size 4 \
    --lr 1e-4 \
    --num_channels 128 \
    --num_res_blocks 2 \
    --attention_resolutions "16,8" \
    --diffusion_steps 1000 \
    --noise_schedule "cosine" \
    --log_interval 100 \
    --save_interval 5000 \
    --use_fp16 \
    --ema_rate "0.9999"

echo "Training completed!"
