#!/bin/bash

# CT to MRI Inference Script

echo "Running CT to MRI inference..."

# Parameters
CT_IMAGE="data/test_ct.png"
CHECKPOINT="models/checkpoints/model_best.pt"
OUTPUT_DIR="results/samples"
NUM_SAMPLES=4

# Create output directory
mkdir -p $OUTPUT_DIR

# Run inference
python scripts/inference/generate_mri.py \
    --ct_image $CT_IMAGE \
    --checkpoint $CHECKPOINT \
    --output_dir $OUTPUT_DIR \
    --num_samples $NUM_SAMPLES \
    --use_ddim \
    --image_size 256

echo "Inference completed! Results saved to $OUTPUT_DIR"
