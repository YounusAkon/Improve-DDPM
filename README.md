# CT to MRI Diffusion Model

A PyTorch implementation of a conditional diffusion model for translating CT images to MRI images using paired medical datasets.

## Features

- **Conditional Diffusion Model**: Uses CT images as conditioning for MRI generation
- **Medical Image Support**: Handles DICOM, NIfTI, and PNG formats
- **Distributed Training**: Multi-GPU training with MPI support
- **Mixed Precision**: FP16 training for memory efficiency
- **DDIM Sampling**: Fast sampling for inference
- **Comprehensive Evaluation**: PSNR, SSIM, MAE, MSE metrics

## Project Structure

\`\`\`
ct-to-mri-diffusion/
├── data/
│   ├── ct_images/          # CT images (PNG format)
│   ├── mri_images/         # MRI images (PNG format)
│   └── processed/          # Preprocessed data
├── models/
│   ├── checkpoints/        # Model checkpoints
│   ├── configs/           # Configuration files
│   └── conditional_unet.py # Conditional UNet model
├── scripts/
│   ├── training/          # Training scripts
│   ├── inference/         # Inference scripts
│   └── run_*.sh          # Shell scripts
├── utils/
│   ├── medical_dataset.py # Dataset loaders
│   ├── data_preprocessing.py # Data preprocessing
│   └── evaluation_metrics.py # Evaluation metrics
└── logs/                  # Training logs
\`\`\`

## Installation

1. Install dependencies:
\`\`\`bash
pip install torch torchvision torchaudio
pip install numpy pillow opencv-python
pip install nibabel scikit-image
pip install mpi4py blobfile
\`\`\`

2. Set up the project:
\`\`\`bash
python scripts/setup_project.py
\`\`\`

## Data Preparation

1. **For NIfTI files**:
\`\`\`bash
python utils/data_preprocessing.py \
    --ct_dir /path/to/ct/nifti/files \
    --mri_dir /path/to/mri/nifti/files \
    --output_dir data \
    --create_split
\`\`\`

2. **For PNG files**: Place paired CT and MRI images in `data/ct_images/` and `data/mri_images/`

## Training

1. **Single GPU**:
\`\`\`bash
python scripts/training/train_ct_mri.py \
    --data_dir data \
    --batch_size 4 \
    --lr 1e-4 \
    --image_size 256
\`\`\`

2. **Multi-GPU**:
\`\`\`bash
mpiexec -n 4 python scripts/training/train_ct_mri.py \
    --data_dir data \
    --batch_size 4 \
    --lr 1e-4 \
    --image_size 256
\`\`\`

3. **Using shell script**:
\`\`\`bash
bash scripts/run_training.sh
\`\`\`

## Inference

Generate MRI from CT image:
\`\`\`bash
python scripts/inference/generate_mri.py \
    --ct_image path/to/ct/image.png \
    --checkpoint models/checkpoints/model_best.pt \
    --output_dir results/samples \
    --use_ddim \
    --num_samples 4
\`\`\`

## Configuration

Edit `configs/ct_mri_config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data preprocessing settings
- Sampling configurations

## Model Architecture

- **Conditional UNet**: Modified UNet that takes CT images as conditioning
- **Attention Mechanisms**: Multi-head attention at multiple resolutions
- **Residual Blocks**: Deep residual connections for better gradient flow
- **Timestep Embedding**: Sinusoidal embeddings for diffusion timesteps

## Evaluation

The model includes comprehensive evaluation metrics:
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error

## Tips for Medical Images

1. **Preprocessing**: Proper windowing is crucial for medical images
2. **Data Augmentation**: Use medical-specific augmentations
3. **Evaluation**: Use domain-specific metrics beyond standard image metrics
4. **Validation**: Always validate with medical experts

## Troubleshooting

1. **Memory Issues**: Reduce batch size or use gradient checkpointing
2. **Training Instability**: Try different learning rates or beta schedules
3. **Poor Quality**: Increase model capacity or training steps
4. **Slow Sampling**: Use DDIM with fewer steps

## Citation

If you use this code, please cite the original diffusion model papers:
- Improved Denoising Diffusion Probabilistic Models
- Denoising Diffusion Implicit Models
# test
# test
