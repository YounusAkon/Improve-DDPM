"""
Evaluation metrics for medical image generation
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2


def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index"""
    # Convert to numpy
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()

    # Ensure 2D
    if img1.ndim > 2:
        img1 = img1.squeeze()
    if img2.ndim > 2:
        img2 = img2.squeeze()

    return ssim(img1, img2, data_range=img1.max() - img1.min())


def calculate_mae(img1, img2):
    """Calculate Mean Absolute Error"""
    return torch.mean(torch.abs(img1 - img2))


def calculate_mse(img1, img2):
    """Calculate Mean Squared Error"""
    return torch.mean((img1 - img2) ** 2)


def evaluate_generated_images(generated_images, target_images):
    """
    Evaluate generated images against targets

    Args:
        generated_images: Tensor of generated images [B, C, H, W]
        target_images: Tensor of target images [B, C, H, W]

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': [],
        'mse': []
    }

    batch_size = generated_images.shape[0]

    for i in range(batch_size):
        gen_img = generated_images[i]
        target_img = target_images[i]

        # Calculate metrics
        psnr_val = calculate_psnr(gen_img, target_img)
        ssim_val = calculate_ssim(gen_img, target_img)
        mae_val = calculate_mae(gen_img, target_img)
        mse_val = calculate_mse(gen_img, target_img)

        metrics['psnr'].append(psnr_val.item() if torch.is_tensor(psnr_val) else psnr_val)
        metrics['ssim'].append(ssim_val)
        metrics['mae'].append(mae_val.item())
        metrics['mse'].append(mse_val.item())

    # Calculate averages
    avg_metrics = {
        'avg_psnr': np.mean(metrics['psnr']),
        'avg_ssim': np.mean(metrics['ssim']),
        'avg_mae': np.mean(metrics['mae']),
        'avg_mse': np.mean(metrics['mse']),
        'std_psnr': np.std(metrics['psnr']),
        'std_ssim': np.std(metrics['ssim']),
        'std_mae': np.std(metrics['mae']),
        'std_mse': np.std(metrics['mse'])
    }

    return avg_metrics, metrics
