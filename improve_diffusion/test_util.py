"""
Utilities for testing and evaluation of CT-MRI translation models
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def calculate_metrics(pred_images, target_images):
    """
    Calculate evaluation metrics for generated images

    Args:
        pred_images: Generated images tensor [B, C, H, W]
        target_images: Ground truth images tensor [B, C, H, W]

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': [],
        'mse': [],
        'lpips': []
    }

    batch_size = pred_images.shape[0]

    for i in range(batch_size):
        pred = pred_images[i].cpu().numpy()
        target = target_images[i].cpu().numpy()

        # Convert from [-1, 1] to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        # Ensure single channel
        if pred.ndim == 3:
            pred = pred[0]
        if target.ndim == 3:
            target = target[0]

        # Calculate metrics
        psnr_val = psnr(target, pred, data_range=1.0)
        ssim_val = ssim(target, pred, data_range=1.0)
        mae_val = np.mean(np.abs(pred - target))
        mse_val = np.mean((pred - target) ** 2)

        metrics['psnr'].append(psnr_val)
        metrics['ssim'].append(ssim_val)
        metrics['mae'].append(mae_val)
        metrics['mse'].append(mse_val)

    # Calculate averages
    avg_metrics = {}
    for key in metrics:
        if metrics[key]:  # Check if list is not empty
            avg_metrics[f'avg_{key}'] = np.mean(metrics[key])
            avg_metrics[f'std_{key}'] = np.std(metrics[key])

    return avg_metrics, metrics


def save_comparison_grid(source_images, target_images, pred_images, save_path, num_samples=4):
    """
    Save a comparison grid showing source, target, and predicted images

    Args:
        source_images: Source CT images [B, C, H, W]
        target_images: Target MRI images [B, C, H, W]
        pred_images: Predicted MRI images [B, C, H, W]
        save_path: Path to save the comparison image
        num_samples: Number of samples to show
    """
    num_samples = min(num_samples, source_images.shape[0])

    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 3, 9))

    for i in range(num_samples):
        # Convert tensors to numpy and normalize to [0, 1]
        source = (source_images[i].cpu().numpy().squeeze() + 1) / 2
        target = (target_images[i].cpu().numpy().squeeze() + 1) / 2
        pred = (pred_images[i].cpu().numpy().squeeze() + 1) / 2

        # Plot source (CT)
        axes[0, i].imshow(source, cmap='gray')
        axes[0, i].set_title(f'CT {i + 1}')
        axes[0, i].axis('off')

        # Plot target (MRI)
        axes[1, i].imshow(target, cmap='gray')
        axes[1, i].set_title(f'Real MRI {i + 1}')
        axes[1, i].axis('off')

        # Plot prediction
        axes[2, i].imshow(pred, cmap='gray')
        axes[2, i].set_title(f'Generated MRI {i + 1}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_model(model, diffusion, data_loader, device, num_batches=10, save_dir=None):
    """
    Evaluate model on validation data

    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        data_loader: Validation data loader
        device: Device to run evaluation on
        num_batches: Number of batches to evaluate
        save_dir: Directory to save evaluation results

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    all_metrics = []

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (source, target) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break

            source = source.to(device)
            target = target.to(device)

            # Generate samples
            sample_shape = target.shape
            model_kwargs = {"condition": source}

            generated = diffusion.p_sample_loop(
                model,
                sample_shape,
                model_kwargs=model_kwargs,
                device=device,
                progress=False
            )

            # Calculate metrics
            batch_metrics, _ = calculate_metrics(generated, target)
            all_metrics.append(batch_metrics)

            # Save comparison images
            if save_dir and batch_idx < 5:  # Save first 5 batches
                save_path = os.path.join(save_dir, f'comparison_batch_{batch_idx}.png')
                save_comparison_grid(source, target, generated, save_path)

    # Aggregate metrics
    final_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                final_metrics[key] = np.mean(values)

    return final_metrics


def visualize_diffusion_process(model, diffusion, source_image, device, save_path, num_steps=10):
    """
    Visualize the diffusion sampling process

    Args:
        model: Trained diffusion model
        diffusion: Diffusion process
        source_image: Source CT image [1, C, H, W]
        device: Device to run on
        save_path: Path to save visualization
        num_steps: Number of steps to visualize
    """
    model.eval()
    source_image = source_image.to(device)

    # Sample with intermediate steps
    sample_shape = source_image.shape
    model_kwargs = {"condition": source_image}

    samples = []
    step_indices = np.linspace(0, diffusion.num_timesteps - 1, num_steps, dtype=int)

    with torch.no_grad():
        img = torch.randn(sample_shape, device=device)

        for i, step in enumerate(reversed(range(diffusion.num_timesteps))):
            if step in step_indices:
                samples.append(img.clone())

            t = torch.tensor([step] * sample_shape[0], device=device)
            out = diffusion.p_sample(
                model, img, t, model_kwargs=model_kwargs
            )
            img = out["sample"]

        samples.append(img)  # Final result

    # Create visualization
    fig, axes = plt.subplots(2, len(samples), figsize=(len(samples) * 2, 4))

    # Show source image in first row
    source_np = (source_image[0].cpu().numpy().squeeze() + 1) / 2
    for i in range(len(samples)):
        axes[0, i].imshow(source_np, cmap='gray')
        axes[0, i].set_title('Source CT')
        axes[0, i].axis('off')

    # Show sampling process in second row
    for i, sample in enumerate(samples):
        sample_np = (sample[0].cpu().numpy().squeeze() + 1) / 2
        axes[1, i].imshow(sample_np, cmap='gray')
        if i < len(samples) - 1:
            step = step_indices[i] if i < len(step_indices) else 0
            axes[1, i].set_title(f'Step {step}')
        else:
            axes[1, i].set_title('Final')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_model_inference(model_path, source_image_path, device='cuda'):
    """
    Test model inference on a single image

    Args:
        model_path: Path to trained model checkpoint
        source_image_path: Path to source CT image
        device: Device to run inference on

    Returns:
        Generated MRI image as numpy array
    """
    # Load model (this would need to be implemented based on your model structure)
    # model = load_model(model_path, device)

    # Load and preprocess image
    source_img = Image.open(source_image_path).convert('L')
    source_img = source_img.resize((256, 256))

    # Convert to tensor
    source_tensor = torch.from_numpy(np.array(source_img)).float()
    source_tensor = source_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    source_tensor = (source_tensor / 127.5) - 1.0  # Normalize to [-1, 1]
    source_tensor = source_tensor.to(device)

    # Generate MRI
    with torch.no_grad():
        # This would use your trained model and diffusion process
        # generated = generate_mri(model, diffusion, source_tensor)
        pass

    # Convert back to numpy
    # generated_np = (generated[0].cpu().numpy().squeeze() + 1) / 2
    # return generated_np

    return None  # Placeholder


class ModelTester:
    """
    Comprehensive model testing class
    """

    def __init__(self, model, diffusion, device):
        self.model = model
        self.diffusion = diffusion
        self.device = device

    def run_full_evaluation(self, val_loader, save_dir):
        """Run complete evaluation suite"""

        print("Running model evaluation...")

        # Quantitative evaluation
        metrics = evaluate_model(
            self.model, self.diffusion, val_loader,
            self.device, save_dir=save_dir
        )

        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Qualitative evaluation
        sample_batch = next(iter(val_loader))
        source, target = sample_batch

        # Visualize diffusion process
        process_save_path = os.path.join(save_dir, 'diffusion_process.png')
        visualize_diffusion_process(
            self.model, self.diffusion, source[:1],
            self.device, process_save_path
        )

        return metrics
