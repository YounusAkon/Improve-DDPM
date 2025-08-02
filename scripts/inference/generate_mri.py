# """
# Inference script to generate MRI from CT images
# """
#
# import argparse
# import os
# import torch
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
#
# import sys
#
# sys.path.append('../../')
#
# from models.conditional_unet import CTMRIDiffusionModel
# from gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
# from gaussian_diffusion import ModelMeanType, ModelVarType, LossType
# import dist_util
#
#
# def load_model(checkpoint_path, args):
#     """Load trained model from checkpoint"""
#
#     model = CTMRIDiffusionModel(
#         image_size=args.image_size,
#         in_channels=1,
#         model_channels=args.num_channels,
#         out_channels=1,
#         num_res_blocks=args.num_res_blocks,
#         attention_resolutions=args.attention_resolutions,
#         dropout=0.0,  # No dropout during inference
#         num_heads=args.num_heads
#     )
#
#     # Load checkpoint
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     model.load_state_dict(checkpoint)
#     model.to(dist_util.dev())
#     model.eval()
#
#     return model
#
#
# def create_diffusion(args):
#     """Create diffusion process for sampling"""
#     betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
#
#     return GaussianDiffusion(
#         betas=betas,
#         model_mean_type=ModelMeanType.EPSILON,
#         model_var_type=ModelVarType.FIXED_SMALL,
#         loss_type=LossType.MSE,
#         rescale_timesteps=True,
#     )
#
#
# def preprocess_ct_image(image_path, image_size=256):
#     """Preprocess CT image for inference"""
#
#     transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
#     ])
#
#     image = Image.open(image_path).convert('L')
#     tensor = transform(image).unsqueeze(0)  # Add batch dimension
#
#     return tensor
#
#
# def postprocess_mri_image(tensor):
#     """Convert tensor back to PIL image"""
#
#     # Denormalize from [-1, 1] to [0, 1]
#     tensor = (tensor + 1) / 2
#     tensor = torch.clamp(tensor, 0, 1)
#
#     # Convert to PIL
#     tensor = tensor.squeeze(0).squeeze(0)  # Remove batch and channel dims
#     array = (tensor.cpu().numpy() * 255).astype(np.uint8)
#
#     return Image.fromarray(array)
#
#
# def generate_mri(model, diffusion, ct_image, num_samples=1, use_ddim=True):
#     """Generate MRI image from CT image"""
#
#     batch_size = ct_image.shape[0]
#
#     # Sample noise
#     noise = torch.randn(batch_size, 1, ct_image.shape[2], ct_image.shape[3])
#     noise = noise.to(dist_util.dev())
#     ct_image = ct_image.to(dist_util.dev())
#
#     # Generate samples
#     if use_ddim:
#         # DDIM sampling (faster)
#         samples = diffusion.ddim_sample_loop(
#             model,
#             shape=noise.shape,
#             noise=noise,
#             model_kwargs={'condition': ct_image},
#             eta=0.0,
#             progress=True
#         )
#     else:
#         # DDPM sampling (slower but potentially better quality)
#         samples = diffusion.p_sample_loop(
#             model,
#             shape=noise.shape,
#             noise=noise,
#             model_kwargs={'condition': ct_image},
#             progress=True
#         )
#
#     return samples
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--ct_image", required=True, help="Path to CT image")
#     parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
#     parser.add_argument("--output_dir", default="../../results/samples", help="Output directory")
#     parser.add_argument("--image_size", type=int, default=256)
#     parser.add_argument("--num_channels", type=int, default=128)
#     parser.add_argument("--num_res_blocks", type=int, default=2)
#     parser.add_argument("--num_heads", type=int, default=4)
#     parser.add_argument("--attention_resolutions", default="16,8")
#     parser.add_argument("--diffusion_steps", type=int, default=1000)
#     parser.add_argument("--noise_schedule", default="cosine")
#     parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampling")
#     parser.add_argument("--num_samples", type=int, default=1)
#
#     args = parser.parse_args()
#
#     # Create output directory
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     print("Loading model...")
#     model = load_model(args.checkpoint, args)
#
#     print("Creating diffusion process...")
#     diffusion = create_diffusion(args)
#
#     print("Preprocessing CT image...")
#     ct_tensor = preprocess_ct_image(args.ct_image, args.image_size)
#
#     print("Generating MRI image...")
#     with torch.no_grad():
#         generated_mri = generate_mri(
#             model, diffusion, ct_tensor,
#             num_samples=args.num_samples,
#             use_ddim=args.use_ddim
#         )
#
#     print("Saving results...")
#     for i in range(args.num_samples):
#         # Save generated MRI
#         mri_image = postprocess_mri_image(generated_mri[i:i + 1])
#         output_path = os.path.join(args.output_dir, f"generated_mri_{i}.png")
#         mri_image.save(output_path)
#
#         # Save input CT for comparison
#         ct_image = postprocess_mri_image(ct_tensor[i:i + 1])
#         ct_output_path = os.path.join(args.output_dir, f"input_ct_{i}.png")
#         ct_image.save(ct_output_path)
#
#     print(f"Results saved to {args.output_dir}")
#
#
# if __name__ == "__main__":
#     main()
''' 2nd addition'''

"""
Inference script to generate MRI from CT images
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys

sys.path.append('../../')

from models.conditional_unet import CTMRIDiffusionModel
from improve_diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType
)
import improve_diffusion.dist_util as dist_util


def load_model(checkpoint_path, args):
    """Load trained model from checkpoint"""

    model = CTMRIDiffusionModel(
        image_size=args.image_size,
        in_channels=1,
        model_channels=args.num_channels,
        out_channels=1,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        dropout=0.0,  # No dropout during inference
        num_heads=args.num_heads
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(dist_util.dev())
    model.eval()

    return model


def create_diffusion(args):
    """Create diffusion process for sampling"""
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)

    return GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=True,
    )


def preprocess_ct_image(image_path, image_size=256):
    """Preprocess CT image for inference"""

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    image = Image.open(image_path).convert('L')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return tensor


def postprocess_mri_image(tensor):
    """Convert tensor back to PIL image"""

    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to PIL
    tensor = tensor.squeeze(0).squeeze(0)  # Remove batch and channel dims
    array = (tensor.cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(array)


def generate_mri(model, diffusion, ct_image, num_samples=1, use_ddim=True):
    """Generate MRI image from CT image"""

    batch_size = ct_image.shape[0]

    # Sample noise
    noise = torch.randn(batch_size, 1, ct_image.shape[2], ct_image.shape[3])
    noise = noise.to(dist_util.dev())
    ct_image = ct_image.to(dist_util.dev())

    # Generate samples
    if use_ddim:
        # DDIM sampling (faster)
        samples = diffusion.ddim_sample_loop(
            model,
            shape=noise.shape,
            noise=noise,
            model_kwargs={'condition': ct_image},
            eta=0.0,
            progress=True
        )
    else:
        # DDPM sampling (slower but potentially better quality)
        samples = diffusion.p_sample_loop(
            model,
            shape=noise.shape,
            noise=noise,
            model_kwargs={'condition': ct_image},
            progress=True
        )

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct_image", required=True, help="Path to CT image")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", default="../../results/samples", help="Output directory")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attention_resolutions", default="16,8")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--noise_schedule", default="cosine")
    parser.add_argument("--use_ddim", action="store_true", help="Use DDIM sampling")
    parser.add_argument("--num_samples", type=int, default=1)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(args.checkpoint, args)

    print("Creating diffusion process...")
    diffusion = create_diffusion(args)

    print("Preprocessing CT image...")
    ct_tensor = preprocess_ct_image(args.ct_image, args.image_size)

    print("Generating MRI image...")
    with torch.no_grad():
        generated_mri = generate_mri(
            model, diffusion, ct_tensor,
            num_samples=args.num_samples,
            use_ddim=args.use_ddim
        )

    print("Saving results...")
    for i in range(args.num_samples):
        # Save generated MRI
        mri_image = postprocess_mri_image(generated_mri[i:i + 1])
        output_path = os.path.join(args.output_dir, f"generated_mri_{i}.png")
        mri_image.save(output_path)

        # Save input CT for comparison
        ct_image = postprocess_mri_image(ct_tensor[i:i + 1])
        ct_output_path = os.path.join(args.output_dir, f"input_ct_{i}.png")
        ct_image.save(ct_output_path)

    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
