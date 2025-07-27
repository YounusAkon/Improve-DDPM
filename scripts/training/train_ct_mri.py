"""
Training script for CT to MRI diffusion model
"""

import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

# Import diffusion components
import sys

sys.path.append('../../')

from gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule
from gaussian_diffusion import ModelMeanType, ModelVarType, LossType
from train_util import TrainLoop
from utils.medical_dataset import CTMRIDataset, create_data_loader
from models.conditional_unet import CTMRIDiffusionModel
from resample import create_named_schedule_sampler
import dist_util
import logger


def create_argparser():
    defaults = dict(
        data_dir="../../data",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=4,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,

        # Model parameters
        image_size=256,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        attention_resolutions="16,8",
        dropout=0.1,

        # Diffusion parameters
        diffusion_steps=1000,
        noise_schedule="cosine",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
    )

    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true" if v else "store_false")
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))

    return parser


def create_model_and_diffusion(args):
    """Create model and diffusion process"""

    # Create model
    model = CTMRIDiffusionModel(
        image_size=args.image_size,
        in_channels=1,  # Grayscale medical images
        model_channels=args.num_channels,
        out_channels=1,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        dropout=args.dropout,
        num_heads=args.num_heads,
        use_checkpoint=True
    )

    # Create diffusion process
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)

    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
        rescale_timesteps=args.rescale_timesteps,
    )

    return model, diffusion


def main():
    args = create_argparser().parse_args()

    # Setup distributed training
    dist_util.setup_dist()
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args)
    model.to(dist_util.dev())

    logger.log("Creating data loader...")

    # Create dataset
    dataset = CTMRIDataset(
        ct_dir=os.path.join(args.data_dir, "ct_images"),
        mri_dir=os.path.join(args.data_dir, "mri_images"),
        image_size=args.image_size,
        mode='train'
    )

    # Create data loader
    data_loader = create_data_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    # Convert to iterator
    def data_generator():
        while True:
            for batch in data_loader:
                # Prepare batch for diffusion training
                mri_images = batch['mri'].to(dist_util.dev())
                ct_images = batch['ct'].to(dist_util.dev())

                # Return target images and conditioning
                yield mri_images, {'condition': ct_images}

    data = data_generator()

    logger.log("Creating schedule sampler...")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


if __name__ == "__main__":
    main()
