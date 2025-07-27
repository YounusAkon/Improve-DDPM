"""
Conditional UNet for CT to MRI Translation
Modified UNet that takes CT images as conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from unet import UNetModel, TimestepEmbedSequential, ResBlock, AttentionBlock
from nn import conv_nd, linear, zero_module, normalization, timestep_embedding


class ConditionalUNetModel(UNetModel):
    """
    UNet model conditioned on CT images for MRI generation
    """

    def __init__(self,
                 condition_channels=1,  # CT image channels
                 *args, **kwargs):
        # Modify input channels to include conditioning
        original_in_channels = kwargs.get('in_channels', 1)
        kwargs['in_channels'] = original_in_channels + condition_channels

        super().__init__(*args, **kwargs)

        self.condition_channels = condition_channels
        self.original_in_channels = original_in_channels

        # Conditioning encoder
        self.condition_encoder = nn.Sequential(
            conv_nd(2, condition_channels, 64, 3, padding=1),
            nn.ReLU(),
            conv_nd(2, 64, 128, 3, padding=1),
            nn.ReLU(),
            conv_nd(2, 128, self.model_channels, 3, padding=1)
        )

    def forward(self, x, timesteps, condition=None, **kwargs):
        """
        Forward pass with CT conditioning

        Args:
            x: Target MRI image (noisy)
            timesteps: Diffusion timesteps
            condition: CT image for conditioning
        """
        if condition is not None:
            # Encode conditioning information
            cond_features = self.condition_encoder(condition)

            # Concatenate along channel dimension
            x = torch.cat([x, condition], dim=1)

        return super().forward(x, timesteps, **kwargs)


class CTMRIDiffusionModel(nn.Module):
    """
    Complete CT to MRI diffusion model
    """

    def __init__(self,
                 image_size=256,
                 in_channels=1,
                 model_channels=128,
                 out_channels=1,
                 num_res_blocks=2,
                 attention_resolutions="16,8",
                 dropout=0.1,
                 channel_mult=(1, 2, 4, 8),
                 use_checkpoint=False,
                 num_heads=4):
        super().__init__()

        # Parse attention resolutions
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        self.unet = ConditionalUNetModel(
            condition_channels=in_channels,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            use_checkpoint=use_checkpoint,
            num_heads=num_heads,
            use_scale_shift_norm=True
        )

    def forward(self, x, timesteps, condition=None):
        return self.unet(x, timesteps, condition=condition)
