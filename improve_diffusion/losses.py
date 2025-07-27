"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py

Extended with L1 and Perceptual losses for medical image translation.
"""

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + th.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


class L1Loss(nn.Module):
    """L1 Loss for medical image translation"""

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features
    Adapted for grayscale medical images
    """

    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                 weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()

        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers
        self.weights = weights

        # Extract specific layers
        self.feature_extractor = nn.ModuleDict()
        layer_names = {
            '3': 'relu1_2',
            '8': 'relu2_2',
            '15': 'relu3_3',
            '22': 'relu4_3'
        }

        for i, layer in enumerate(vgg):
            if str(i) in layer_names:
                self.feature_extractor[layer_names[str(i)]] = nn.Sequential(*list(vgg.children())[:i + 1])

        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # Convert grayscale to RGB if needed
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        loss = 0.0
        for i, layer_name in enumerate(self.layers):
            if layer_name in self.feature_extractor:
                pred_features = self.feature_extractor[layer_name](pred)
                target_features = self.feature_extractor[layer_name](target)
                loss += self.weights[i] * F.mse_loss(pred_features, target_features)

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for medical image translation
    Combines MSE, L1, and Perceptual losses
    """

    def __init__(self, mse_weight=1.0, l1_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight

        self.l1_loss = L1Loss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, pred, target):
        # MSE Loss
        mse_loss = F.mse_loss(pred, target)

        # L1 Loss
        l1_loss = self.l1_loss(pred, target)

        # Perceptual Loss
        perceptual_loss = self.perceptual_loss(pred, target)

        # Combined loss
        total_loss = (
                self.mse_weight * mse_loss +
                self.l1_weight * l1_loss +
                self.perceptual_weight * perceptual_loss
        )

        return {
            'total': total_loss,
            'mse': mse_loss,
            'l1': l1_loss,
            'perceptual': perceptual_loss
        }


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Loss
    Useful for medical image quality assessment
    """

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = th.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        ssim_value = self.ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return 1 - ssim_value  # Convert to loss (lower is better)
