"""
Respace diffusion timesteps for faster sampling.

This file comes from OpenAI's improved-diffusion repository, slightly adapted
for your project so it works without modification.
"""

import numpy as np

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a set of evenly spaced timesteps to use from the original diffusion process.

    :param num_timesteps: int
        Total number of diffusion steps in the base process.
    :param section_counts: str or list[int]
        Either a comma-separated list of section counts or an integer list.
        For example: "10" means 10 steps total.
                     "10,10,10" means 30 steps split into 3 sections.
                     "ddim25" means use 25 steps in DDIM style.
    :return: set[int]
        The timesteps to retain from the original process.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            # DDIM special case
            count = int(section_counts[len("ddim"):])
            return np.linspace(0, num_timesteps - 1, count, dtype=int).tolist()

        section_counts = [int(x) for x in section_counts.split(",") if x]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    all_steps = []
    start = 0
    for i, count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < count:
            raise ValueError("Too many timesteps requested in section")
        frac_stride = (size - 1) / (count - 1) if count > 1 else 1
        cur_steps = [start + round(frac_stride * j) for j in range(count)]
        all_steps += cur_steps
        start += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process that uses a reduced number of timesteps compared to the base process.
    This is useful for faster sampling.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = kwargs.get("betas").shape[0]

        # Map the reduced timesteps to the original ones
        last_alpha_cumprod = 1.0
        new_betas = []
        for i in range(self.original_num_steps):
            if i in self.use_timesteps:
                alpha_cumprod = kwargs["alphas_cumprod"][i]
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        kwargs["betas"] = np.array(new_betas, dtype=np.float64)
        super().__init__(**kwargs)
