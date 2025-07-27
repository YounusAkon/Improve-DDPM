"""
Paired dataset loader for CT-MRI image translation
"""

import os
import random
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


def load_paired_data(
        *,
        source_dir,
        target_dir,
        batch_size,
        image_size,
        deterministic=False,
        augment=True
):
    """
    Create a generator over paired (source, target) image pairs.

    :param source_dir: directory containing source images (CT)
    :param target_dir: directory containing target images (MRI)
    :param batch_size: the batch size of each returned pair
    :param image_size: the size to which images are resized
    :param deterministic: if True, yield results in a deterministic order
    :param augment: if True, apply data augmentation
    """
    if not source_dir or not target_dir:
        raise ValueError("Both source and target directories must be specified")

    source_files = _list_image_files_recursively(source_dir)
    target_files = _list_image_files_recursively(target_dir)

    # Ensure paired data by matching filenames
    paired_files = _match_paired_files(source_files, target_files)

    dataset = PairedImageDataset(
        image_size,
        paired_files,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        augment=augment
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )

    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """List all image files in directory recursively"""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "bmp", "tiff"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def _match_paired_files(source_files, target_files):
    """Match source and target files by filename"""
    source_dict = {}
    target_dict = {}

    # Create dictionaries with basename as key
    for path in source_files:
        basename = bf.basename(path)
        name_without_ext = os.path.splitext(basename)[0]
        source_dict[name_without_ext] = path

    for path in target_files:
        basename = bf.basename(path)
        name_without_ext = os.path.splitext(basename)[0]
        target_dict[name_without_ext] = path

    # Find matching pairs
    paired_files = []
    for name in source_dict:
        if name in target_dict:
            paired_files.append((source_dict[name], target_dict[name]))

    if len(paired_files) == 0:
        raise ValueError("No matching paired files found")

    print(f"Found {len(paired_files)} paired images")
    return paired_files


class PairedImageDataset(Dataset):
    """Dataset for paired source-target images"""

    def __init__(self, resolution, paired_files, shard=0, num_shards=1, augment=True):
        super().__init__()
        self.resolution = resolution
        self.local_pairs = paired_files[shard:][::num_shards]
        self.augment = augment

        # Base transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])

        # Augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
            ])
        else:
            self.augment_transform = None

    def __len__(self):
        return len(self.local_pairs)

    def __getitem__(self, idx):
        source_path, target_path = self.local_pairs[idx]

        # Load source image (CT)
        with bf.BlobFile(source_path, "rb") as f:
            source_pil = Image.open(f)
            source_pil.load()

        # Load target image (MRI)
        with bf.BlobFile(target_path, "rb") as f:
            target_pil = Image.open(f)
            target_pil.load()

        # Convert to grayscale for medical images
        source_pil = source_pil.convert("L")
        target_pil = target_pil.convert("L")

        # Apply same augmentation to both images
        if self.augment_transform is not None:
            # Set same random seed for both images
            seed = random.randint(0, 2 ** 32)

            random.seed(seed)
            torch.manual_seed(seed)
            source_pil = self.augment_transform(source_pil)

            random.seed(seed)
            torch.manual_seed(seed)
            target_pil = self.augment_transform(target_pil)

        # Apply base transforms
        source_tensor = self.base_transform(source_pil)
        target_tensor = self.base_transform(target_pil)

        # Normalize to [-1, 1]
        source_tensor = source_tensor * 2.0 - 1.0
        target_tensor = target_tensor * 2.0 - 1.0

        return source_tensor, target_tensor


class MedicalImageDataset(Dataset):
    """
    Specialized dataset for medical images with proper preprocessing
    """

    def __init__(self, source_dir, target_dir, resolution=256,
                 window_source=None, window_target=None, augment=True):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.resolution = resolution
        self.window_source = window_source  # (center, width) for CT windowing
        self.window_target = window_target  # (center, width) for MRI windowing
        self.augment = augment

        # Get paired files
        source_files = _list_image_files_recursively(source_dir)
        target_files = _list_image_files_recursively(target_dir)
        self.paired_files = _match_paired_files(source_files, target_files)

        # Medical image transforms
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])

        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])

    def __len__(self):
        return len(self.paired_files)

    def apply_windowing(self, image_array, window_center, window_width):
        """Apply medical image windowing"""
        if window_center is None or window_width is None:
            return image_array

        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2

        image_array = np.clip(image_array, window_min, window_max)
        image_array = (image_array - window_min) / (window_max - window_min)

        return image_array

    def __getitem__(self, idx):
        source_path, target_path = self.paired_files[idx]

        # Load images
        source_img = Image.open(source_path).convert("L")
        target_img = Image.open(target_path).convert("L")

        # Apply windowing if specified
        if self.window_source:
            source_array = np.array(source_img)
            source_array = self.apply_windowing(source_array, *self.window_source)
            source_img = Image.fromarray((source_array * 255).astype(np.uint8))

        if self.window_target:
            target_array = np.array(target_img)
            target_array = self.apply_windowing(target_array, *self.window_target)
            target_img = Image.fromarray((target_array * 255).astype(np.uint8))

        # Apply augmentation
        if self.augment and self.augment_transform:
            seed = random.randint(0, 2 ** 32)

            random.seed(seed)
            torch.manual_seed(seed)
            source_img = self.augment_transform(source_img)

            random.seed(seed)
            torch.manual_seed(seed)
            target_img = self.augment_transform(target_img)

        # Convert to tensors
        source_tensor = self.transform(source_img)
        target_tensor = self.transform(target_img)

        # Normalize to [-1, 1]
        source_tensor = source_tensor * 2.0 - 1.0
        target_tensor = target_tensor * 2.0 - 1.0

        return source_tensor, target_tensor
