"""
Medical Image Dataset for CT to MRI Translation
Paired dataset loader for diffusion model training
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import nibabel as nib
from pathlib import Path


class CTMRIDataset(Dataset):
    """
    Dataset for paired CT-MRI images
    Supports both DICOM and NIfTI formats
    """

    def __init__(self, ct_dir, mri_dir, image_size=256, mode='train', transform=None):
        self.ct_dir = Path(ct_dir)
        self.mri_dir = Path(mri_dir)
        self.image_size = image_size
        self.mode = mode

        # Get paired image files
        self.ct_files = sorted(list(self.ct_dir.glob('*.png')))  # Assuming preprocessed PNG files
        self.mri_files = sorted(list(self.mri_dir.glob('*.png')))

        # Ensure paired data
        assert len(self.ct_files) == len(self.mri_files), "CT and MRI file counts must match"

        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.ct_files)

    def __getitem__(self, idx):
        # Load CT image (condition)
        ct_path = self.ct_files[idx]
        ct_image = Image.open(ct_path).convert('L')  # Grayscale
        ct_tensor = self.transform(ct_image)

        # Load MRI image (target)
        mri_path = self.mri_files[idx]
        mri_image = Image.open(mri_path).convert('L')  # Grayscale
        mri_tensor = self.transform(mri_image)

        return {
            'ct': ct_tensor,
            'mri': mri_tensor,
            'ct_path': str(ct_path),
            'mri_path': str(mri_path)
        }


class NIfTIDataset(Dataset):
    """
    Dataset for NIfTI format medical images
    """

    def __init__(self, ct_dir, mri_dir, slice_range=(50, 150), image_size=256):
        self.ct_dir = Path(ct_dir)
        self.mri_dir = Path(mri_dir)
        self.slice_range = slice_range
        self.image_size = image_size

        self.ct_files = sorted(list(self.ct_dir.glob('*.nii*')))
        self.mri_files = sorted(list(self.mri_dir.glob('*.nii*')))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Build slice index
        self.slice_indices = []
        for i, (ct_file, mri_file) in enumerate(zip(self.ct_files, self.mri_files)):
            ct_img = nib.load(ct_file)
            num_slices = ct_img.shape[2]

            start_slice = max(0, self.slice_range[0])
            end_slice = min(num_slices, self.slice_range[1])

            for slice_idx in range(start_slice, end_slice):
                self.slice_indices.append((i, slice_idx))

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_indices[idx]

        # Load CT slice
        ct_img = nib.load(self.ct_files[file_idx])
        ct_slice = ct_img.get_fdata()[:, :, slice_idx]
        ct_slice = self.normalize_slice(ct_slice)
        ct_tensor = self.transform(ct_slice)

        # Load MRI slice
        mri_img = nib.load(self.mri_files[file_idx])
        mri_slice = mri_img.get_fdata()[:, :, slice_idx]
        mri_slice = self.normalize_slice(mri_slice)
        mri_tensor = self.transform(mri_slice)

        return {
            'ct': ct_tensor,
            'mri': mri_tensor,
            'file_idx': file_idx,
            'slice_idx': slice_idx
        }

    def normalize_slice(self, slice_data):
        """Normalize slice to 0-255 range"""
        slice_data = np.clip(slice_data, 0, np.percentile(slice_data, 99))
        slice_data = (slice_data / slice_data.max() * 255).astype(np.uint8)
        return slice_data


def create_data_loader(dataset, batch_size=4, shuffle=True, num_workers=4):
    """Create data loader with proper settings"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
