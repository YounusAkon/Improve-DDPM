"""
Data preprocessing utilities for medical images
"""

import os
import numpy as np
import nibabel as nib
from PIL import Image
import cv2
from pathlib import Path
import argparse


def normalize_medical_image(image_array, window_center=None, window_width=None):
    """
    Normalize medical image with optional windowing
    """
    if window_center is not None and window_width is not None:
        # Apply windowing
        window_min = window_center - window_width // 2
        window_max = window_center + window_width // 2
        image_array = np.clip(image_array, window_min, window_max)

    # Normalize to 0-255
    image_array = image_array - image_array.min()
    image_array = image_array / (image_array.max() + 1e-8) * 255

    return image_array.astype(np.uint8)


def extract_slices_from_nifti(nifti_path, output_dir, prefix,
                              slice_range=(50, 150),
                              window_center=None, window_width=None):
    """
    Extract 2D slices from 3D NIfTI file
    """
    # Load NIfTI file
    nifti_img = nib.load(nifti_path)
    image_data = nifti_img.get_fdata()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract slices
    start_slice = max(0, slice_range[0])
    end_slice = min(image_data.shape[2], slice_range[1])

    extracted_slices = []

    for slice_idx in range(start_slice, end_slice):
        slice_data = image_data[:, :, slice_idx]

        # Skip empty slices
        if slice_data.max() == 0:
            continue

        # Normalize
        normalized_slice = normalize_medical_image(
            slice_data, window_center, window_width
        )

        # Save as PNG
        slice_filename = f"{prefix}_slice_{slice_idx:03d}.png"
        slice_path = os.path.join(output_dir, slice_filename)

        Image.fromarray(normalized_slice).save(slice_path)
        extracted_slices.append(slice_path)

    return extracted_slices


def preprocess_ct_mri_dataset(ct_dir, mri_dir, output_dir,
                              ct_window=(40, 400), mri_window=None):
    """
    Preprocess paired CT-MRI dataset
    """
    ct_input_dir = Path(ct_dir)
    mri_input_dir = Path(mri_dir)

    ct_output_dir = os.path.join(output_dir, "ct_images")
    mri_output_dir = os.path.join(output_dir, "mri_images")

    os.makedirs(ct_output_dir, exist_ok=True)
    os.makedirs(mri_output_dir, exist_ok=True)

    # Get paired files
    ct_files = sorted(list(ct_input_dir.glob("*.nii*")))
    mri_files = sorted(list(mri_input_dir.glob("*.nii*")))

    assert len(ct_files) == len(mri_files), "CT and MRI file counts must match"

    print(f"Processing {len(ct_files)} paired volumes...")

    for i, (ct_file, mri_file) in enumerate(zip(ct_files, mri_files)):
        print(f"Processing volume {i + 1}/{len(ct_files)}: {ct_file.name}")

        # Extract CT slices
        ct_slices = extract_slices_from_nifti(
            ct_file, ct_output_dir, f"ct_{i:03d}",
            window_center=ct_window[0], window_width=ct_window[1]
        )

        # Extract MRI slices
        mri_slices = extract_slices_from_nifti(
            mri_file, mri_output_dir, f"mri_{i:03d}",
            window_center=mri_window[0] if mri_window else None,
            window_width=mri_window[1] if mri_window else None
        )

        print(f"  Extracted {len(ct_slices)} CT slices and {len(mri_slices)} MRI slices")

    print("Preprocessing completed!")


def create_train_val_split(data_dir, val_ratio=0.2):
    """
    Create train/validation split
    """
    ct_dir = os.path.join(data_dir, "ct_images")
    mri_dir = os.path.join(data_dir, "mri_images")

    ct_files = sorted(os.listdir(ct_dir))
    mri_files = sorted(os.listdir(mri_dir))

    # Group by volume
    volumes = {}
    for ct_file in ct_files:
        volume_id = ct_file.split('_slice_')[0]
        if volume_id not in volumes:
            volumes[volume_id] = {'ct': [], 'mri': []}
        volumes[volume_id]['ct'].append(ct_file)

    for mri_file in mri_files:
        volume_id = mri_file.split('_slice_')[0]
        if volume_id in volumes:
            volumes[volume_id]['mri'].append(mri_file)

    # Split volumes
    volume_ids = list(volumes.keys())
    np.random.shuffle(volume_ids)

    split_idx = int(len(volume_ids) * (1 - val_ratio))
    train_volumes = volume_ids[:split_idx]
    val_volumes = volume_ids[split_idx:]

    # Create split directories
    for split_name, volume_list in [('train', train_volumes), ('val', val_volumes)]:
        split_ct_dir = os.path.join(data_dir, f"{split_name}_ct")
        split_mri_dir = os.path.join(data_dir, f"{split_name}_mri")

        os.makedirs(split_ct_dir, exist_ok=True)
        os.makedirs(split_mri_dir, exist_ok=True)

        for volume_id in volume_list:
            # Copy CT files
            for ct_file in volumes[volume_id]['ct']:
                src = os.path.join(ct_dir, ct_file)
                dst = os.path.join(split_ct_dir, ct_file)
                os.symlink(src, dst)

            # Copy MRI files
            for mri_file in volumes[volume_id]['mri']:
                src = os.path.join(mri_dir, mri_file)
                dst = os.path.join(split_mri_dir, mri_file)
                os.symlink(src, dst)

    print(f"Created train/val split: {len(train_volumes)} train, {len(val_volumes)} val volumes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct_dir", required=True, help="Directory containing CT NIfTI files")
    parser.add_argument("--mri_dir", required=True, help="Directory containing MRI NIfTI files")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed images")
    parser.add_argument("--ct_window_center", type=int, default=40, help="CT window center")
    parser.add_argument("--ct_window_width", type=int, default=400, help="CT window width")
    parser.add_argument("--create_split", action="store_true", help="Create train/val split")

    args = parser.parse_args()

    # Preprocess dataset
    preprocess_ct_mri_dataset(
        args.ct_dir, args.mri_dir, args.output_dir,
        ct_window=(args.ct_window_center, args.ct_window_width)
    )

    # Create train/val split if requested
    if args.create_split:
        create_train_val_split(args.output_dir)


if __name__ == "__main__":
    main()
