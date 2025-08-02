"""
CT to MRI Diffusion Model Project Setup
"""

import os


def create_project_structure():
    """Create the complete project directory structure"""

    directories = [
        "data/ct_images",
        "data/mri_images",
        "data/processed",
        "models/checkpoints",
        "models/configs",
        "logs",
        "results/samples",
        "results/evaluation",
        "scripts/training",
        "scripts/inference",
        "utils"
    ]

    print("Creating project structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")

    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Place your CT images in data/ct_images/")
    print("2. Place your MRI images in data/mri_images/")
    print("3. Run data preprocessing")
    print("4. Start training")


if __name__ == "__main__":
    create_project_structure()
