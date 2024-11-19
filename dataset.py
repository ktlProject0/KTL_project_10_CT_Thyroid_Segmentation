import os
import glob
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, direc, mode='eval', output_shape=(128, 128, 128)):
        """
        Custom dataset for loading NIfTI image and mask pairs and resizing to the specified output shape.
        """
        self.mode = mode
        self.output_shape = output_shape
        
        # Get image and mask file paths
        self.image_paths = sorted(glob.glob(os.path.join(direc, 'images', '*.nii.gz')) + 
                                  glob.glob(os.path.join(direc, 'images', '*.nii')))
        self.mask_paths = sorted(glob.glob(os.path.join(direc, 'masks', '*.nii.gz')) + 
                                 glob.glob(os.path.join(direc, 'masks', '*.nii')))

        if not self.image_paths or not self.mask_paths:
            raise FileNotFoundError(f"No NIfTI files found in directory: {direc}/images or {direc}/masks")

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Number of images and masks do not match.")

    def __len__(self):
        """
        Return the number of NIfTI files in the directory.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and preprocess a single NIfTI image and mask pair at the given index.
        """
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

        # Normalize the image to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Ensure the mask is binary
        mask = (mask > 127).astype(np.float32) 

        # Resize the image and mask to the desired output shape
        image_tensor = torch.nn.functional.interpolate(
            torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            size=self.output_shape,
            mode='trilinear',
            align_corners=False
        ).squeeze(0)  # Remove the added dimensions

        mask_tensor = torch.nn.functional.interpolate(
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=self.output_shape,
            mode='nearest'
        ).squeeze(0)

        # Apply augmentation if in training mode
        if self.mode == 'train':
            image_tensor, mask_tensor = self._augment_data(image_tensor, mask_tensor)

        return {'input': image_tensor, 'target': mask_tensor}

    def _augment_data(self, image_tensor, mask_tensor):
        """
        Apply augmentations to image and mask tensors.
        """
        # Convert tensors to numpy arrays
        image_np = image_tensor.numpy()
        mask_np = mask_tensor.numpy()

        # Random flip along axes
        if np.random.rand() > 0.5:  # Flip along depth
            image_np = np.flip(image_np, axis=0).copy()
            mask_np = np.flip(mask_np, axis=0).copy()
        if np.random.rand() > 0.5:  # Flip along height
            image_np = np.flip(image_np, axis=1).copy()
            mask_np = np.flip(mask_np, axis=1).copy()
        if np.random.rand() > 0.5:  # Flip along width
            image_np = np.flip(image_np, axis=2).copy()
            mask_np = np.flip(mask_np, axis=2).copy()

        # Random rotation (90-degree steps)
        if np.random.rand() > 0.5:
            k = np.random.choice([1, 2, 3])  # Random number of 90-degree rotations
            image_np = np.rot90(image_np, k, axes=(1, 2)).copy()
            mask_np = np.rot90(mask_np, k, axes=(1, 2)).copy()

        # Add Gaussian noise to the image
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, image_np.shape)
            image_np = image_np + noise
            image_np = np.clip(image_np, 0, 1)  # Ensure the range is still [0, 1]

        # Convert back to tensors
        return torch.tensor(image_np, dtype=torch.float32), torch.tensor(mask_np, dtype=torch.float32)


if __name__ == '__main__':
    # Define directories for training and testing datasets
    train_dir = './data/train'  # Root directory containing 'images/' and 'masks/'
    test_dir = './data/test'    # Root directory containing 'images/' and 'masks/'

    # Create train and test datasets
    train_dataset = CustomDataset(direc=train_dir, mode='train', output_shape=(256, 256, 256))
    test_dataset = CustomDataset(direc=test_dir, mode='eval', output_shape=(256, 256, 256))

    print(f"[INFO] Number of training samples: {len(train_dataset)}")
    print(f"[INFO] Number of testing samples: {len(test_dataset)}")
