import os
import glob
import copy
from natsort import natsorted
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, direc, mode='eval', window_center=40, window_width=400, output_shape=(256, 256)):
        self.mode = mode
        self.window_center = window_center
        self.window_width = window_width
        self.output_shape = output_shape

        # Load file paths
        img_path = natsorted(glob.glob(os.path.join(direc, 'images', '*')))
        mask_path = natsorted(glob.glob(os.path.join(direc, 'masks', '*')))
        self.meta_df = pd.DataFrame({"image": img_path, 'label': mask_path})

        # Create slice information
        self.slice_info = []
        for idx, row in self.meta_df.iterrows():
            image = sitk.GetArrayFromImage(sitk.ReadImage(row['image']))
            depth = image.shape[0]  # Number of slices
            for slice_idx in range(depth):
                self.slice_info.append((idx, slice_idx))

    def __len__(self):
        return len(self.slice_info)

    def apply_window(self, image):
        min_value = self.window_center - self.window_width / 2
        max_value = self.window_center + self.window_width / 2
        image = np.clip(image, min_value, max_value)
        image = ((image - min_value) / (max_value - min_value)) * 255
        return image.astype(np.uint8)

    def __getitem__(self, idx):
        # Get the file and slice indices
        file_idx, slice_idx = self.slice_info[idx]
        sample = self.meta_df.iloc[file_idx, :].to_dict()

        # Load 3D image and mask
        image = sitk.GetArrayFromImage(sitk.ReadImage(sample['image']))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(sample['label']))

        # Extract the 2D slice
        image_slice = image[slice_idx, :, :]
        mask_slice = mask[slice_idx, :, :]

        # Apply windowing and normalization to the image
        image_slice = self.apply_window(image_slice) / 255.0
        image_slice = (image_slice - 0.5) / 0.5  # Normalize to [-1, 1]

        # Ensure mask is binary
        mask_slice = np.clip(mask_slice, 0, 1).astype(np.float32)

        # Resize image and mask
        image_tensor = torch.nn.functional.interpolate(
            torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=self.output_shape,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        mask_tensor = torch.nn.functional.interpolate(
            torch.tensor(mask_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            size=self.output_shape,
            mode='nearest'
        ).squeeze(0)

        sample = {'image': image_tensor, 'mask': mask_tensor}

        # Apply augmentation for training mode
        if self.mode == 'train':
            sample = self._augment_sample(sample)

        return {'input': sample['image'], 'target': sample['mask']}

    def _augment_sample(self, sample):
        image, mask = sample['image'], sample['mask']
        if torch.rand(1).item() > 0.5:
            image = image.flip(dims=(1,))
            mask = mask.flip(dims=(1,))
        if torch.rand(1).item() > 0.5:
            image = image.flip(dims=(2,))
            mask = mask.flip(dims=(2,))
        return {'image': image, 'mask': mask}


if __name__ == '__main__':
    train_dataset = CustomDataset(direc='./data/train', mode='train', output_shape=(256, 256))
    test_dataset = CustomDataset(direc='./data/test', mode='eval', output_shape=(256, 256))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
