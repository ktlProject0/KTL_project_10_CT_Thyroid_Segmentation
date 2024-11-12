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
    def __init__(self, direc, mode='eval', window_center=40, window_width=400, output_shape=(96, 128, 128)):
        self.mode = mode
        self.window_center = window_center
        self.window_width = window_width
        self.output_shape = output_shape
        img_path = natsorted(glob.glob(os.path.join(direc, 'images', '*')))
        mask_path = natsorted(glob.glob(os.path.join(direc, 'masks', '*')))
        self.meta_df = pd.DataFrame({"image": img_path, 'label': mask_path})
        self.cache = {}

    def __len__(self):
        return len(self.meta_df)

    def apply_window(self, image):
        min_value = self.window_center - self.window_width / 2
        max_value = self.window_center + self.window_width / 2
        image = np.clip(image, min_value, max_value)
        image = ((image - min_value) / (max_value - min_value)) * 255
        return image.astype(np.uint8)

    def __getitem__(self, idx):
        if idx in self.cache:
            sample = self.cache[idx]
        else:
            sample = self.meta_df.iloc[idx, :].to_dict()
            image = sitk.GetArrayFromImage(sitk.ReadImage(sample['image']))
            image = self.apply_window(image) / 255.0
            image = (image - 0.5) / 0.5  # [-1, 1]로 정규화

            # 마스크 데이터는 [0, 1] 값만 남도록 처리
            mask = sitk.GetArrayFromImage(sitk.ReadImage(sample['label'])).astype(np.float32)
            mask = np.clip(mask, 0, 1)  # [0, 1] 범위 유지

            # 이미지 및 마스크 리사이즈
            image = torch.nn.functional.interpolate(
                torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                size=self.output_shape,
                mode='trilinear',
                align_corners=False
            ).squeeze(0)
            mask = torch.nn.functional.interpolate(
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                size=self.output_shape,
                mode='nearest'
            ).squeeze(0)

            sample['image'] = image
            sample['mask'] = mask
            self.cache[idx] = sample

        # 데이터 증강 적용
        sample_ = copy.deepcopy(sample)
        if self.mode == 'train':
            if np.random.rand() > 0.5:
                sample_['image'] = sample_['image'].flip(dims=(2,))
                sample_['mask'] = sample_['mask'].flip(dims=(2,))
            if np.random.rand() > 0.5:
                sample_['image'] = sample_['image'].flip(dims=(3,))
                sample_['mask'] = sample_['mask'].flip(dims=(3,))

        return {'input': sample_['image'], 'target': sample_['mask']}

if __name__ == '__main__':
    train_dataset = CustomDataset(direc='./data/train', mode='train')
    test_dataset = CustomDataset(direc='./data/test', mode='test')
    
    for sample in train_dataset:
        print(sample['input'].shape, sample['target'].shape)
