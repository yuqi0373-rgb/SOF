import os
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DATASETS = {
    'busi': dict(directory='busi', image_extension='.png', mask_suffix='_mask.png', size=256),
    'glas': dict(directory='glas', image_extension='.png', mask_suffix='.png', size=512),
    'cvc': dict(directory='cvc', image_extension='.png', mask_suffix='.png', size=256),
    'isic17': dict(directory='ISIC17', image_extension='.jpg', mask_suffix='_segmentation.png', size=256),
}


def dataset_directory(root, dataset):
    path = Path(root) / DATASETS[dataset]['directory']
    if not path.exists() and dataset == 'cvc':
        path = Path(root) / 'chasedb1'
    return path


def transforms(size, training=False):
    operations = [A.RandomRotate90(), A.HorizontalFlip()] if training else []
    return A.Compose(operations + [A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


def read_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return image


def preprocess(image, size):
    value = transforms(size)(image=image)['image']
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1))).float()


class SegmentationDataset(Dataset):
    def __init__(self, root, dataset, image_ids, size, training=False):
        self.root = dataset_directory(root, dataset)
        self.spec = DATASETS[dataset]
        self.image_ids = list(image_ids)
        self.transform = transforms(size, training)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = read_image(self.root / 'images' / (image_id + self.spec['image_extension']))
        path = self.root / 'masks' / '0' / (image_id + self.spec['mask_suffix'])
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f'Cannot read mask: {path}')
        sample = self.transform(image=image, mask=mask)
        x = torch.from_numpy(np.ascontiguousarray(sample['image'].transpose(2, 0, 1))).float()
        y = torch.from_numpy(np.ascontiguousarray(sample['mask'][None] > 0)).float()
        return x, y
