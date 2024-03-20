import os
import numpy as np

from torchvision.io import read_image
from torch.utils.data import Dataset


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_names[index])
        mask_path = os.path.join(self.mask_dir, self.image_names[index].replace('jpg', 'png'))
        image = read_image(image_path) / 255
        mask = read_image(mask_path).float()

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def __len__(self):
        return len(self.image_names)
