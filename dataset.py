"""
This module provides a custom dataset class specifically designed for loading and 
managing image segmentation data from the Carvana dataset. It inherits from PyTorch's 
`torch.utils.data.Dataset` class.

Example:
    ```python
    from dataset import CarvanaDataset

    train_images_dir = "/path/to/carvana/train_images"
    train_masks_dir = "/path/to/carvana/train_masks"

    dataset = CarvanaDataset(train_images_dir, train_masks_dir)

    image, mask = dataset[0]  # Access image and mask for the first sample
    ```
"""

import os
from typing import Optional, Callable, Tuple

from torchvision.io import read_image
from torch.utils.data import Dataset
from torch import Tensor


class CarvanaDataset(Dataset):
    """
    A custom dataset class specifically designed for loading
    and managing image segmentation data from the Carvana dataset.

    Dataset Structure:
        The Carvana dataset is expected to have the following directory structure:

        carvana_data/
        ├── train_images/
        │   ├── image_name.jpg
        │   └── ... (other image files)
        └── train_masks/
            ├── image_name.png  (corresponding mask for image_name.jpg)
            └── ... (other mask files)

    Args:
        image_dir (str): Path to the directory containing the images in
                         JPEG format.
        mask_dir (str): Path to the directory containing the corresponding
                        segmentation masks (.png) for the training images.
        transform (callable, optional): A function or transform object to apply to
                                        the loaded data. Defaults to None.

    Note:
        The default assumption is that the image values range from 0-255.
        Similarly, the segmentation masks are assumed to be binary (0 or 1 values).
        Change
        ```python
        image = read_image(image_path) / 255
        mask = read_image(mask_path).float()
        ```
        if needed.
    """

    def __init__(
        self, image_dir: str, mask_dir: str, transform: Optional[Callable] = None
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_names = os.listdir(image_dir)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image_path = os.path.join(self.image_dir, self.image_names[index])
        mask_path = os.path.join(
            self.mask_dir, self.image_names[index].replace("jpg", "png")
        )
        image = read_image(image_path) / 255
        mask = read_image(mask_path).float()

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

    def __len__(self) -> int:
        return len(self.image_names)
