"""
Since the Dataset didn't have a validation set, I used 20% of the training set 
to create one.
"""

import os
import sys
import shutil
import random
import glob

from typing import Tuple


def create_validation_path(image_path: str) -> str:
    """
    Creates path for an image within the validation set by modifying
    the original training set path.

    Args:
        image_path (str): The path of the image in training set.

    Returns:
        str: The corresponding path for the image within the validation set.
    """
    return image_path.replace("train", "validation")


def create_validation_mask_path(image_path: str) -> Tuple[str, str]:
    """
    Generates paths for both the corresponding mask (in the training set)
    and its counterpart in the validation set based on the image path.

    Args:
        image_path (str): The path of the image in training set.

    Returns:
        tuple(str, str): A tuple containig two strings:
            - Path of the mask in the training set.
            - Corresponding path for the mask within the validation set.
    """
    current_path = image_path.replace("images", "masks").replace(".jpg", ".png")
    new_path = current_path.replace("train", "validation")
    return current_path, new_path


if __name__ == "__main__":
    os.makedirs(r"Data\validation_images", exist_ok=True)
    os.makedirs(r"Data\validation_masks", exist_ok=True)
    if os.listdir(r"Data\validation_images"):
        print("Validation set already exists.")
        sys.exit(0)

    VALIDATION_PERCENT = 20
    images_path = glob.glob(r"Data\train_images\*.jpg")
    validation_count = int(len(images_path) * VALIDATION_PERCENT / 100)

    selected_images_path = random.sample(images_path, validation_count)
    images_validation_path = list(map(create_validation_path, selected_images_path))
    masks_paths = list(map(create_validation_mask_path, selected_images_path))

    for i in range(validation_count):
        shutil.move(selected_images_path[i], images_validation_path[i])
        shutil.move(masks_paths[i][0], masks_paths[i][1])
