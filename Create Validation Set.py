"""Since the Dataset didn't have a validation set, I used 20% of the training set to make one."""

import os 
import shutil
import random 
from glob import glob 


def create_validation_path(image_path: str) -> str:
    """
    Creates the path for an image within the validation set by modifying the original training set path.
    
    Args:
        image_path (str): The path of the image in training set.
    
    Returns:
        str: The corresponding path for the image within the validation set.

    """
    return image_path.replace('train', 'validation')


def create_validation_mask_path(image_path: str) -> tuple[str, str]:
    """
    Generates paths for both the mask and its corresponding mask in the validation set.
    
    Args:
        image_path (str): The path of the image in training set.
    
    Returns:
        tuple(str, str): A tuple containing:
        A tuple containig two strings:
            - The path of the mask in the training set.
            - The corresponding path for the mask within the validation set.

    """
    current_path = image_path.replace('images', 'masks').replace('.jpg', '.png')
    new_path = current_path.replace('train', 'validation')
    return current_path, new_path


if __name__ == '__main__':
    os.mkdir(r'Data\validation_images')
    os.mkdir(r'Data\validation_masks')

    images_path = glob(r'Data\train_images\*.jpg')
    validation_count = int(len(images_path) * 0.2)
    

    selected_images_path = random.sample(images_path, validation_count)
    images_validation_path = list(map(create_validation_path, selected_images_path))
    masks_paths = list(map(create_validation_mask_path, selected_images_path))

    for i in range(validation_count):
        shutil.move(selected_images_path[i], images_validation_path[i])
        shutil.move(masks_paths[i][0], masks_paths[i][1])
