"""Utilities for training and evaluation."""
import os

import torch
import torchvision

from tqdm import tqdm

from torch.utils.data import DataLoader
from dataset import CarvanaDataset


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    checkpoint_path: str = r"model_weights\chechpoint.pth",
) -> None:
    """
    Loads a model checkpoint from the specified path.

    Args:
        model (nn.Module): The model to load the weights into.
        optimizer (Optional[torch.optim.Optimizer], optional): The optimizer
                   to load its state if available in the checkpoint (for training).
                   Defaults to None.
        checkpoint_path (str, optional): Path to the checkpoint file.
                         Defaults to "model_weights/checkpoint.pth".
    """
    print("---  Loading checkpoint  ---")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


def save_checkpoint(
    model_state_dict, optimizer_state_dict, filename: str = "checkpoint.pth.tar"
) -> None:
    """
    Saves the model and optimizer state dictionaries to a checkpoint file.

    Args:
        model_state_dict (Dict[str, Any]): The model state dictionary.
        optimizer_state_dict (Optional[Dict[str, Any]], optional): The optimizer
                                state dictionary.
        filename (str, optional): The filename for the checkpoint file.
                                   Defaults to "checkpoint.pth.tar".
    """
    print("---  Saving checkpoint  ---")
    checkpoint = {"state_dict": model_state_dict, "optimizer": optimizer_state_dict}
    torch.save(checkpoint, filename)


def print_evaluation_metrics(
    model: torch.nn.Module, dataloader: DataLoader, device: str
) -> None:
    """
    Calculates and prints evaluation metrics (accuracy and Dice score) for a given model
    and dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): A DataLoader containing the
        evaluation dataset.
        device (torch.device): The device to use for computation.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for data, targets in tqdm(dataloader):
            data = data.to(device)
            targets = targets.to(device)

            predictions = torch.sigmoid(model(data))
            predictions = (predictions > 0.5).float()

            num_correct += (predictions == targets).sum()
            num_pixels += torch.numel(predictions)

            dice_score += (2.0 * (predictions * targets).sum()) / (
                (predictions + targets).sum() + 1e-8
            )

    print(f"Accuracy: {num_correct}/{num_pixels}={num_correct/num_pixels * 100:.2f}")
    print(f"Dice Score: {dice_score/len(dataloader):.5f}")


def save_predictions_as_mask(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    num_save: int = 10,
    save_directory: str = "saved_images\\",
) -> None:
    """
    Saves input images, predictions and corresponding ground-truth masks from
    a model and dataset.

    Args:
        model (nn.Module): The model used to generate predictions.
        dataloader (torch.utils.data.DataLoader): A DataLoader containing the dataset
                                                  for generating predictions.
        device (torch.device): The device to use for computation.
        num_save (int, optional): The number of images, predictions and masks to save.
                                   Defaults to 10.
        save_directory (str, optional): The directory to save the images.
                                         Defaults to "saved_images/".

    Returns:
        None
    """
    os.makedirs(save_directory, exist_ok=True)
    selected_targets = []
    selected_data = []
    num_selected = 0

    for data, targets in dataloader:

        if len(data) == 1:
            random_index = 0
        else:
            random_index = torch.randint(high=len(data) - 1, size=(1,)).item()

        selected_data.append(data[random_index])
        selected_targets.append(targets[random_index])

        num_selected += 1
        if num_selected >= num_save:
            break

    selected_data = torch.stack(selected_data)
    selected_targets = torch.stack(selected_targets)

    model.to(device)
    with torch.no_grad():
        predictions = model(selected_data.to(device))
        predictions = (predictions > 0.5).float()

    for idx, prediction in enumerate(predictions):
        torchvision.utils.save_image(prediction, save_directory + f"{idx}_pred.jpg")
        torchvision.utils.save_image(
            selected_targets[idx], save_directory + f"{idx}_mask.jpg"
        )
        torchvision.utils.save_image(selected_data[idx], save_directory + f"{idx}.jpg")


def get_dataloader(
    images_directory: str,
    masks_directory: str,
    batch_size: int,
    shuffle: bool = True,
    transform=None,
) -> DataLoader:
    """
    Creates and returns a PyTorch DataLoader.

    Args:
        images_directory (str): Path to the directory containing the image files.
        masks_directory (str): Path to the directory containing the corresponding
                               mask files.
        batch_size (int): The batch size for the dataloader.
        shuffle (bool, optional): Whether to shuffle the data before creating batches.
                                  Defaults to True.
        transform (Optional[Callable], optional): Transform(s).
                                                  Defaults to None.

    Returns:
        torch.utils.data.DataLoader: The created DataLoader for dataset.
    """
    dataset = CarvanaDataset(images_directory, masks_directory, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
