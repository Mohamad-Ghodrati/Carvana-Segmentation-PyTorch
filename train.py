from typing import Callable

import torch
import torch.nn as nn
import torchvision

from tqdm import tqdm

from model import UNET
from hyperparameters import Hyperparameters
from utils import (
    load_checkpoint,
    save_checkpoint,
    print_evaluation_metrics,
    save_predictions_as_mask,
    get_dataloader,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    scaler: torch.cuda.amp.GradScaler = None,
    epoch_number: int = 0,
    device: str = "cuda",
) -> None:
    """
     Trains the model for a single epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataloader (torch.utils.data.DataLoader): The dataloader
                                                  containing training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (Callable[torch.Tensor, torch.Tensor]): Loss function.
        scaler (Optional[torch.cuda.amp.GradScaler], optional):
                                    GradScaler for mixed-precision training
                                    (if applicable). Defaults to None.
        epoch_number (int, optional): The current epoch number. Defaults to 0.
        device (str, optional): The device to use for computation.
                 Defaults to 'cuda'.

    """
    dataloader_iterator = tqdm(dataloader, colour="green", leave=False)

    model.train()
    model.to(device)

    for batch, (data, targets) in enumerate(dataloader_iterator):
        data = data.to(device)
        targets = targets.to(device)

        if device == "cuda" and scaler is not None:
            loss = perform_mixed_precision_training_step(
                model, optimizer, loss_fn, scaler, data, targets
            )

        else:
            loss = perform_training_step(model, optimizer, loss_fn, data, targets)

        dataloader_iterator.set_description(
            f"[EPOCH {epoch_number}] [Batch {batch+1}/{len(dataloader)}]"
        )
        dataloader_iterator.set_postfix(batch_loss=loss.item())


def perform_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Performs a single training step for the model."""
    predictions = model(data)
    loss = loss_fn(predictions, targets)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


def perform_mixed_precision_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    scaler: torch.cuda.amp.GradScaler,
    data: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Performs a single training step for the model using mixed precision."""
    with torch.cuda.amp.autocast():
        predictions = model(data)
        loss = loss_fn(predictions, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    return loss


def main():
    hyperparams = Hyperparameters()
    model = UNET(in_channels=3, out_channels=1).to(hyperparams.device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    transform = torchvision.transforms.Resize(
        (hyperparams.image_height, hyperparams.image_width)
    )

    train_dataloader = get_dataloader(
        hyperparams.train_image_directory,
        hyperparams.train_mask_directory,
        batch_size=hyperparams.batch_size,
        shuffle=True,
        transform=transform,
    )
    validation_dataloader = get_dataloader(
        hyperparams.val_image_directory,
        hyperparams.val_mask_directory,
        batch_size=hyperparams.batch_size,
        shuffle=False,
        transform=transform,
    )

    if hyperparams.load_model:
        load_checkpoint(model, optimizer, hyperparams.checkpoint_path)

    for epoch in range(hyperparams.num_epochs):
        train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            loss_fn,
            scaler,
            epoch + 1,
            hyperparams.device,
        )

    save_checkpoint(
        model.state_dict(), optimizer.state_dict(), "model_weights\\chechpoint.pth"
    )

    print_evaluation_metrics(model, validation_dataloader, hyperparams.device)

    save_predictions_as_mask(model, validation_dataloader, hyperparams.device)


if __name__ == "__main__":
    main()
