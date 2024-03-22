"""Training Hyperparamaters."""

from torch.cuda import is_available


class Hyperparameters:
    def __init__(self):
        self.learning_rate = 1e-4
        self.device = "cuda" if is_available() else "cpu"
        self.batch_size = 1
        self.num_epochs = 3
        self.image_height = 160
        self.image_width = 240
        self.load_model = True
        self.checkpoint_path = r"model_weights\chechpoint (2024_03_21 19_24_06 UTC).pth"
        self.train_image_directory = r"Data\train_images"
        self.train_mask_directory = r"Data\train_masks"
        self.val_image_directory = r"Data\validation_images"
        self.val_mask_directory = r"Data\validation_masks"
