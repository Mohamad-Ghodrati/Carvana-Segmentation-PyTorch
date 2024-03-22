# Carvana U-Net Segmentation with Training Framework (PyTorch)

## Introduction

This repository implements a U-Net architecture for image segmentation tasks, 
accompanied by a comprehensive training framework using PyTorch. 
The framework provides functionalities for:

- Mixed precision training
- Checkpointing
- Evaluation and visualization
- ...

## Dataset Structure

Your dataset is expected to have the following directory structure:  
>__*Note that you can change the extentions and folder names in the code.*__
```
    carvana_data/
    ├── train_images/
    │   ├── image_name.jpg
    │   └── ... (other image files)
    └── train_masks/
        ├── image_name.png  (corresponding mask for image_name.jpg)
        └── ... (other mask files)
```

## Usage

1. Clone this repository.
2. Create your dataset directories (images and masks) and organize them appropriately (train/validation).
3. Adjust hyperparameters in `hyperparameters.py` if needed.
4. Run the training script:

```bash
python train.py
```

## Training Results


Upon successful training completion, the code will save 10 prediction masks by default (you can modify this value in the code). These masks will be saved in a designated output directory (defaults to `'saved_images\'`).
| Input image | Ground Truth | Preiction |
|---|---|---|
| ![Image 1 description](saved_images\0.jpg) | ![Image 2 description](saved_images\0_mask.jpg) | ![Image 3 description](saved_images\0_pred.jpg) |

| Input image | Ground Truth | Preiction |
|---|---|---|
| ![Image 1 description](saved_images\1.jpg) | ![Image 2 description](saved_images\1_mask.jpg) | ![Image 3 description](saved_images\1_pred.jpg) |


## Contributing

I welcome contributions to improve this project! Feel free to create pull requests with bug fixes, enhancements, or new features.

## Disclaimer

This code is provided for educational purposes and may require adjustments for specific use cases.
