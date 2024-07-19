"""
This module defines various configuration parameters for a machine learning project.

The following configuration parameters are defined:
- DEVICE: The device to use for training, either "cuda" if a GPU is available or "cpu".
- TRAIN_DIR: The directory containing the training data.
- VAL_DIR: The directory containing the validation data.
- BATCH_SIZE: The batch size for training and validation.
- LEARNING_RATE: The learning rate for the optimization algorithm.
- LAMBDA_IDENTITY: The weight of the identity loss term in the overall loss function.
- LAMBDA_CYCLE: The weight of the cycle consistency loss term in the overall loss function.
- NUM_WORKERS: The number of worker processes to use for data loading.
- NUM_EPOCHS: The number of training epochs.
- LOAD_MODEL: A flag indicating whether to load a pre-trained model.
- SAVE_MODEL: A flag indicating whether to save the trained model.
- CHECKPOINT_GEN_H: The filename for the generator H checkpoint.
- CHECKPOINT_GEN_Z: The filename for the generator Z checkpoint.
- CHECKPOINT_CRITIC_H: The filename for the critic H checkpoint.
- CHECKPOINT_CRITIC_Z: The filename for the critic Z checkpoint.

The `transforms` variable defines a set of data augmentation transformations to be applied to the input images, including resizing, horizontal flipping, and normalization.
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)