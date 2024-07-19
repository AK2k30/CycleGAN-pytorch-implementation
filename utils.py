"""
Utility functions for saving and loading model checkpoints, and setting random seeds.

save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    Saves the current state of the model and optimizer to a checkpoint file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        filename (str, optional): The filename to save the checkpoint to. Defaults to "my_checkpoint.pth.tar".

load_checkpoint(checkpoint_file, model, optimizer, lr):
    Loads a model and optimizer from a checkpoint file, and updates the learning rate.

    Args:
        checkpoint_file (str): The path to the checkpoint file.
        model (torch.nn.Module): The model to load the state dict into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state dict into.
        lr (float): The new learning rate to set for the optimizer.

seed_everything(seed=42):
    Sets random seeds for reproducibility.

    Args:
        seed (int, optional): The seed to use. Defaults to 42.
"""
import random, torch, os, numpy as np
import torch.nn as nn
import config
import copy

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False