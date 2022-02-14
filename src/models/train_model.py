import logging
import os
import pickle
from pathlib import Path
import wandb

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


from src.models.SpatialTN import Net
from src.models.Hyperparameters import Hyperparameters as hp
from src.data import make_dataset
from src.utils import SaveLoad

from src.utils import visualization


def train(
    check_path,
    model_path,
    valid_loss_min_input=0.05,
    misplacement=False,
    parameterize=False,
):
    # Check if there is a GPU available to use
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 0
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # define file paths
    project_dir = Path(__file__).resolve().parents[2]
    """Saves the relevant training images, the model, and the results"""
    # Set file paths depending on running locally or on Azure
    best_model_path = project_dir.joinpath(model_path)
    checkpoint_path = project_dir.joinpath(check_path)

    # hypperparameters config
    hype = hp().config

    train_loader, test_loader = make_dataset.data(
        hype["batch_size"], hype["crop_size"], misplacement
    )

    dataiter = iter(train_loader)
    images, _ = dataiter.next()
    height = images.shape[2]
    width = images.shape[3]
    print("image shape", images.shape)

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info("Training STN")

    # Initialize the model and transfer to GPU if available
    STN = Net(
        hype["num_classes"],
        hype["channels"],
        hype["filter1_out"],
        hype["filter2_out"],
        hype["kernel_size"],
        hype["padding"],
        hype["stride"],
        height,
        width,
        hype["pool"],
        parameterize,
    )
    model = STN.to(device)

    optimizer = optim.Adam(model.parameters(), lr=hype["lr"])

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input
    # initialize wandb
    wandb.init(
        project="Bayesian DL",
        name="STN_DEEP_MNIST_ver_3",
        entity="zefko",
    )

    for epoch in range(1, hype["epochs"] + 1):

        wandb.watch(model, optimizer, log="all", log_freq=10)
        train_loss = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 500 == 0:
                wandb.log({"epoch": epoch, "loss": loss.item()})
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0

            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                # sum up batch loss
                test_loss += F.nll_loss(output, target, size_average=False).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            wandb.log(
                {
                    "Validation loss": test_loss,
                    "Validation_accuracy": 100.0 * correct / len(test_loader.dataset),
                }
            )

            print(
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(test_loader.dataset),
                    100.0 * correct / len(test_loader.dataset),
                )
            )

        # create checkpoint variable and add important data
        checkpoint = {
            "epoch": epoch + 1,
            "valid_loss_min": test_loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # save checkpoint
        SaveLoad.save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        # log the stn outputs
        visualization.wandb_pred(model, test_loader, device)

        ## TODO: save the model if validation loss has decreased
        if test_loss <= valid_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, test_loss
                )
            )
            # save checkpoint as best model
            SaveLoad.save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = test_loss
