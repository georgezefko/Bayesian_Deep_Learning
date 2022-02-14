import logging
import numpy as np
import torch
from torch import nn, optim
from laplace import Laplace
import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from src.models.SpatialTN import Net
from src.models.Hyperparameters import Hyperparameters as hp

from pathlib import Path
from src.data import make_dataset
from src.utils import SaveLoad
import torch.distributions as dists
import click
from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
from netcal.metrics import ECE


@click.command()
@click.argument(
    "model_path", type=click.Path(), default="models/colab_misMNIST_100.pth"
)
@click.option(
    "-m",
    "--misplacement",
    type=bool,
    default=False,
    help="Select false to train the model on misplacement MNIST (default=False)",
)
@click.option(
    "-pr",
    "--parameterize",
    type=bool,
    default=False,
    help="Select false to train only the scale theta parameters use only on misplacement MNIST (default=False)",
)
def laplace(model_path, misplacement=False, parameterize=False):

    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 0
    # define file paths
    project_dir = Path(__file__).resolve().parents[2]
    """Saves the relevant training images, the model, and the results"""
    # Set file paths depending on running locally
    load_model = project_dir.joinpath(model_path)

    # hypperparameters config
    hype = hp().config

    train_loader, test_loader = make_dataset.data(
        hype["batch_size"], hype["crop_size"], misplacement
    )

    targets = torch.cat([y for x, y in test_loader], dim=0)  # .numpy()

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
    model = STN.to(device).eval()
    optimizer = optim.Adam(model.parameters(), lr=hype["lr"])

    print("load model")
    # trained_model, _, _, _ = SaveLoad.load_ckp(load_model, model, optimizer, device)
    # train_loader.eval()
    # model.load_state_dict(torch.load(load_model, map_location=device))
    checkpoint = torch.load(load_model, map_location=device)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint["state_dict"])

    # start laplace
    # probs_map = predict(test_loader, model, laplace=False)
    # acc_map = (probs_map.argmax(-1) == targets).mean()
    # torch_prob = torch.from_numpy(probs_map)
    # torch_target = torch.from_numpy(targets)
    # nll_map = -dists.Categorical(torch_prob).log_prob(torch_target).mean()
    probs_map = predict(test_loader, model, laplace=False)
    print(probs_map)
    acc_map = (probs_map.argmax(-1) == targets).float().mean()
    ece_map = ECE(bins=15).measure(probs_map.numpy(), targets.numpy())
    nll_map = -dists.Categorical(probs_map).log_prob(targets).mean()

    logger.info(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")
    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    la = Laplace(
        model,
        "classification",
        subset_of_weights="last_layer",
        hessian_structure="kron",
    )
    la.fit(train_loader)
    la.optimize_prior_precision(method="marglik")
    # probs_laplace = predict(test_loader, la, laplace=True)
    # acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
    # nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    # logger.info(f"[Laplace] Acc.: {acc_laplace:.1%}; NLL: {nll_laplace:.3}")
    # print(f"[Laplace] Acc.: {acc_laplace:.1%}; NLL: {nll_laplace:.3}")
    probs_laplace = predict(test_loader, la, laplace=True)
    acc_laplace = (probs_laplace.argmax(-1) == targets).float().mean()
    ece_laplace = ECE(bins=15).measure(probs_laplace.numpy(), targets.numpy())
    nll_laplace = -dists.Categorical(probs_laplace).log_prob(targets).mean()

    logger.info(
        f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
    )
    print(
        f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
    )


@torch.no_grad()
def predict(dataloader, model, laplace=False):
    py = []

    for x, _ in dataloader:
        if laplace:
            py.append(model(x.cpu()))
        else:
            py.append(torch.softmax(model(x.cpu()), dim=-1))

    return torch.cat(py).cpu()


if __name__ == "__main__":
    laplace()
