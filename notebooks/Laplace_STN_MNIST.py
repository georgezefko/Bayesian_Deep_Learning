import warnings

warnings.simplefilter("ignore", UserWarning)

from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
import torch
import logging
from pathlib import Path
from laplace import Laplace, marglik_training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from netcal.metrics import ECE
from src.models.Hyperparameters import Hyperparameters as hp
from src.models.SpatialTN_2 import Net

from src.data import make_dataset
from src.utils import compute_dim
from laplace.utils import ModuleNameSubnetMask
import click





def laplace(model,dataloader,method='last',module='fc1'):
    if method == 'last':
        la = Laplace(
            model,
            "classification",
            subset_of_weights="last_layer",
            hessian_structure="kron",
        )
        la.fit(dataloader)
        la.optimize_prior_precision(method="marglik")
    else:
        subnetwork_mask = ModuleNameSubnetMask(model, module_names=[module])
        subnetwork_mask.select()
        subnetwork_indices = subnetwork_mask.indices.type(torch.LongTensor)
        la = Laplace(
            model,
            "classification",
            subset_of_weights="subnetwork",
            hessian_structure="full",
            subnetwork_indices = subnetwork_indices.type(torch.LongTensor),
        )
        la.fit(dataloader)
        la.prior_precision = torch.tensor([10000])
        
    return la

@click.command()
@click.argument(
    "trained_model_filepath",
    type=click.Path(),
    default="saved_models/STN_MNIST_10_HPC.pth",
)
@click.option(
    "-m",
    "--misplacement",
    type=bool,
    default=False,
    help="Select true to train the model on misplacement MNIST (default=False)",
)
@click.option(
    "-pr",
    "--parameterize",
    type=bool,
    default=False,
    help="Select true to train only the scale theta parameters use only on misplacement MNIST (default=False)",
)

@click.option(
    "-lo",
    "--load",
    type=bool,
    default=True,
    help="Select false to create data (default=True)",
)
@click.option(
    "-sa",
    "--save",
    type=bool,
    default=True,
    help="Select false to not overwrite existing data (default=True)",
)


def main(trained_model_filepath,parameterize=False, misplacement=False,load=True,save=True):
    """Evaluates the trained network using test subset of data"""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating a trained network using a test subset")

    # Check if there is a GPU available to use
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the test data
    project_dir = Path(__file__).resolve().parents[2]

    # Load the hyperparameters
    hype = hp().config

    train_loader,_,test_loader = make_dataset.data(
        hype["batch_size"], hype["crop_size"], misplacement,load,save
    )
    
    STN = Net(
        hype["channels"],
        hype["enc_sizes"],
        hype["loc_sizes"],
        hype["kernel_size"],
        hype["padding"],
        hype["num_classes"],
        parameterize)

    
    model = STN.to(device).eval()
    logger.info(model)

    

    #trained_model_filepath = '/Users/georgioszefkilis/Bayesian_Deep_Learning/saved_models/colab_misMNIST_20_STN2__ver_2.pth'
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )

    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"])

    @torch.no_grad()
    def predict(dataloader, model, laplace=False):
        py = []
        target = []
        for x, t in dataloader:
            x,t = x.to(device),t.to(device)
            target.append(t)
            if laplace:
                py.append(model(x))
            else:
                py.append(torch.softmax(model(x), dim=-1))

        images = torch.cat(py).cpu()
        labels =torch.cat(target, dim=0).cpu()
        acc_map = (images.argmax(-1) == labels).float().mean()
        ece_map = ECE(bins=15).measure(images.numpy(), labels.numpy())
        nll_map = -dists.Categorical(images).log_prob(labels).mean()
        
        return acc_map,ece_map,nll_map

    logger.info("MAP evaluation")
    print('Map evaluation')

    acc_map,ece_map,nll_map = predict(test_loader, model, laplace=False)

    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    
    # Laplace
    logger.info("Laplace evaluation")
    print('Laplace evaluation')
    la = laplace(model,train_loader,method='last')
    acc_laplace,ece_laplace,nll_laplace = predict(test_loader, la, laplace=True)

    print(
        f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
    )
    for name, module in model.named_modules():
        print(name)
    #subetwork laplace
    logger.info("Subnetwork Laplace evaluation")
    print('Subnetwork Laplce evaluation')
    sub_laplace = laplace(model,train_loader,method='sub',module='base.base_net.1.0')
    acc_sublaplace,ece_sublaplace,nll_sublaplace = predict(test_loader, sub_laplace, laplace=True)

    print(
        f"[Laplace] Acc.: {acc_sublaplace:.1%}; ECE: {ece_sublaplace:.1%}; NLL: {nll_sublaplace:.3}"
    )



if __name__ == "__main__":
    main()
