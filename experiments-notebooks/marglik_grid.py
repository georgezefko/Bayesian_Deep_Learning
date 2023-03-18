import warnings

warnings.simplefilter("ignore", UserWarning)

from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
import torch
import logging
import torchvision
from pathlib import Path
from laplace import Laplace, marglik_training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
from src.models.Hyperparameters import Hyperparameters as hp

from src.models.SpatialTN_2 import Net
#from src.models.stn_gdsrb import Net
from src.data import make_dataset
from src.utils import compute_dim
from src.utils.SaveLoad import save_ckp
from laplace.utils import ModuleNameSubnetMask
from src.utils import laplace_checks
from src.utils import visualization
from src.utils import reliability_diagram
import click
import wandb
from laplace.utils import LargestMagnitudeSubnetMask
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes



@click.command()
@click.argument(
    "trained_model_filepath",
    type=click.Path(),
    default='/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_MNIST_10k_cheat_HPC.pth'
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
    default=False,
    help="Select false to not overwrite existing data (default=True)",
)

@click.option(
    "-sub",
    "--subset",
    type=bool,
    default=False,
    help="Select True to load subset",
)

def main(trained_model_filepath,parameterize=False, misplacement=False,rotation=False,load=True,save=True,subset=False):
    """Evaluates the trained network using test subset of data"""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating a trained network using a test subset")

    # Check if there is a GPU available to use
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb.init(
          project="Bayesian DL",
          name="marglik",
         entity="zefko",
    )
    # Load the test data
    project_dir = Path(__file__).resolve().parents[2]

    # Load the hyperparameters
    hype = hp().config
    #remove from make dataset
    train_loader,val_loader,test_loader = make_dataset.data(
        hype["batch_size"], hype["crop_size"],hype["train_subset"],hype["dataset"], misplacement,rotation,load,save,subset
    )
    
    print("finished loading data")
    dataiter = iter(train_loader)
    images, _ = dataiter.next()
  
    print("image shape", images.shape)


    STN = Net(
        hype["channels"],
        hype["enc_sizes"],
        hype["loc_sizes"],
        hype['pool'],
        hype['stride'],
        hype["kernel_size"],
        hype["padding"],
        hype["num_classes"],
        parameterize)

    
    model = STN.to(device).eval()
    logger.info(model)


    state_dict = torch.load(
                project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
            )

    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"],strict=False)
    print('Weights and Biases:',[ (name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad])
    summary(model,(images.shape[1],images.shape[2],images.shape[3]))

    
    logger.info("MAP evaluation")
    print('Map evaluation')

    acc_map,ece_map,nll_map,_,_ = laplace_checks.predict(test_loader, model, laplace=False)

    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    #subetwork laplace
    logger.info("Subnetwork Laplace evaluation")
    print('Subnetwork Laplace evaluation')
    sub_laplace,_ = laplace_checks.laplace(model,train_loader,method='sub',module=['stn.fc_loc.0'])

    acc_sub,ece_sub,nll_sub,_,_ = laplace_checks.predict(test_loader,sub_laplace,laplace=True)

    print(
        f"[Sub Network Laplace] Acc.: {acc_sub:.1%}; ECE: {ece_sub:.1%}; NLL: {nll_sub:.3}"
    )
    
    logger.info("Grid CV search manual")
    print("Grid CV search manual")

    grid_prior, grid_likelihood,optimal_prior = grid_search(sub_laplace,test_loader)
    log_marginal_likelihood_op = sub_laplace.log_marginal_likelihood(prior_precision=optimal_prior)

    # logger.info("Grid CV search Lap")
    # print("Grid CV search Lap")

    # grid_prior_lap, grid_likelihood_lap  = cv_grid(sub_laplace,test_loader,val_loader)

    logger.info("Marglik")
    print("Marglik")
    marg_prior,marg_likelihood = marglik(sub_laplace,test_loader)

    breakpoint()
    fig,ax = plt.subplots(figsize =(10,10))
    
    ax.scatter(grid_prior, [i.cpu() for i in grid_likelihood], marker='o', color='red')

    ax.scatter(optimal_prior.cpu(), -log_marginal_likelihood_op.cpu(), marker='v', color='yellow', label='CV')
    # ax.scatter(grid_prior_lap.cpu(), grid_likelihood_lap.cpu(), marker='*', color='green', label='CV Lap')

    ax.scatter(marg_prior.cpu(), marg_likelihood.cpu(), marker='x', color='blue', label='Marglik')
    ax.set_xlabel('Prior Precision')
    ax.set_ylabel('Log Marginal Likelihood')
    ax.legend()
    

    plt.savefig('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/marglik_grid_10k.png',bbox_inches='tight')
    
def grid_search(sub_laplace,test_loader):
    
    

    optimal_prior_precision = laplace_checks.grid_cv(sub_laplace,test_loader)
    print('Optimal prior precision', optimal_prior_precision)

    sub_laplace.prior_precision = optimal_prior_precision

    
    prior_pre = torch.tensor([0.01,0.5,2,5,10,15,20,30,35])
    #prior_pre = torch.tensor([0.01,0.5,5,10,20,30,40,50])
    #prior_pre = torch.tensor([0.01, 0.1, 0.5, 1, 2, 5, 15, 100,200,500])
    # prior_pre = torch.logspace(
    #             0, 2, 10
    #          )
    
    marglik = []
    for i in prior_pre:
    
        log_marginal_likelihood = sub_laplace.log_marginal_likelihood(prior_precision= i)
        marglik.append(-log_marginal_likelihood)
        print(f'prior {i} Log marginal likelihood {-log_marginal_likelihood}')
        

    acc_sub_op_cv,ece_sub_op_cv,nll_sub_op_cv,_,_ = laplace_checks.predict(test_loader,sub_laplace,laplace=True)

    print(
        f"[Sub Network Laplace Grid CV] Acc.: {acc_sub_op_cv:.1%}; ECE: {ece_sub_op_cv:.1%}; NLL: {nll_sub_op_cv:.3}"
    )
    return prior_pre, marglik,optimal_prior_precision

def marglik(sub_laplace,test_loader):

    

    sub_laplace.optimize_prior_precision(method='marglik',verbose=True)
    print('optimal prior precision',sub_laplace.prior_precision)

    log_marginal_likelihood_op = sub_laplace.log_marginal_likelihood(prior_precision=sub_laplace.prior_precision)
    print('Log marginal likelihood', -log_marginal_likelihood_op)

    acc_sub_op,ece_sub_op,nll_sub_op,_,_ = laplace_checks.predict(test_loader,sub_laplace,laplace=True)

    print(
        f"[Sub Network Laplace] Acc.: {acc_sub_op:.1%}; ECE: {ece_sub_op:.1%}; NLL: {nll_sub_op:.3}"
    )


    return sub_laplace.prior_precision, -log_marginal_likelihood_op

# def cv_grid(sub_laplace,test_loader,val_loader):

    

#     sub_laplace.optimize_prior_precision(method='CV',val_loader = val_loader,log_prior_prec_min=0,log_prior_prec_max=2,grid_size=10,verbose=True)
#     print('optimal prior precision',sub_laplace.prior_precision)

#     log_marginal_likelihood_op = sub_laplace.log_marginal_likelihood(prior_precision=sub_laplace.prior_precision)
#     print('Log marginal likelihood', -log_marginal_likelihood_op)

#     acc_sub_op,ece_sub_op,nll_sub_op,_,_ = laplace_checks.predict(test_loader,sub_laplace,laplace=True)

#     print(
#         f"[Sub Network Laplace] Acc.: {acc_sub_op:.1%}; ECE: {ece_sub_op:.1%}; NLL: {nll_sub_op:.3}"
#     )

   
#     return sub_laplace.prior_precision, -log_marginal_likelihood_op
    
if __name__ == "__main__":
    main()
    
