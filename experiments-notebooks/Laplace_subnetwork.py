import warnings

warnings.simplefilter("ignore", UserWarning)

from laplace.baselaplace import FullLaplace
from laplace.curvature.backpack import BackPackGGN
import torch
import logging
import torchvision
from torchvision import datasets, transforms
from pathlib import Path
from laplace import Laplace, marglik_training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from netcal.metrics import ECE
from src.models.Hyperparameters import Hyperparameters as hp
#from src.models.SpatialTN_2 import Net
#from src.models.Vanilla_CNN import Vanilla_Net
from src.models.stn_gdsrb import Net
from src.data import make_dataset
from src.utils import compute_dim
from src.utils.SaveLoad import save_ckp
from laplace.utils import ModuleNameSubnetMask
from src.utils import laplace_checks
from src.utils import visualization
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
#plt.rcParams["axes.grid"] = False


@click.command()
@click.argument(
    "trained_model_filepath",
    type=click.Path(),
    default='/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_Mapiliary.pth'
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
    help="Select True to use subset of data (default=False)",
)



def main(trained_model_filepath,parameterize=False, misplacement=False,load=True,save=True,subset=False):
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
        hype["batch_size"], hype["crop_size"],hype["train_subset"],hype["dataset"], misplacement,load,save,subset
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
    

    #trained_model_filepath = '/Users/georgioszefkilis/Bayesian_Deep_Learning/saved_models/colab_misMNIST_20_STN2__ver_2.pth'
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )

    wandb.init(
          project="Bayesian DL",
          name="Laplace_subnetwork",
         entity="zefko",
    )
    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"])
    print('Weights and Biases:',[ (name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad])
    #summary(model,(images.shape[1],images.shape[2],images.shape[3]))

    wandb.log({"model":  [(name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad]})

    logger.info("MAP evaluation")
    print('Map evaluation')

    acc_map,ece_map,nll_map,images_map,labels = laplace_checks.predict(test_loader, model, laplace=False)

    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    
    #subetwork laplace
    logger.info("Subnetwork Laplace evaluation")
    print('Subnetwork Laplace evaluation')
    #breakpoint()
    sub_laplace,_ = laplace_checks.laplace(model,train_loader,method='sub',module=['stn.fc_loc.2'])
    sub_laplace.optimize_prior_precision(method='marglik')
    #optimal_prior_precision = laplace_checks.grid_cv(sub_laplace,test_loader)
    #sub_laplace.prior_precision = torch.tensor([15.0])#optimal_prior_precision
    print(sub_laplace.prior_precision)

    
    
    test_images = []
    for x,_ in test_loader:
        x = x.to(device)
        test_images.append(x)

    test = torch.cat(test_images)
    test = test[100:400] #gives cuda error for more images when it gets the predictive samples in line 181

    
    #breakpoint()
    #check 2 evaluate images and sample distribution
    acc_sub,ece_sub,nll_sub,images_sub_la,_ = laplace_checks.predict(test_loader, sub_laplace, laplace=True)

    print(f"[Subnetwork Laplace] Acc.: {acc_sub:.1%}; ECE: {ece_sub:.1%}; NLL: {nll_sub:.3}")

    uncertain_images = laplace_checks.uncertainties(test_loader,images_sub_la,images_map)
    
    breakpoint()
    preds = sub_laplace.predictive_samples(test,pred_type='glm',n_samples=500,diagonal_output=True)

    #test = torch.cat(test_images).cpu()
    #breakpoint()
    # visualize images with uncertainties
    visualization.visualise_samples(uncertain_images,test_loader,test.cpu(),labels,preds,images_sub_la,images_map,n_samples=5,plotype='uncertainties')

    #breakpoint()
    # visualize misclassified images
    # misclassified_images = []
    # for i in range(len(test_loader.dataset)):
    #     vla,pla = torch.max(images_sub_la[i],-1)
    #     v,p= torch.max(images_map[i],-1)
    #     if (p != labels[i]):
    #         misclassified_images.append(i)
    # visualization.visualise_samples(misclassified_images,test_loader,test.cpu(),labels,preds,images_sub_la,images_map,n_samples=5,plotype='misclassification')

    # # # #plot OOD
    # # breakpoint()
    # # laplace_checks.ood_eval(model,sub_laplace,hype["batch_size"],hype["crop_size"],hype["train_subset"],
    # #                        device,'KMNIST','OOD',misplacement,load,save,subset=False)
    
    # #plot samples from subnetwork laplace
    # breakpoint()
    subnet_samples = sub_laplace.sample(n_samples=100)
    
    visualization.visualise_stn(subnet_samples,uncertain_images,test.cpu(),model,device,n_samples=8)
    breakpoint()

if __name__ == "__main__":
    main()