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
from src.models.Hyperparameters import Hyperparameters as hp
#from src.models.SpatialTN_2 import Net
#from src.models.STN_seq import Net
from src.models.stn_gdsrb import Net
from src.models.Vanilla_CNN import Vanilla_Net
from src.data import make_dataset
from src.utils import compute_dim
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



@click.command()
@click.argument(
    "trained_model_filepath",
    type=click.Path(),
    default='/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_Mapiliary_deep.pth'
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
        hype["pool"],
        hype["stride"],
        hype["kernel_size"],
        hype["padding"],
        hype["num_classes"],
        parameterize)
    
    # Vanilla = Vanilla_Net(
    #     hype["channels"],
    #     hype["enc_sizes"],
    #     hype["pool"],
    #     hype["stride"],
    #     hype["kernel_size"],
    #     hype["padding"],
    #     hype["num_classes"])
        
    # model = Vanilla.to(device).eval()

    
    model = STN.to(device).eval()
    logger.info(model)

    

    #trained_model_filepath = '/Users/georgioszefkilis/Bayesian_Deep_Learning/saved_models/colab_misMNIST_20_STN2__ver_2.pth'
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )


    wandb.init(
          project="Bayesian DL",
          name="Laplace_Check",
         entity="zefko",
    )
    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"])
    print('Weights and Biases:',[ (name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad])
    summary(model,(images.shape[1],images.shape[2],images.shape[3]))

    wandb.log({"model":  [(name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad]})
    

    #weights0 = []
    #weights2 = []
    #weightslast = []
    #for name,p in model.named_parameters():
    #    if name=='stn.fc_loc.0.weight':
    #        weights0.append(p.cpu().detach().numpy())
    #    elif name=='stn.fc_loc.2.weight':
    #        weights2.append(p.cpu().detach().numpy())
    #    elif name=='fc2.weight':
    #        weightslast.append(p.cpu().detach().numpy())

    #fig,axes = plt.subplots(1,3,figsize=(15,10))
    #axes[0].hist(weights0[0],bins=15,facecolor='blue',alpha=0.5)
    #axes[1].hist(weights2[0],bins=15,facecolor='blue',alpha=0.5)
    #axes[2].hist(weightslast[0],bins=15,facecolor='blue',alpha=0.5)
    #axes[0].set_title('STN second last layer',size=15)
    #axes[1].set_title('STN last layer',size=15)
    #axes[2].set_title('Classifier last layer',size=15)
    #plt.savefig('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/STN_lastlayer_MNIST_weights.png',bbox_inches='tight')


    logger.info("MAP evaluation")
    print('Map evaluation')

    acc_map,ece_map,nll_map,_,_ = laplace_checks.predict(test_loader, model, laplace=False)

    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    
    # Laplace
    logger.info("Laplace evaluation")
    print('Laplace evaluation')
    la,_ = laplace_checks.laplace(model,train_loader,method='last')
    
    #subetwork laplace
    logger.info("Subnetwork Laplace evaluation")
    print('Subnetwork Laplace evaluation')
    sub_laplace,subnetwork_indices = laplace_checks.laplace(model,train_loader,method='sub',module=['stn.fc_loc.2'])
    
    laplace_samples_default, sublaplace_samples_default,laplace_sample_priors,\
    sublaplace_sample_priors,laplace_sample_no_opt,la_metrics,sub_metrics = laplace_checks.sanity_check_laplace(la,sub_laplace,test_loader,subnetwork_indices)
    
    
    
    laplace_data = {
    'MAP':[f'{acc_map:.1%}',f'{ece_map:.1%}',f'{nll_map:.3}'],
    'Laplace optimization':[f'{la_metrics[0][3]:.1%}',f'{la_metrics[1][3]:.1%}',f'{la_metrics[2][3]:.3}'],
    'Laplace no optimization':[f'{la_metrics[0][0]:.1%}',f'{la_metrics[1][0]:.1%}',f'{la_metrics[2][0]:.3}'],
    'Laplace small prior':[f'{la_metrics[0][1]:.1%}',f'{la_metrics[1][1]:.1%}',f'{la_metrics[2][1]:.3}'],
    'Laplace large prior':[f'{la_metrics[0][2]:.1%}',f'{la_metrics[1][2]:.1%}',f'{la_metrics[2][2]:.3}'],
    'Subnetwork Laplace':[f'{sub_metrics[0][0]:.1%}',f'{sub_metrics[1][0]:.1%}',f'{sub_metrics[2][0]:.3}'],
    'Subnetwork Laplace small prior':[f'{sub_metrics[0][1]:.1%}',f'{sub_metrics[1][1]:.1%}',f'{sub_metrics[2][1]:.3}'],
    'Subnetwork Laplace large prior':[f'{sub_metrics[0][2]:.1%}',f'{sub_metrics[1][2]:.1%}',f'{sub_metrics[2][2]:.3}'],
    }
    #'Subnetwork Laplace optimization':[f'{sub_metrics[0][3]:.1%}',f'{sub_metrics[1][3]:.1%}',f'{sub_metrics[2][3]:.3}']
    indices = ['Acc','ECE','NLL']
    laplace_report = pd.DataFrame(laplace_data,index=indices)
    laplace_report.to_csv('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/STN_lastlayer_MNIST_report.xslx')

    laplace_table = wandb.Table(data=laplace_report)
    wandb.log({'laplace_report':laplace_table})


    
    laplace_checks.results(laplace_samples_default, sublaplace_samples_default,laplace_sample_priors ,sublaplace_sample_priors,laplace_sample_no_opt,la_metrics,sub_metrics,module='STN 2nd last layer')

    #check 2 evaluate images and sample distribution


if __name__ == "__main__":
    main()
