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
from src.data import make_dataset
from src.utils import compute_dim
from src.utils import laplace_checks

from laplace.utils import ModuleNameSubnetMask
import click
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





class Base(nn.Module):
    def __init__(self, enc_sizes, kernel, pad):
        super().__init__()

        
        conv_blocks =[compute_dim.conv_block(in_f, out_f, kernel_size=kernel, padding=pad) 
                       for in_f, out_f in zip(enc_sizes, enc_sizes[1:])]

        self.base_net = nn.Sequential(*conv_blocks)
        

    def forward(self,x):
        x = self.base_net(x)
    
        return x

class Vanilla(nn.Module):
    def __init__(self, in_c, enc_sizes, kernel, pad,n_classes):
        super().__init__()
    
        self.enc_sizes = [in_c, *enc_sizes]
        self.kernel = kernel
        self.pad = pad
        self.n_classes = n_classes
        
        self.base = Base(self.enc_sizes,self.kernel,self.pad)
        self.fc1 = nn.Linear(
            self.enc_sizes[-1] * 4* 4 , 50
        )
        self.fc2 = nn.Linear(50, self.n_classes)
        

    def forward(self,x):
        
        x = self.base(x)        
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x


@click.command()
@click.argument(
    "trained_model_filepath",
    type=click.Path(),
    default=r"/zhome/fc/5/104708/Desktop/Thesis/saved_models/colab_best_Vanilla_MNIST_20.pth",
)

def main(trained_model_filepath):
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

    train_loader,_, test_loader = make_dataset.data(
        hype["batch_size"], hype["crop_size"], misplacement=False
    )
    dataiter = iter(test_loader)
    images, _ = dataiter.next()
    print("image shape", images.shape)

    
    model = Vanilla(
        hype["channels"],
        hype["enc_sizes"],
        hype["kernel_size"],
        hype["padding"],
        hype["num_classes"]).to(device).eval()

    
    logger.info(model)

    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )

    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"])

    print('Weights and Biases',[ (name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad])
    summary(model,(1,28,28))
    logger.info("MAP evaluation")
    print('Map evaluation')

    acc_map,ece_map,nll_map = laplace_checks.predict(test_loader, model, laplace=False)

    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    
    # Laplace
    la,_ = laplace_checks.laplace(model,train_loader,method='last')


    
    #subetwork laplace
    sub_laplace,subnetwork_indices = laplace_checks.laplace(model,train_loader,method='sub',module=['fc2'])
    #
    laplace_samples_default, sublaplace_samples_default,laplace_sample_priors ,sublaplace_sample_priors,laplace_sample_no_opt,la_metrics,sub_metrics = laplace_checks.sanity_check_laplace(la,sub_laplace,test_loader,subnetwork_indices)
    
    
    laplace_data = {
    'MAP':[f'{acc_map:.1%}',f'{ece_map:.1%}',f'{nll_map:.3}'],
    'Laplace optimization':[f'{la_metrics[0][3]:.1%}',f'{la_metrics[1][3]:.1%}',f'{la_metrics[2][3]:.1%}'],
    'Laplace no optimization':[f'{la_metrics[0][0]:.1%}',f'{la_metrics[1][0]:.1%}',f'{la_metrics[2][0]:.1%}'],
    'Laplace small prior':[f'{la_metrics[0][1]:.1%}',f'{la_metrics[1][1]:.1%}',f'{la_metrics[2][1]:.1%}'],
    'Laplace large prior':[f'{la_metrics[0][2]:.1%}',f'{la_metrics[1][2]:.1%}',f'{la_metrics[2][2]:.1%}'],
    'Subnetwork Laplace':[f'{sub_metrics[0][0]:.1%}',f'{sub_metrics[1][0]:.1%}',f'{sub_metrics[2][0]:.1%}'],
    'Subnetwork Laplace small prior':[f'{sub_metrics[0][1]:.1%}',f'{sub_metrics[1][1]:.1%}',f'{sub_metrics[2][1]:.1%}'],
    'Subnetwork Laplace large prior':[f'{sub_metrics[0][2]:.1%}',f'{sub_metrics[1][2]:.1%}',f'{sub_metrics[2][2]:.1%}']
    }
    indices = ['Acc','ECE','NLL']
    laplace_report = pd.DataFrame(laplace_data,index=indices)
    laplace_report.to_csv('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/CNN_MNIST_report.xslx')
    
    laplace_checks.results(laplace_samples_default, sublaplace_samples_default,laplace_sample_priors ,sublaplace_sample_priors,laplace_sample_no_opt,la_metrics,sub_metrics,module='conv1')


if __name__ == "__main__":
    main()
