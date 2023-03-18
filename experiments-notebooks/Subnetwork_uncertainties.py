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

#from src.models.SpatialTN_2 import Net
#from src.models.stn_pcam import Net
from src.models.stn_gdsrb import Net
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
    default='/zhome/fc/5/104708/Desktop/Thesis/saved_models/colab_best_MNIST_20_STN2__ver_2lr.pth'
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
    wandb.init(
          project="Bayesian DL",
          name="laplace_evaluation",
         entity="zefko",
    )
    # Load the test data
    project_dir = Path(__file__).resolve().parents[2]

    # Load the hyperparameters
    hype = hp().config
    #remove from make dataset
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

     
    # prior_pre = torch.tensor([0.1,0.5,1,10,20,30])
    laplace_checks.subset_performance(model,device,hype["batch_size"],hype["crop_size"],
                                    hype["dataset"],misplacement,load,save,module=['stn.fc_loc.0'],method='sub')

    state_dict = torch.load(
                project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
            )

    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"],strict=False)
    print('Weights and Biases:',[ (name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad])
    summary(model,(images.shape[1],images.shape[2],images.shape[3]))

    
    logger.info("MAP evaluation")
    print('Map evaluation')

    acc_map,ece_map,nll_map,images_map,labels = laplace_checks.predict(test_loader, model, laplace=False)

    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    #subetwork laplace
    logger.info("Subnetwork Laplace evaluation")
    print('Subnetwork Laplace evaluation')
    sub_laplace,_ = laplace_checks.laplace(model,train_loader,method='sub',module=['stn.fc_loc.4'])

    # acc_sub,ece_sub,nll_sub,sub_la_images,_ = laplace_checks.predict(test_loader,sub_laplace,laplace=True)

    # print(
    #     f"[Sub Network Laplace] Acc.: {acc_sub:.1%}; ECE: {ece_sub:.1%}; NLL: {nll_sub:.3}"
    # )
    
    logger.info("Reliability diagram before prior precision")
    print('Reliability diagram before prior precision"')
    y_true = labels.cpu().detach().numpy()
    _,y_pred= torch.max(images_map,-1)
    y_conf = images_map.cpu().detach().numpy()

    # _,y_pred_la = torch.max(sub_la_images,-1)
    # y_conf_la = sub_la_images.cpu().detach().numpy()

    labels_encoded = torch.nn.functional.one_hot(labels, num_classes=hype['num_classes'])


    #n_bins = 15

    #diagram = ReliabilityDiagram(n_bins,metric = 'ECE')
    # miscalibration_map = diagram.plot(y_conf, y_true,filename='/zhome/fc/5/104708/Desktop/Thesis/src/visualization/MAP Calibration.png')  # visualize miscalibration of uncalibrated
    # miscalibration_laplace = diagram.plot(y_conf_la, y_true,filename='/zhome/fc/5/104708/Desktop/Thesis/src/visualization/Laplace_calibration.png')   # visualize miscalibration of calibrated
    
    # eceM = ECE(bins=15).measure(y_conf, y_true)
    # eceL = ECE(bins=15).measure(y_conf_la, y_true)
    # print(eceM,eceL)
    # diagram = reliability_diagram.diagrams(y_conf,y_conf_la,labels_encoded,title='MAP Realiability Diagram')

    
    # diagram.draw_reliability_graph(y_conf,labels_encoded,title='MAP Realiability before Diagram')
    # diagram.draw_reliability_graph(y_conf_la,labels_encoded,title='Laplace Realiability no Diagram')
    # diagram.draw_calibration_graph(y_conf,y_conf_la,labels_encoded,title='Calibration no Curve')

    # logger.info("Grid CV search")
    # print("Grid CV search")
    

    #optimal_prior_precision = laplace_checks.grid_cv(sub_laplace,test_loader)
    #print('Optimal prior precision', optimal_prior_precision)
    sub_laplace.optimize_prior_precision(method='marglik')
    print('optimal prior precision',sub_laplace.prior_precision)

    #subetwork laplace
    #logger.info("Subnetwork Laplace evaluation optimal prior precision")
    #print('Subnetwork Laplace evaluation optmal prior precision')
    #sub_laplace.prior_precision = optimal_prior_precision

    acc_sub_op,ece_sub_op,nll_sub_op,sub_la_images_op,_ = laplace_checks.predict(test_loader,sub_laplace,laplace=True)

    print(
        f"[Sub Network Laplace] Acc.: {acc_sub_op:.1%}; ECE: {ece_sub_op:.1%}; NLL: {nll_sub_op:.3}"
    )
    
    # logger.info("Reliability diagram after prior precision optimisation")
    # print('Reliability diagram after prior precision"')
   

    # _,y_pred_la_op = torch.max(sub_la_images_op,-1)
    y_conf_la_op = sub_la_images_op.cpu().detach().numpy()

    # # labels_encoded = torch.nn.functional.one_hot(labels, num_classes=10)
    # miscalibration_laplaceOpt = diagram.plot(y_conf_la_op, y_true,filename='/zhome/fc/5/104708/Desktop/Thesis/src/visualization/LaplaceOpt_calibration.png')   # visualize miscalibration of calibrated
    # eceLO = ECE(bins=15).measure(y_conf_la_op, y_true)
    # print('optimal prior',eceLO)

    diagram = reliability_diagram.diagrams(y_conf,y_conf_la_op,labels_encoded,title='MAP Realiability Diagram')

    
    diagram.draw_reliability_graph(y_conf,labels_encoded,title='MAP Realiability Diagram')
    diagram.draw_reliability_graph(y_conf_la_op,labels_encoded,title='Laplace Realiability Diagram')
    diagram.draw_calibration_graph(y_conf,y_conf_la_op,labels_encoded,title='Calibration Curve')


    
    # prior_pre = torch.tensor([0.1,0.5,1,10,20,30])
    # laplace_checks.subset_performance(model,prior_pre,device,hype["batch_size"],hype["crop_size"],
    #                                 hype["dataset"],misplacement,load,save,module=['stn.fc_loc.0'],method='sub')
if __name__ == "__main__":
    main()
    
