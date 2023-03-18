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
    default='/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_Mapiliary_final_2.pth'
)
@click.option(
    "-m",
    "--misplacement",
    type=bool,
    default=False,
    help="Select true to train the model on misplacement MNIST (default=False)",
)

@click.option(
    "-ro",
    "--rotation",
    type=bool,
    default=False,
    help="Select false to train the model on rotated MNIST (default=False)",
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

    # Load the test data
    project_dir = Path(__file__).resolve().parents[2]

    # Load the hyperparameters
    hype = hp().config

    train_loader,valid_loader,test_loader = make_dataset.data(
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

    wandb.init(
          project="Bayesian DL",
          name="Laplace_Check",
         entity="zefko",
    )
     ################# Evaluate Subsets############
    # if hype['dataset']=='MNIST' and parameterize == False:

    #     logger.info("Evaluate Subset Performance")
    #     print("Evaluate Subset Performance")
        
    #     laplace_checks.subset_performance(model,device,hype["batch_size"],hype["crop_size"],
    #                                     hype["dataset"],misplacement,load,save,module=['stn.fc_loc.0','stn.fc_loc.2'],method='sub')
    
    ######### Load  path for the rest of the evaluation approaches#############
    
    #trained_model_filepath = '/Users/georgioszefkilis/Bayesian_Deep_Learning/saved_models/colab_misMNIST_20_STN2__ver_2.pth'
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )


    
    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"])
    print('Weights and Biases:',[ (name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad])
    summary(model,(images.shape[1],images.shape[2],images.shape[3]))

    wandb.log({"model":  [(name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad]})
    

    logger.info("MAP evaluation")
    print('Map evaluation')

    acc_map,ece_map,nll_map,images_map,labels = laplace_checks.predict(test_loader, model, laplace=False)

    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    
    # Laplace
    # logger.info("Laplace evaluation")
    # print('Laplace evaluation')
    # la,_ = laplace_checks.laplace(model,train_loader,method='last')
    
    #subetwork laplace
    logger.info("Subnetwork Laplace evaluation")
    print('Subnetwork Laplace evaluation')
    sub_laplace,subnetwork_indices = laplace_checks.laplace(model,train_loader,method='sub',module=['stn.fc_loc.2'])
    

    ####################################################################
    ###Evaluate weights distribution#################################################
    # logger.info("Evaluate Weights Distribution")
    # laplace_samples_default, sublaplace_samples_default,laplace_sample_priors,\
    # sublaplace_sample_priors,laplace_sample_no_opt,la_metrics,sub_metrics = laplace_checks.sanity_check_laplace(la,sub_laplace,test_loader,subnetwork_indices)
    
    
    # laplace_data = {
    # 'MAP':[f'{acc_map:.1%}',f'{ece_map:.1%}',f'{nll_map:.3}'],
    # 'Laplace optimization':[f'{la_metrics[0][3]:.1%}',f'{la_metrics[1][3]:.1%}',f'{la_metrics[2][3]:.3}'],
    # 'Laplace no optimization':[f'{la_metrics[0][0]:.1%}',f'{la_metrics[1][0]:.1%}',f'{la_metrics[2][0]:.3}'],
    # 'Laplace small prior':[f'{la_metrics[0][1]:.1%}',f'{la_metrics[1][1]:.1%}',f'{la_metrics[2][1]:.3}'],
    # 'Laplace large prior':[f'{la_metrics[0][2]:.1%}',f'{la_metrics[1][2]:.1%}',f'{la_metrics[2][2]:.3}'],
    # 'Subnetwork Laplace':[f'{sub_metrics[0][0]:.1%}',f'{sub_metrics[1][0]:.1%}',f'{sub_metrics[2][0]:.3}'],
    # 'Subnetwork Laplace small prior':[f'{sub_metrics[0][1]:.1%}',f'{sub_metrics[1][1]:.1%}',f'{sub_metrics[2][1]:.3}'],
    # 'Subnetwork Laplace large prior':[f'{sub_metrics[0][2]:.1%}',f'{sub_metrics[1][2]:.1%}',f'{sub_metrics[2][2]:.3}'],
    # }
    # #'Subnetwork Laplace optimization':[f'{sub_metrics[0][3]:.1%}',f'{sub_metrics[1][3]:.1%}',f'{sub_metrics[2][3]:.3}']
    # indices = ['Acc','ECE','NLL']
    # laplace_report = pd.DataFrame(laplace_data,index=indices)
    # laplace_report.to_csv('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/STN_lastlayer_MNIST_report.xslx')

    # laplace_table = wandb.Table(data=laplace_report)
    # wandb.log({'laplace_report':laplace_table})


    
    # laplace_checks.results(laplace_samples_default, sublaplace_samples_default,laplace_sample_priors ,sublaplace_sample_priors,laplace_sample_no_opt,la_metrics,sub_metrics,module='STN 2nd last layer')
    #######################################################################################

    ############# Subnetwork Samples#################################################
    logger.info("Evaluate Subnetwork Samples")
    print("Evaluate Subnetwork Samples")
    sub_laplace.optimize_prior_precision(method='marglik')
    #optimal_prior_precision = laplace_checks.grid_cv(sub_laplace,valid_loader)
    #sub_laplace.prior_precision = optimal_prior_precision
    print('Optimal prior precision',sub_laplace.prior_precision)

    
    
    # test_images = []
    # for x,_ in test_loader:
    #     x = x.to(device)
    #     test_images.append(x)

    # test = torch.cat(test_images)
    # test = test[:500] #gives cuda error for more images when it gets the predictive samples in line 181

    
    # # #check 2 evaluate images and sample distribution
    # acc_sub,ece_sub,nll_sub,images_sub_la,_ = laplace_checks.predict(test_loader, sub_laplace, laplace=True)

    # print(f"[Subnetwork Laplace] Acc.: {acc_sub:.1%}; ECE: {ece_sub:.1%}; NLL: {nll_sub:.3}")

    # uncertain_images = laplace_checks.uncertainties(test_loader,images_sub_la,images_map)
    
    
    # preds = sub_laplace.predictive_samples(test,pred_type='nn',n_samples=100,diagonal_output=True)

    # #breakpoint()
    # #visualize images with uncertainties
    # visualization.visualise_samples(uncertain_images,test_loader,test.cpu(),labels,preds,images_sub_la,images_map,n_samples=5,plotype='uncertainties')

    # # visualize images with misclassification
    # misclassified_images = []
    # for i in range(len(test_loader.dataset)):
    #     vla,pla = torch.max(images_sub_la[i],-1)
    #     v,p= torch.max(images_map[i],-1)
    #     if (p != labels[i]):
    #         misclassified_images.append(i)
    # visualization.visualise_samples(misclassified_images,test_loader,test.cpu(),labels,preds,images_sub_la,images_map,n_samples=5,plotype='misclassification')

    ##out of distribution and in distribution shift
    laplace_checks.ood_eval(model,sub_laplace,hype["batch_size"],hype["crop_size"],hype["train_subset"],
                        device,'CIFAR',misplacement,load,save,subset=False)

    subnet_samples = sub_laplace.sample(n_samples=100)
    
    visualization.visualise_stn(subnet_samples,uncertain_images,test.cpu(),model,device,n_samples=8)

    # ######## Evaluate calibration##########
    logger.info("Evaluate Calibration")
    print("Evaluate Calibration")
    y_conf = images_map.cpu().detach().numpy()
    labels_encoded = torch.nn.functional.one_hot(labels, num_classes=hype['num_classes'])

    
    # #acc_sub_op,ece_sub_op,nll_sub_op,sub_la_images_op,_ = laplace_checks.predict(test_loader,sub_laplace,laplace=True)

    
    
    
    y_conf_la_op = images_sub_la.cpu().detach().numpy()

   

    diagram = reliability_diagram.diagrams(y_conf,y_conf_la_op,labels_encoded,title='MAP Realiability Diagram')

    
    diagram.draw_reliability_graph(y_conf,labels_encoded,title='MAP Realiability Diagram')
    diagram.draw_reliability_graph(y_conf_la_op,labels_encoded,title='Laplace Realiability Diagram')
    diagram.draw_calibration_graph(y_conf,y_conf_la_op,labels_encoded,title='Calibration Curve')

   

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
