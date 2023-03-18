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
from src.models.SpatialTN_2 import Net
#from src.models.STN_seq import Net
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
          name="Uncertainty Experiment",
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

    

    state_dict = torch.load(
                project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
            )

    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"])
    print('Weights and Biases:',[ (name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad])
    #summary(model,(1,images.shape[2],images.shape[3]))


    test_images= []
    for x,_ in test_loader:
        x = x.to(device)
        test_images.append(x)
        
    test = torch.cat(test_images)
    #test = test[:2000]
    
    logger.info("MAP evaluation")
    print('Map evaluation')

    acc_map,ece_map,nll_map,images_map,labels = laplace_checks.predict(test_loader, model, laplace=False)

    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    #Diagonal laplace
    logger.info("Diagonal Laplace evaluation")
    print('Diagonal Laplace evaluation')
    diag_la,_ = laplace_checks.laplace(model,train_loader,method='diag')
    diag_la.optimize_prior_precision(method='marglik')
    acc_diag,ece_diag,nll_diag,diag_la_images,_ = laplace_checks.predict(test_loader,diag_la,laplace=True)
    print(
            f"[Diagonal Laplace] Acc.: {acc_diag:.1%}; ECE: {ece_diag:.1%}; NLL: {nll_diag:.3}"
        )
    diag_uncertain_images = laplace_checks.uncertainties(test_loader,diag_la_images,images_map)
    breakpoint()
    diag_preds= diag_la.predictive_samples(test,pred_type='nn',n_samples=100)#,diagonal_output=True)
    visualization.visualise_samples(diag_uncertain_images,test_loader,test.cpu(),labels,diag_preds,diag_la_images,images_map,plotype='Uncertainties Diagonal')

    diag_misclassified_images = []
    for i in range(len(test_loader.dataset)):
        vla,pla = torch.max(diag_la_images[i],-1)
        v,p= torch.max(images_map[i],-1)
        if (p != labels[i]):
            diag_misclassified_images.append(i)

    visualization.visualise_samples(diag_misclassified_images,test_loader,test.cpu(),labels,diag_preds,diag_la_images,images_map,plotype='misclasification uncertainties Diagonal')

    ### OOD Samples
    #plot OOD
   
    laplace_checks.ood_eval(model,diag_la,hype["batch_size"],hype["crop_size"],hype["train_subset"],
                           device,'KMNIST',plotype = 'OOD Diagonal Layer',misplacement=misplacement,load=load,save=save,subset=False)

    #last layer laplace
    breakpoint()
    last_la,_ = laplace_checks.laplace(model,train_loader,method='last')
    last_la.optimize_prior_precision(method='marglik')
    acc_last,ece_last,nll_last,last_la_images,_ = laplace_checks.predict(test_loader,last_la,laplace=True)
    print(
            f"[Last Layer Laplace] Acc.: {acc_last:.1%}; ECE: {ece_last:.1%}; NLL: {nll_last:.3}"
        )
    last_uncertain_images = laplace_checks.uncertainties(test_loader,last_la_images,images_map)
    last_preds= last_la.predictive_samples(test,pred_type='nn',n_samples=100)#,diagonal_output=True)
    visualization.visualise_samples(last_uncertain_images,test_loader,test.cpu(),labels,last_preds,last_la_images,images_map, plotype='Uncertainties Last Layer')


    last_misclassified_images = []
    for i in range(len(test_loader.dataset)):
        vla,pla = torch.max(last_la_images[i],-1)
        v,p= torch.max(images_map[i],-1)
        if (p != labels[i]):
            last_misclassified_images.append(i)
    visualization.visualise_samples(last_misclassified_images,test_loader,test.cpu(),labels,last_preds,last_la_images,images_map,plotype='Misclassification uncertianties last layer')

    ### OOD Samples
    #plot OOD
   
    laplace_checks.ood_eval(model,last_la,hype["batch_size"],hype["crop_size"],hype["train_subset"],
                           device,'KMNIST',plotype = 'OOD Last Layer',misplacement=misplacement,load=load,save=save,subset=False)
    breakpoint()
    #Subnetwork
    sub_laplace,_ = laplace_checks.laplace(model,train_loader,method='sub',module=['stn.fc_loc.0'])
    sub_laplace.optimize_prior_precision(method='marglik')
    print(sub_laplace.prior_precision)

    acc_sub,ece_sub,nll_sub,sub_la_images,_ = laplace_checks.predict(test_loader, sub_laplace, laplace=True)

    print(f"[Subnetwork Laplace] Acc.: {acc_sub:.1%}; ECE: {ece_sub:.1%}; NLL: {nll_sub:.3}")

    sub_uncertain_images = laplace_checks.uncertainties(test_loader,sub_la_images,images_map)
    sub_preds= sub_laplace.predictive_samples(test,pred_type='nn',n_samples=100)#,diagonal_output=True)
    visualization.visualise_samples(sub_uncertain_images,test_loader,test.cpu(),labels,sub_preds,sub_la_images,images_map, plotype='Uncertainties Subnetwork')


    sub_misclassified_images = []
    for i in range(len(test_loader.dataset)):
        vla,pla = torch.max(sub_la_images[i],-1)
        v,p= torch.max(images_map[i],-1)
        if (p != labels[i]):
            sub_misclassified_images.append(i)
    visualization.visualise_samples(sub_misclassified_images,test_loader,test.cpu(),labels,sub_preds,sub_la_images,images_map,plotype='Misclassification uncertianties Subnetwork')

    ### OOD Samples
    #plot OOD
   
    laplace_checks.ood_eval(model,sub_laplace,hype["batch_size"],hype["crop_size"],hype["train_subset"],
                           device,'KMNIST',plotype = 'OOD Subnetwork',misplacement=misplacement,load=load,save=save,subset=False)

    #metrics report

    laplace_data = {
    'MAP':[f'{acc_map:.1%}',f'{ece_map:.1%}',f'{nll_map:.3}'],
    'Diagonal':[f'{acc_diag:.1%}',f'{ece_diag:.1%}',f'{nll_diag:.3}'],
    'Last Layer':[f'{acc_last:.1%}',f'{ece_last:.1%}',f'{nll_last:.3}'],
    'Subnetwork':[f'{acc_sub:.1%}',f'{ece_sub:.1%}',f'{nll_sub:.3}'],
    
    }
    indices = ['Acc','ECE','NLL']
    laplace_report = pd.DataFrame(laplace_data,index=indices)
    laplace_report.to_csv('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/Uncertaintiesreport.xslx')

    laplace_table = wandb.Table(data=laplace_report)
    wandb.log({'laplace_report':laplace_table})



    
if __name__ == "__main__":
    main()
    
