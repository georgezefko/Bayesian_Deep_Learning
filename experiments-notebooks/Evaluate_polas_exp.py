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
from src.models.Vanilla_CNN import Vanilla_Net
#from src.models.stn_pola_exp import Net
from src.models.SpatialTN_2 import Net
from src.data import make_dataset
from src.utils import compute_dim
from src.utils.SaveLoad import save_ckp
from laplace.utils import ModuleNameSubnetMask
from src.utils import laplace_checks
from src.utils import visualization
import click
import wandb
import pickle
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
    "-ro",
    "--rotation",
    type=bool,
    default=False,
    help="Select false to train the model on rotated MNIST (default=False)",
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

    train_loader,_,test_loader = make_dataset.data(
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
    
    # Vanilla = Vanilla_Net(
    #     hype["channels"],
    #     hype["enc_sizes"],
    #     hype["pool"],
    #     hype["stride"],
    #     hype["kernel_size"],
    #     hype["padding"],
    #     hype["num_classes"])
        
    # model = Vanilla.to(device)

    model = STN.to(device).eval()
    logger.info(model)
    

    #trained_model_filepath = '/Users/georgioszefkilis/Bayesian_Deep_Learning/saved_models/colab_misMNIST_20_STN2__ver_2.pth'
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )

    wandb.init(
          project="Bayesian DL",
          name="Laplace_rotMNIST",
         entity="zefko",
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

    #########Expected average transformation##########
    set = torchvision.datasets.KMNIST(root='src/data', train=False,
                                        download=True, transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ))
    loader = torch.utils.data.DataLoader(set, batch_size=hype['batch_size'],
                                            shuffle=False, num_workers=2)
    true_transformation = []
    loc_transformation = []
    loc_out = []
    inputs = []
    for x,t in loader:
        x,t = x.to(device),t.to(device)
        #t = torch.flatten(t.type(torch.LongTensor)).to(device)
        # transformed,thetas = model.stn(x)
        # true_transformation.append(t)
        # loc_transformation.append(thetas)
        # loc_out.append(transformed)
        inputs.append(x)

    # true_transfo = torch.cat(true_transformation)
    # stn_transfo = torch.cat(loc_transformation)
    # stn_out= torch.cat(loc_out)
    test = torch.cat(inputs)
    
    uncertain_images = []

    breakpoint()



    #subetwork laplace
    #logger.info("Subnetwork Laplace evaluation")
    print('Subnetwork Laplace evaluation')
    #breakpoint()
    sub_laplace,_ = laplace_checks.laplace(model,train_loader,method='sub',module=['stn.fc_loc.0','stn.fc_loc.2'])
    sub_laplace.optimize_prior_precision(method='marglik')
    #optimal_prior_precision = laplace_checks.grid_cv(sub_laplace,test_loader)
    #sub_laplace.prior_precision = torch.tensor([15.0])#optimal_prior_precision
    print(sub_laplace.prior_precision)

    viz_samples = sub_laplace.sample(n_samples=100)
    
    visualization.visualise_stn(viz_samples,uncertain_images,test.cpu(),model,device,name='stn variances KMNIST',n_samples=8)
    # ####### Get samples#####
    # TODO calculate the variance
    with torch.no_grad():
        model.eval()
        transformation = []
        theta_samples = []
        mean = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
        subnet_samples = sub_laplace.sample(n_samples=10)
        #data,target = next(iter(dataloader))
        # for data, target in test_loader:
        #     data, target = data.to(device), target.to(device)
        #     #breakpoint()
            #print(data.shape)
            
        
        
        
        for sample in subnet_samples:
            torch.nn.utils.vector_to_parameters(sample, model.parameters())
            transformed_input_tensor,thetas = model.stn(test)#.to(device)
            transformation.append(transformed_input_tensor)
            theta_samples.append(thetas)
        torch.nn.utils.vector_to_parameters(mean,model.parameters())
            #transform = torch.cat(transformation)
            # for _ in range(len(transformation)):
            #     transformation_targets.append(target)
        breakpoint()
        transform = torch.stack(transformation, dim=1)

        theta_transform = torch.stack(theta_samples,dim=1)

        th_one, th_two, th_three, th_four, th_five, th_six = theta_variance(theta_transform)

        theta_var_outputs={'one':th_one,'two':th_two,'three':th_three,
                            'four':th_four,'five':th_five,'six':th_six}

        transfo_var = transform.cpu().detach().numpy()
        variances = [np.var(transfo_var[i]) for i in range(len(transform))]

    
        RESULTS_PATH = '/zhome/fc/5/104708/Desktop/Thesis/Polas_exp'
        pickle.dump(theta_var_outputs, open(RESULTS_PATH + '_KMNIST_variance_Laplace.p', 'wb'))
        breakpoint()
        # no_transfo = test.cpu().detach().numpy()
        # original_var = [np.var(no_transfo[i]) for i in range(len(test))]
        in_grid = convert_image_np(test[:1].squeeze(0).cpu())
        
        out_grid = convert_image_np(
               torchvision.utils.make_grid(transform[:1].squeeze(0).cpu()))
        #breakpoint()
        f, ax = plt.subplots(3, 1,figsize=(15,15))
        # ax[0].hist(original_var, bins = 20, facecolor = 'orange',label = 'Inputs',alpha=0.5)
        ax[0].hist(variances, bins = 20, facecolor='blue',label = 'Transformation',alpha=0.5)

        # ax[0].axvline(np.mean(original_var), color='orange', linestyle='dashed', linewidth=1)
        ax[0].axvline(np.mean(variances), color='blue', linestyle='dashed', linewidth=1)
        ax[0].legend(loc='upper right')

        ax[1].imshow(in_grid)
        ax[1].set_title('Input samples')
        ax[2].imshow(out_grid)
        ax[2].set_title('Transformation samples')
        plt.savefig('/zhome/fc/5/104708/Desktop/Thesis/src/visualization/variance3.png',bbox_inches='tight')
        f.tight_layout()

            # transform_target = torch.cat(transformation_targets)

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def theta_variance(theta):
    theta_reshape = theta.view(10000,10,6)
    theta_one = []
    theta_two = []
    theta_three = []
    theta_four = []
    theta_five = []
    theta_six = []
    

    for image in range(theta_reshape.shape[0]):
            
        theta_one.append(theta_reshape[image][:10][:,0])
        theta_two.append(theta_reshape[image][:10][:,1])
        theta_three.append(theta_reshape[image][:10][:,2])
        theta_four.append(theta_reshape[image][:10][:,3])
        theta_five.append(theta_reshape[image][:10][:,4])
        theta_six.append(theta_reshape[image][:10][:,5])
    

    theta_one_var = [np.var(theta_one[i].cpu().detach().numpy()) for i in range(len(theta_one))]
    theta_two_var = [np.var(theta_two[i].cpu().detach().numpy()) for i in range(len(theta_two))]
    theta_three_var = [np.var(theta_three[i].cpu().detach().numpy()) for i in range(len(theta_three))]
    theta_four_var = [np.var(theta_four[i].cpu().detach().numpy()) for i in range(len(theta_four))]
    theta_five_var = [np.var(theta_five[i].cpu().detach().numpy()) for i in range(len(theta_five))]
    theta_six_var = [np.var(theta_six[i].cpu().detach().numpy()) for i in range(len(theta_six))]

    return theta_one_var,theta_two_var,theta_three_var,theta_four_var,theta_five_var,theta_six_var

if __name__ == "__main__":
    main()