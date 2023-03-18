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
from torch import nn, optim
import torch.nn.functional as F
import torch.distributions as dists
from netcal.metrics import ECE
from src.models.Hyperparameters import Hyperparameters as hp
from src.models.SpatialTN_2 import Net
#from src.models.STN_seq import Net
from src.data import make_dataset
from src.utils import compute_dim
from laplace.utils import ModuleNameSubnetMask
from src.utils import laplace_checks
from src.utils import visualization
from src.utils import SaveLoad
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
    default='/zhome/fc/5/104708/Desktop/Thesis/saved_models/STN_MNIST_100_HPC.pth'
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
    help="Select True to use subset (default=False)",
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
    
    train_loader,valid_loader,_ = make_dataset.data(
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

    optimizer = optim.Adam([
                {'params': model.base.parameters()},
                {'params':model.fc1.parameters()},
                {'params':model.fc2.parameters()},
                {'params':model.stn.fc_loc.parameters(),'lr': hype["learning_rate_stn"]},
                {'params': model.stn.localization.parameters(), 'lr': hype["learning_rate_stn"]}
            ], lr=hype["learning_rate_base"])

    criterion = nn.CrossEntropyLoss()
    #trained_model_filepath = '/Users/georgioszefkilis/Bayesian_Deep_Learning/saved_models/colab_misMNIST_20_STN2__ver_2.pth'
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )

    wandb.init(
          project="Bayesian DL",
          name="Continual Training",
         entity="zefko",
    )
    # initialize state_dict from checkpoint to model
    model,optimizer,start_epoch,valid_loss_min = SaveLoad.load_ckp(trained_model_filepath,model,optimizer)    
    
    print('Weights and Biases:',[ (name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad])
    summary(model,(1,images.shape[2],images.shape[3]))

    wandb.log({"model":  [(name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad]})



    #fit subnetowrk laplace
    sub_laplace,_ = laplace_checks.laplace(model,train_loader,method='sub',module=['stn.fc_loc.0'])
    sub_laplace.optimize_prior_precision(method='marglik')
    breakpoint()
    #sub_laplace.prior_precision = 30#torch.tensor([5])
    #### Second last layer STN
    #30-> 57% == MAP
    #20->58%
    #15->57% == MAP
    #10->59%
    #5->57% == MAP
    #1(defualt) -> 56%-57%==MAP
    #0.1->39% <MAP
    #sub_laplace.optimize_prior_precision(method='marglik')

    #last layer STN

    #30-> 33% == MAP
    #20->33%
    #15->30% == MAP
    #10->26%
    #5->20% 
    #1(defualt) -> 16%
    #0.5 -> 18%
    #0.1->13% <MAP

    # freeze weights in localizer
    model.stn.fc_loc[0].weight.requires_grad =False
    model.stn.fc_loc[2].weight.requires_grad =False
    model.stn.localization[0][0].weight.requires_grad = False
    model.stn.localization[0][2].weight.requires_grad = False
    model.stn.localization[1][0].weight.requires_grad = False
    model.stn.localization[1][2].weight.requires_grad = False

    
    
    
    mean = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    model.train()
    for epoch in range(start_epoch, 140):
        wandb.watch(model, optimizer, log="all", log_freq=10)
        train_loss = 0
        
        #new_data = []
        #new_target_set = []
        for batch_idx, (data, target) in enumerate(train_loader):
            #print(batch_idx)
            data, target = data.to(device), target.to(device)
            #breakpoint()
            subnet_samples = sub_laplace.sample(n_samples=100)
            transformation = []
            transformation_targets = []
            for sample in subnet_samples:
                torch.nn.utils.vector_to_parameters(sample, model.parameters())
                transformed_input_tensor = model.stn(data.to(device))
                transformation.append(transformed_input_tensor)
            torch.nn.utils.vector_to_parameters(mean,model.parameters())
            transform = torch.cat(transformation)
            for _ in range(len(transformation)):
                transformation_targets.append(target)

            transform_target = torch.cat(transformation_targets)

            #breakpoint()
            data = transform#torch.cat((data,transform))
            target = transform_target#torch.cat((target,transform_target))
        
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 500 == 0:
                wandb.log({"epoch": epoch, "loss": loss.item()})
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        with torch.no_grad():
                model.eval()
                test_loss = 0
                correct = 0

                for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    # sum up batch loss
                    test_loss += criterion(output, target).item()
                    # get the index of the max log-probability
                    pred = output.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(valid_loader.dataset)

                wandb.log(
                        {
                            "Validation loss": test_loss,
                            "Validation_accuracy": 100.0 * correct / len(valid_loader.dataset),
                        }
                )
                
                print(
                    "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                        test_loss,
                        correct,
                        len(valid_loader.dataset),
                        100.0 * correct / len(valid_loader.dataset),
                    )
                )

                if test_loss <= valid_loss_min:
                    print(
                        "Validation loss decreased ({:.6f} --> {:.6f})".format(
                            valid_loss_min, test_loss
                        )
                    )

if __name__=='__main__':
    main()