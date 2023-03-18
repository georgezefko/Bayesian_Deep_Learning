import logging
from pathlib import Path
import wandb
import numpy as np
import torch

from torch import nn, optim

from torchsummary import summary
from src.models.Vanilla_CNN import Vanilla_Net
from src.models.SpatialTN import Net
#from src.models.stn_pcam import Net

from src.models.Hyperparameters import Hyperparameters as hp
from src.data import make_dataset
from src.utils import SaveLoad,visualization



def train(
    check_path,
    model_path,
    valid_loss_min_input=0.05,
    misplacement=False,
    rotation=False,
    parameterize=False,
    load=True,
    save=True,
    subset=False, 
):
    # Check if there is a GPU available to use
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 0
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # define file paths
    project_dir = Path(__file__).resolve().parents[2]
    """Saves the relevant training images, the model, and the results"""
    # Set file paths depending on running locally or on Azure
    best_model_path = project_dir.joinpath(model_path)
    checkpoint_path = project_dir.joinpath(check_path)
    # cam_path = str(project_dir) + "/src/data/PCAM/"

    # hypperparameters config
    hype = hp().config
    
    # split set
    train_loader,val_loader,_ = make_dataset.data(hype["batch_size"], hype["crop_size"], hype["train_subset"], hype["dataset"],
                                misplacement,rotation,load,save,subset)

    
    print("finished loading data")
    dataiter = iter(train_loader)
    images, _ = dataiter.next()
    
    print("image shape", images.shape)

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info("Training STN")
    # initialize wandb for the different datasets
    if hype['dataset']=='PCAM':

        if hype['stn']:
            wandb.init(
                project="Bayesian DL",
                name="STN_PCAM",
                entity="zefko",)
        else:
            wandb.init(
                project="Bayesian DL",
                name="Vanilla_PCAM",
                entity="zefko",)

    elif hype['dataset']=='GTSRB':

        if hype['stn']:
            
            wandb.init(
                project="Bayesian DL",
                name="STN_GTSRB",
                entity="zefko",)
        else:
            wandb.init(
                project="Bayesian DL",
                name="Vanilla_GTSRB",
                entity="zefko",)
    
    elif hype['dataset']=='GTSDB':

        if hype['stn']:
            
            wandb.init(
                project="Bayesian DL",
                name="STN_GTSDB",
                entity="zefko",)
        else:
            wandb.init(
                project="Bayesian DL",
                name="Vanilla_GTSDB",
                entity="zefko",)

    elif hype['dataset']=='Mapiliary':

        if hype['stn']:
            
            wandb.init(
                project="Bayesian DL",
                name="STN_Mapiliary",
                entity="zefko",)
        else:
            wandb.init(
                project="Bayesian DL",
                name="Vanilla_Mapiliary",
                entity="zefko",)
    else:
        if hype['stn']:
            if misplacement:
                wandb.init(
                    project="Bayesian DL",
                    name="STN_misMNIST",
                    entity="zefko",)
            else:
                wandb.init(
                    project="Bayesian DL",
                    name="STN_MNIST",
                    entity="zefko",)
        else:
            if misplacement:
                wandb.init(
                    project="Bayesian DL",
                    name="Vanilla_misMNIST",
                    entity="zefko",)
            else:
                wandb.init(
                    project="Bayesian DL",
                    name="Vanilla_MNIST",
                    entity="zefko",)
    

    # Initialize the model and transfer to GPU if available
    if hype["stn"]:

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

        model = STN.to(device)

    else:
        Vanilla = Vanilla_Net(
        hype["channels"],
        hype["enc_sizes"],
        hype["pool"],
        hype["stride"],
        hype["kernel_size"],
        hype["padding"],
        hype["num_classes"])



        Vanilla = Vanilla_Net(
        
        hype["num_classes"])
        
        model = Vanilla.to(device)
    
    
    logger.info(model)
    
    print('Weights and Biases',[ (name,np.prod(p.size())) for name,p in model.named_parameters() if p.requires_grad])
    

    summary(model,(images.shape[1],images.shape[2],images.shape[3]))

    
    if hype['stn']:
        
        optimizer = optim.Adam([
                    {'params': model.base.parameters()},
                    {'params':model.fc1.parameters()},
                    {'params':model.fc2.parameters()},
                    {'params':model.stn.fc_loc.parameters(),'lr': hype["learning_rate_stn"]},
                    {'params': model.stn.localization.parameters(), 'lr': hype["learning_rate_stn"]}
            ], lr=hype["learning_rate_base"])
    else:
        optimizer = optim.Adam(model.parameters(),hype["learning_rate_base"],)

    criterion = nn.CrossEntropyLoss()

    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input

    

    for epoch in range(1, hype["epochs"] + 1):

        #wandb.watch(model, optimizer, log="all", log_freq=10)
        train_loss = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            target = torch.flatten(target.type(torch.LongTensor)).to(device)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 500 == 0:
                #wandb.log({"epoch": epoch, "loss": loss.item()})
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

            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                target = torch.flatten(target.type(torch.LongTensor)).to(device)
                output = model(data)

                # sum up batch loss
                test_loss += criterion(output, target).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(val_loader.dataset)

            wandb.log(
                {
                    "Validation loss": test_loss,
                    "Validation_accuracy": 100.0 * correct / len(val_loader.dataset),
                }
            )

            print(
                "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                    test_loss,
                    correct,
                    len(val_loader.dataset),
                    100.0 * correct / len(val_loader.dataset),
                )
            )

        # create checkpoint variable and add important data
        checkpoint = {
            "epoch": epoch + 1,
            "valid_loss_min": test_loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        # save checkpoint
        SaveLoad.save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
        if hype['stn']:
            # log the stn outputs
            visualization.wandb_pred(model, val_loader, device)

        ## TODO: save the model if validation loss has decreased
        if test_loss <= valid_loss_min:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                    valid_loss_min, test_loss
                )
            )
            # save checkpoint as best model
            SaveLoad.save_ckp(checkpoint, True, checkpoint_path, best_model_path)
            valid_loss_min = test_loss
        
    wandb.log({'channels':hype["channels"],
    'batch_size':hype["batch_size"],
    'lr base':hype["learning_rate_base"],
    'lr_loc':hype["learning_rate_stn"],
    'loc size':hype["loc_sizes"],
        'base size':hype["enc_sizes"],
        'pool':hype["pool"],
        'stride':hype["stride"],
        'kernel size':hype["kernel_size"],
        'padding':hype["padding"],
        'classes':hype["num_classes"]})
