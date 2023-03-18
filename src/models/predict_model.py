import logging
from pathlib import Path
from numpy import False_
import torch
from src.models.SpatialTN_2 import Net
from src.models.Vanilla_CNN import Vanilla_Net
from src.models.Hyperparameters import Hyperparameters as hp
from src.utils import SaveLoad
from src.data import make_dataset
import click
import torch.distributions as dists
from netcal.metrics import ECE


@click.command()
@click.argument(
    "trained_model_filepath",
    type=click.Path(),
    default="/zhome/fc/5/104708/Desktop/Thesis/saved_models/Vanilla_MAP.pth"
)
@click.option(
    "-m",
    "--misplacement",
    type=bool,
    default=False,
    help="Select false to train the model on misplacement MNIST (default=False)",
)
@click.option(
    "-pr",
    "--parameterize",
    type=bool,
    default=False,
    help="Select false to train only the scale theta parameters use only on misplacement MNIST (default=False)",
)

@click.option(
    "-lo",
    "--load",
    type=bool,
    default=True,
    help="Select false to create data (default=True)",
)
def predict(trained_model_filepath, misplacement=False,
    parameterize=False,
    load=True,
    save=True,
    subset=False, ):

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

    _,_,test_loader= make_dataset.data(hype["batch_size"], hype["crop_size"], hype["train_subset"], hype["dataset"],
                                misplacement,load,save,subset)

    dataiter = iter(test_loader)
    images, _ = dataiter.next()
    print("image shape", images.shape)



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
        
        model = Vanilla.to(device)
    

    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )

    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"])
    # model.load_state_dict(state_dict)

    with torch.no_grad():
        model.eval()
        correct = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            ps = torch.exp(output)

            # Keep track of how many are correctly classified
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == target.view(*top_class.shape)
            correct += equals.type(torch.FloatTensor).sum().item()

        print(
            "\nTest set: Accuracy: {}/{} ({:.0f}%)\n".format(
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

        accuracy = correct / len(test_loader.dataset)

        return accuracy


if __name__ == "__main__":
    predict()
