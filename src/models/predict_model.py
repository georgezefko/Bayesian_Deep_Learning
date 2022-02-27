import logging
from pathlib import Path
from numpy import False_
import torch
from src.models.SpatialTN_2 import Net
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
    default="saved_models/colab_best_misMNIST_20_STN2__ver_2lr.pth",
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
def predict(trained_model_filepath, parameterize=False, misplacement=False):

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

    # test_set_path = str(project_dir) + "/src/data/MNIST/MNIST/processed/test.pt"
    # test_imgs, test_labels = torch.load(test_set_path)
    # test_set = torch.utils.data.TensorDataset(test_imgs, test_labels)
    # test_loader = torch.utils.data.DataLoader(
    #     test_set, batch_size=hype["batch_size"], shuffle=False, num_workers=2
    # )
    # logger.info(f"Length of Test Data : {len(test_set)}")
    _,_, test_loader = make_dataset.data(
        hype["batch_size"], hype["crop_size"], misplacement
    )
    dataiter = iter(test_loader)
    images, _ = dataiter.next()
    print("image shape", images.shape)

    height = images.shape[2]
    width = images.shape[3]

    # STN = Net(
    #     hype["num_classes"],
    #     hype["channels"],
    #     hype["filter1_out"],
    #     hype["filter2_out"],
    #     hype["kernel_size"],
    #     hype["padding"],
    #     hype["stride"],
    #     height,
    #     width,
    #     hype["pool"],
    #     parameterize,
    # )

    STN = Net(
        hype["channels"],
        hype["enc_sizes"],
        hype["loc_sizes"],
        hype["kernel_size"],
        hype["padding"],
        hype["num_classes"],
        parameterize)
    model = STN.to(device)
    logger.info(model)

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

            acc_map = (output.argmax(-1) == target).float().mean()
            ece_map = ECE(bins=15).measure(output.numpy(), target.numpy())
            nll_map = -dists.Categorical(output).log_prob(target).mean()

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
        print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

        accuracy = correct / len(test_loader.dataset)

        return accuracy, acc_map, ece_map, nll_map


if __name__ == "__main__":
    predict()
