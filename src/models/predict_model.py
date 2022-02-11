import logging
from pathlib import Path
import wandb

import torch
import torch.nn.functional as F
from src.models.SpatialTN import Net
from src.models.Hyperparameters import Hyperparameters as hp
from src.utils import SaveLoad


def predict(trained_model_filepath):
    """Evaluates the trained network using test subset of data"""
    logger = logging.getLogger(__name__)
    logger.info("Evaluating a trained network using a test subset")

    # Check if there is a GPU available to use
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained
    hype = hp().config
    model = Net(
        hype["num_classes"],
        hype["channels"],
        hype["filter1_out"],
        hype["filter2_out"],
        hype["image_height"],
        hype["image_width"],
        hype["padding"],
        hype["stride"],
        hype["kernel_size"],
        hype["pool"],
    )
    model = model.to(device)

    project_dir = Path(__file__).resolve().parents[2]
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )

    model.load_state_dict(state_dict)

    # Load the test data
    test_set_path = str(project_dir) + "/data/processed/test.pt"
    test_imgs, test_labels = torch.load(test_set_path)
    test_set = torch.utils.data.TensorDataset(test_imgs, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=hype["batch_size"], shuffle=False, num_workers=2
    )
    logger.info(f"Length of Test Data : {len(test_set)}")

    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        wandb.log(
            {
                "Test loss": test_loss,
                "Test_accuracy": 100.0 * correct / len(test_loader.dataset),
            }
        )

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
        accuracy = correct / len(test_loader.dataset)
        return accuracy, test_loss
