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
from laplace.utils import ModuleNameSubnetMask





class Net(nn.Module):
    def __init__(
        self,
        num_classes,
        channels,
        filter_1_out,
        filter_2_out,
        kernel_size,
        padding,
        stride,
        height,
        width,
        pool,
        
    ):
        super(Net, self).__init__()
        self.num_classes = (num_classes,)
        self.channels = (channels,)
        self.filter_1_out = (filter_1_out,)
        self.filter_2_out = (filter_2_out,)
        self.kernel_size = (kernel_size,)
        self.padding = (padding,)
        self.stride = (stride,)
        self.height = (height,)
        self.width = (width,)
        self.pool = (pool,)
        

        self.conv1 = nn.Conv2d(channels, filter_1_out, kernel_size)
        # evaluating image dimensions after first connvolution
        self.conv1_out_height = compute_dim.compute_conv_dim(
            height, kernel_size, padding, stride
        )
        self.conv1_out_width = compute_dim.compute_conv_dim(
            width, kernel_size, padding, stride
        )

        # first pooling
        self.pool1 = nn.MaxPool2d(pool, pool)
        # evaluating image dimensions after first pooling
        self.conv2_out_height = compute_dim.compute_pool_dim(
            self.conv1_out_height, pool, pool
        )
        self.conv2_out_width = compute_dim.compute_pool_dim(
            self.conv1_out_width, pool, pool
        )

        # Second Convolution
        self.conv2 = nn.Conv2d(filter_1_out, filter_2_out, kernel_size)
        # evaluating image dimensions after second convolution
        self.conv3_out_height = compute_dim.compute_conv_dim(
            self.conv2_out_height, kernel_size, padding, stride
        )
        self.conv3_out_width = compute_dim.compute_conv_dim(
            self.conv2_out_width, kernel_size, padding, stride
        )
        self.conv2_drop = nn.Dropout2d()

        # Second pooling
        self.pool2 = nn.MaxPool2d(pool, pool)
        # evaluating image dimensions after second pooling
        self.conv4_out_height = compute_dim.compute_pool_dim(
            self.conv3_out_height, pool, pool
        )
        self.conv4_out_width = compute_dim.compute_pool_dim(
            self.conv3_out_width, pool, pool
        )

        self.fc1 = nn.Linear(
            filter_2_out * self.conv4_out_height * self.conv4_out_width, 50
        )
        self.fc2 = nn.Linear(50, num_classes)

        

    def forward(self, x):

        
        # convolutional layer 1
        x = F.relu(self.pool1(self.conv1(x)))

        # convolutional layer 2
        x = F.relu(self.pool2(self.conv2_drop(self.conv2(x))))

        x = x.view(
            -1, self.filter_2_out[0] * self.conv4_out_height * self.conv4_out_width
        )

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x

def laplace(model,dataloader,method='last',module='fc1'):
    if method == 'last':
        la = Laplace(
            model,
            "classification",
            subset_of_weights="last_layer",
            hessian_structure="kron",
        )
        la.fit(dataloader)
        la.optimize_prior_precision(method="marglik")
    else:
        subnetwork_mask = ModuleNameSubnetMask(model, module_names=[module])
        subnetwork_mask.select()
        subnetwork_indices = subnetwork_mask.indices
        la = Laplace(
            model,
            "classification",
            subset_of_weights="subnetwork",
            hessian_structure="full",
            subnetwork_indices = subnetwork_indices#.type(torch.LongTensor),
        )
        la.fit(dataloader)
        
    return la







def main():
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

    height = images.shape[2]
    width = images.shape[3]

    Vanilla = Net(
        hype["num_classes"],
        hype["channels"],
        hype["filter1_out"],
        hype["filter2_out"],
        hype["kernel_size"],
        hype["padding"],
        hype["stride"],
        height,
        width,
        hype["pool"],
    )

    
    model = Vanilla.to(device).eval()
    logger.info(model)

    trained_model_filepath = '/Users/georgioszefkilis/Bayesian_Deep_Learning/saved_models/colab_best_Vanilla_MNIST_20.pth'
    state_dict = torch.load(
        project_dir.joinpath(trained_model_filepath), map_location=torch.device(device)
    )

    # initialize state_dict from checkpoint to model
    model.load_state_dict(state_dict["state_dict"])

    @torch.no_grad()
    def predict(dataloader, model, laplace=False):
        py = []
        target = []
        for x, t in dataloader:
            x,t = x.to(device),t.to(device)
            target.append(t)
            if laplace:
                py.append(model(x))
            else:
                py.append(torch.softmax(model(x), dim=-1))

        images = torch.cat(py).cpu()
        labels =torch.cat(target, dim=0).cpu()
        acc_map = (images.argmax(-1) == labels).float().mean()
        ece_map = ECE(bins=15).measure(images.numpy(), labels.numpy())
        nll_map = -dists.Categorical(images).log_prob(labels).mean()
        
        return acc_map,ece_map,nll_map

    logger.info("MAP evaluation")
    print('Map evaluation')

    acc_map,ece_map,nll_map = predict(test_loader, model, laplace=False)

    print(f"[MAP] Acc.: {acc_map:.1%}; ECE: {ece_map:.1%}; NLL: {nll_map:.3}")

    
    # Laplace
    la = laplace(model,train_loader,method='last')
    acc_laplace,ece_laplace,nll_laplace = predict(test_loader, la, laplace=True)

    print(
        f"[Laplace] Acc.: {acc_laplace:.1%}; ECE: {ece_laplace:.1%}; NLL: {nll_laplace:.3}"
    )

    #subetwork laplace
    sub_laplace = laplace(model,train_loader,method='sub',module='fc1')
    acc_sublaplace,ece_sublaplace,nll_sublaplace = predict(test_loader, sub_laplace, laplace=True)

    print(
        f"[Laplace] Acc.: {acc_sublaplace:.1%}; ECE: {ece_sublaplace:.1%}; NLL: {nll_sublaplace:.3}"
    )



if __name__ == "__main__":
    main()
