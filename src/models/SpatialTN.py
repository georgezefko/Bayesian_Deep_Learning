import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import compute_dim
from src.utils import AffineTransform


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
        parameterize,
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
        self.parameterize = parameterize

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

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
        # Regressor for the affine matrix
        self.fc_loc = nn.Sequential(
            #nn.Linear(10 * 3 * 3, 32),  # original
            nn.Linear(10 * 28 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 1 if self.parameterize else 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        if self.parameterize:
            self.fc_loc[2].bias.data.copy_(torch.tensor([1], dtype=torch.float))

        else:
            self.fc_loc[2].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)

        #xs = xs.view(-1, 10 * 3 * 3)  # original
        xs = xs.view(-1, 10 * 28 * 28)

        theta = self.fc_loc(xs)

        if self.parameterize:
            theta = AffineTransform.make_affine_parameters(theta)
        else:
            theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):

        # transform the input
        x = self.stn(x)

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

        return F.log_softmax(x, dim=1)
