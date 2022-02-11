import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.compute_dim import compute_conv_dim, compute_pool_dim
from utils.AffineTransform import make_affine_parameters


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
        scale,
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
        self.scale = (scale,)
        parameterize = self.scale in ["scale"]

        self.conv1 = nn.Conv2d(channels, filter_1_out, kernel_size)
        # evaluating image dimensions after first connvolution
        self.conv1_out_height = compute_conv_dim(height, kernel_size, padding, stride)
        self.conv1_out_width = compute_conv_dim(width, kernel_size, padding, stride)

        # first pooling
        self.pool1 = nn.MaxPool2d(pool, pool)
        # evaluating image dimensions after first pooling
        self.conv2_out_height = compute_pool_dim(self.conv1_out_height, pool, pool)
        self.conv2_out_width = compute_pool_dim(self.conv1_out_width, pool, pool)

        # Second Convolution
        self.conv2 = nn.Conv2d(filter_1_out, filter_2_out, kernel_size)
        # evaluating image dimensions after second convolution
        self.conv3_out_height = compute_conv_dim(
            self.conv2_out_height, kernel_size, padding, stride
        )
        self.conv3_out_width = compute_conv_dim(
            self.conv2_out_width, kernel_size, padding, stride
        )
        self.conv2_drop = nn.Dropout2d()

        # Second pooling
        self.pool2 = nn.MaxPool2d(pool, pool)
        # evaluating image dimensions after second pooling
        self.conv4_out_height = compute_pool_dim(self.conv3_out_height, pool, pool)
        self.conv4_out_width = compute_pool_dim(self.conv3_out_width, pool, pool)

        self.fc1 = nn.Linear(
            filter_2_out * self.conv4_out_height * self.conv4_out_width, 50
        )
        # print(self.fc1)
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
        # apply laplace to the last linera layer for the first attempt
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            # nn.Linear(10 * 3 * 3, 32),#original
            nn.Linear(10 * 28 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 2 * 1 if parameterize else 3 * 2),
        )

        # Initialize the weights/bias with identity transformation
        if parameterize:
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1], dtype=torch.float))

        else:
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(
                torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )

    # Spatial transformer network forward function
    def stn(self, x):
        # print('x',x.size())
        xs = self.localization(x)

        # print('xs',xs.size())
        # xs = xs.view(-1, 10 * 3 * 3) #original
        xs = xs.view(-1, 10 * 28 * 28)

        # print('xs view',xs.size())
        theta = self.fc_loc(xs)
        # print('theta before view',theta.shape)
        if self.parameterize:
            theta = make_affine_parameters(theta)
        else:
            theta = theta.view(-1, 2, 3)
        # print('theta',theta.shape)
        # print(theta)
        # print('size',x.size())

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):

        # print('input',x.size())
        # transform the input
        x = self.stn(x)

        # print('transform',x.size())
        # Perform the usual forward pass
        # convolutional layer 1
        x = F.relu(self.pool1(self.conv1(x)))
        # print('forward1',x.size())

        # convolutional layer 2
        x = F.relu(self.pool2(self.conv2_drop(self.conv2(x))))
        # print('forward2',x.size())

        # convolutional layer 3
        # x = F.relu(self.pool3(self.conv3_drop(self.conv3(x))))
        # print('forward3',x.size())

        # x = x.view(-1, 320) #original
        # print(self.conv3_out_height)
        # print(self.conv4_out_width)
        x = x.view(-1, self.filter_2_out * self.conv4_out_height * self.conv4_out_width)

        # print('flatten',x.size())

        x = F.relu(self.fc1(x))
        # print('forward4',x.size())

        x = F.dropout(x, training=self.training)
        # print('forward5',x.size())

        x = self.fc2(x)
        # print('forward6',x.size())

        return F.log_softmax(x, dim=1)
