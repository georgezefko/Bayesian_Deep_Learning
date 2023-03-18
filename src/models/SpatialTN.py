import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import compute_dim
from src.utils import AffineTransform





class Base(nn.Module):
    def __init__(self, enc_sizes,pool,stride,kernel, pad):
        super().__init__()

        
        conv_blocks =[compute_dim.conv_block(in_f, out_f,pool,stride, kernel_size=kernel, padding=pad) 
                            for in_f, out_f in zip(enc_sizes, enc_sizes[1:])]

        self.base_net = nn.Sequential(*conv_blocks)
        

    def forward(self,x):
        x = self.base_net(x)
    
        return x


class STN(nn.Module):
    def __init__(self,loc_sizes,pool,stride,kernel,pad,parameterize):
        super().__init__()
    
        self.parameterize =parameterize
        
        conv_blocks =[compute_dim.conv_block(in_f, out_f,pool,stride, kernel_size=kernel, padding=pad) 
                       for in_f, out_f in zip(loc_sizes, loc_sizes[1:])]

        self.localization = nn.Sequential(*conv_blocks)

        # Regressor for the affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(loc_sizes[-1] *4*4,32),#32
            nn.ReLU(),
            nn.Linear(32, 2 * 1 if parameterize else 3 * 2),#20
        )
        
        self.fc_loc[-1].weight.data.zero_()
        if parameterize:
            bias = torch.tensor([0,0], dtype=torch.float)
            self.fc_loc[-1].bias.data.copy_(bias[:2].view(-1))

        else:
        
            self.fc_loc[-1].bias.data.copy_(
                 torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
            )
        

    def forward(self,x):
        xs = self.localization(x)
        
        xs = xs.view(-1, xs.shape[1] * xs.shape[2] * xs.shape[3])
        theta = self.fc_loc(xs)

        if self.parameterize:
            theta = AffineTransform.make_affine_parameters(theta)
        else:
            theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid)

        return x

class Net(nn.Module):
    def __init__(self, in_c, enc_sizes, loc_sizes,pool,stride, kernel, pad,n_classes,parameterize):
        super().__init__()
    
        self.enc_sizes = [in_c, *enc_sizes]
        self.loc_sizes = [in_c, *loc_sizes]
        self.pool =pool
        self.stride = stride
        self.kernel = kernel
        self.pad = pad
        self.parameterize = parameterize
        self.n_classes = n_classes
        
        self.base = Base(self.enc_sizes,self.pool,self.stride,self.kernel,self.pad)
        self.stn = STN(self.loc_sizes ,self.pool,self.stride,self.kernel, self.pad, self.parameterize)
        self.fc1 = nn.Linear(
            self.enc_sizes[-1] * 4* 4, 50
        )
        self.fc2 = nn.Linear(50, self.n_classes)
        

    def forward(self,x):
        # transform the input
        x = self.stn(x)

        x = self.base(x)
        
        x = x.flatten(1)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x

