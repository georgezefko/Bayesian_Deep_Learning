
import torch.nn as nn
import torch.nn.functional as F
from src.utils import compute_dim


class Base(nn.Module):
    def __init__(self, enc_sizes,pool,stride, kernel, pad):
        super().__init__()

        
        conv_blocks =[compute_dim.conv_block(in_f, out_f, pool=pool, stride=stride,kernel_size=kernel, padding=pad) 
                       for in_f, out_f in zip(enc_sizes, enc_sizes[1:])]

        self.base_net = nn.Sequential(*conv_blocks)
        

    def forward(self,x):
        x = self.base_net(x)
    
        return x

class Vanilla_Net(nn.Module):
    def __init__(self, in_c, enc_sizes, pool,stride,kernel, pad,n_classes):
        super().__init__()
    
        self.enc_sizes = [in_c, *enc_sizes]
        self.pool = pool
        self.stride = stride
        self.kernel = kernel
        self.pad = pad
        self.n_classes = n_classes
        
        self.base = Base(self.enc_sizes,self.pool,self.stride,self.kernel, self.pad)
        self.fc1 = nn.Linear(
            self.enc_sizes[-1] *12* 12 , 50
        )#def 12*12
        self.fc2 = nn.Linear(50, self.n_classes)
        

    def forward(self,x):
        
        x = self.base(x)        
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return x