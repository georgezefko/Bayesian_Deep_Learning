import torch
import torch.nn as nn


def compute_conv_dim(dim_size, kernel_size, padding, stride):
    # (I-F)+2*P/S +1
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)


def compute_pool_dim(dim_size, kernel_size, stride):
    # (I-F)/S +1
    return int((dim_size - kernel_size) / stride + 1)

def conv_block(in_f, out_f, pool,stride,*args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        #nn.Conv2d(out_f, out_f, *args, **kwargs),
        nn.MaxPool2d(pool, pool),
        nn.BatchNorm2d(out_f),
        
        nn.ReLU()
    )

def convtwo_block(in_f, out_f, pool,stride,*args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.Conv2d(out_f, out_f, *args, **kwargs),
        nn.MaxPool2d(pool, pool),
        nn.BatchNorm2d(out_f),
        nn.ReLU(),
    )
