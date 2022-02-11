def compute_conv_dim(dim_size, kernel_size, padding, stride):
    # (I-F)+2*P/S +1
    return int((dim_size - kernel_size + 2 * padding) / stride + 1)


def compute_pool_dim(dim_size, kernel_size, stride):
    # (I-F)/S +1
    return int((dim_size - kernel_size) / stride + 1)
