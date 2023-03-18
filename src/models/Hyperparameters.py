class Hyperparameters:
    def __init__(
        self,
        config={
            "batch_size": 100,
            "num_classes":10,
            "channels": 1,
            "filter1_out": 16,
            "kernel_size": 5,
            "pool": 2,
            "filter2_out": 32,
            "padding": 0,
            "stride": 1,
            "learning_rate_base": 0.01,
            "learning_rate_stn":0.001,
            "epochs": 50,
            "crop_size": 128,
            "enc_sizes":[16,32],#12,24
            "loc_sizes":[8,10],#32,64
            "train_subset":1000,
            "dataset":'MNIST',
            "stn":True,
        },
    ):
        super().__init__()
        self.config = config

    def config(self):
        return self.config
