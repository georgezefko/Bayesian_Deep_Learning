class Hyperparameters:
    def __init__(
        self,
        config={
            "batch_size": 64,
            "num_classes": 10,
            "channels": 1,
            "filter1_out": 16,
            "kernel_size": 5,
            "pool": 2,
            "filter2_out": 32,
            "padding": 0,
            "stride": 1,
            "learning_rate_base": 0.001,
            "learning_rate_stn":0.0001,
            "epochs": 5,
            "crop_size": 128,
            "enc_sizes":[16,32,64],
            "loc_sizes":[8,16,32,64],
        },
    ):
        super().__init__()
        self.config = config

    def config(self):
        return self.config
