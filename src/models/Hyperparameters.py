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
            "lr": 0.001,
            "epochs": 5,
            "crop_size": 128,
        },
    ):
        super().__init__()
        self.config = config

    def config(self):
        return self.config
