import logging

import click

from src.models.train_model import train
#from src.models.train_gdsrb import train
#from src.models.train_pola_exps import train

@click.command()
@click.argument("check_path", type=click.Path(), default="saved_models/MNIST_10K_test.pth")
@click.argument(
    "model_path", type=click.Path(), default="saved_models/best_MNIST_10K_test.pth"
)
#@click.argument(
#    "pcam_path", type=click.Path(exists=True), default="src/data/PCAM/")
@click.option(
    "-val",
    "--valid_loss_min_input",
    type=int,
    default=0.05,
    help="minimum validation loss threshold for saving model (default=0.05)",
)
@click.option(
    "-m",
    "--misplacement",
    type=bool,
    default=False,
    help="Select false to train the model on misplacement MNIST (default=False)",
)
@click.option(
    "-ro",
    "--rotation",
    type=bool,
    default=False,
    help="Select false to train the model on rotated MNIST (default=False)",
)
@click.option(
    "-pr",
    "--parameterize",
    type=bool,
    default=False,
    help="Select false to train only the scale theta parameters use only on misplacement MNIST (default=False)",
)

@click.option(
    "-lo",
    "--load",
    type=bool,
    default=True,
    help="Select false to create data (default=True)",
)
@click.option(
    "-sa",
    "--save",
    type=bool,
    default=True,
    help="Select false to not overwrite existing data (default=True)",
)

@click.option(
    "-sub",
    "--subset",
    type=bool,
    default=False,
    help="Select True to generate subsets (default=False)",
)





def execute_model(
    check_path, model_path, valid_loss_min_input,misplacement, rotation, parameterize,load,save,subset
):
    """Trains the neural network using MNIST training data"""
    _ = train(check_path, model_path, valid_loss_min_input,misplacement,rotation, parameterize,load,save,subset)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    execute_model()
