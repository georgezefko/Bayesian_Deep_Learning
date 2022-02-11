import logging

import click

from src.models.train_model import train


@click.command()
@click.argument("check_path", type=click.Path(), default="models/checkpoint.pth")
@click.argument("model_path", type=click.Path(), default="models/best_model.pth")
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
    "-pr",
    "--parameterize",
    type=bool,
    default=False,
    help="Select false to train only the scale theta parameters use only on misplacement MNIST (default=False)",
)
def execute_model(
    check_path, model_path, valid_loss_min_input, misplacement, parameterize
):
    """Trains the neural network using MNIST training data"""
    _ = train(check_path, model_path, valid_loss_min_input, misplacement, parameterize)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    execute_model()
