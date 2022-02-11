import logging

import click

from src.models.train_model import train


@click.command()
@click.argument("check_path", type=click.Path(), default="models/checkpoint.pth")
@click.argument("model_path", type=click.Path(), default="models/best_model.pth")
@click.argument(
    "training_figures_filepath", type=click.Path(), default="reports/figures/"
)
@click.option(
    "-val",
    "--min_val",
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
def execute_model(check_path, model_path, valid_loss_min_input, misplacement):
    """Trains the neural network using MNIST training data"""
    _ = train(check_path, model_path, valid_loss_min_input, misplacement)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    execute_model()
