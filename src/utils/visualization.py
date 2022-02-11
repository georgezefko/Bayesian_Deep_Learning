import wandb
import torch
import torchvision
import numpy as np


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def wandb_pred(model, test_loader, device, epoch):
    # create a wandb Artifact to version each test step separately
    test_data_at = wandb.Artifact(
        "test_samples_" + str(wandb.run.id), type=f'"Transformations at"{epoch}'
    )
    # create a wandb.Table() in which to store predictions for each test step
    columns = ["Grid in", "Grid out"]
    test_table = wandb.Table(columns=columns)

    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)
        # for batch, (data,_) in enumerate(test_loader):
        # data = data.to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor)
        )

        # Plot the results side-by-side
        test_table.add_data(wandb.Image(in_grid), wandb.Image(out_grid))

    # log predictions table to wandb
    test_data_at.add(test_table, f'"Transformations at"{epoch}')
    wandb.run.log_artifact(test_data_at)
