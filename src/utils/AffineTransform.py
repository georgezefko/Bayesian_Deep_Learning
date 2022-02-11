import torch


def make_affine_matrix(rot, scale, scale_e, translation_x, translation_y):
    # rot is rotation angle in radians
    a = scale * torch.cos(rot)
    b = -scale * torch.sin(rot)
    c = translation_x

    d = scale * torch.sin(rot)
    e = scale_e * torch.cos(rot)
    f = translation_y

    param_tensor = torch.stack([a, b, c, d, e, f], dim=-1)

    affine_matrix = param_tensor.view([-1, 2, 3])
    return affine_matrix


def make_affine_parameters(params):
    # case where you predict 2 params, the scale (i.e. size of the crop/bounding box is sth you can choose)
    if params.shape[-1] == 2:  # only perform crop - fix scale and rotation.
        theta = torch.zeros([params.shape[0]], device=params.device)
        scale = 1 * torch.ones([params.shape[0]], device=params.device)
        scale_e = 0.5 * torch.ones([params.shape[0]], device=params.device)

        translation_x = params[:, 0]
        translation_y = params[:, 1]
        affine_matrix = make_affine_matrix(
            theta, scale, scale_e, translation_x, translation_y
        )

    # 4 parameters -- this is usually what we use
    elif params.shape[-1] == 4:
        theta = params[:, 0]
        scale = params[:, 1]
        translation_x = params[:, 2]
        translation_y = params[:, 3]
        affine_matrix = make_affine_matrix(theta, scale, translation_x, translation_y)

    # directly predicting 6 parameters (not recommended)
    elif params.shape[-1] == 6:
        affine_matrix = params.view([-1, 2, 3])

    return affine_matrix  # [S * bs, 2, 3]
