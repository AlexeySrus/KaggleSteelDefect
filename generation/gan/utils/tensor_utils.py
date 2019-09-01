import torch
import numpy as np


def crop_batch_by_center(x, shape):
    """
    Crop target area from x image tensor by new shape, shape[:-2] < x.shape[:-2]
    Args:
        x: input image 4-D tensor
        shape: result shape

    Returns:
        cropped image tensor
    """
    target_shape = shape[-2:]
    input_tensor_shape = x.shape[-2:]

    crop_by_y = (input_tensor_shape[0] - target_shape[0]) // 2
    crop_by_x = (input_tensor_shape[1] - target_shape[1]) // 2

    indexes_by_y = (
        crop_by_y, input_tensor_shape[0] - crop_by_y
    )

    indexes_by_x = (
        crop_by_x, input_tensor_shape[1] - crop_by_x
    )

    return x[:, :, indexes_by_y[0]:indexes_by_y[1], indexes_by_x[0]:indexes_by_x[1]]


def reshape_tensor(x, new_shape=(224, 224)):
    scales = np.array(new_shape) / np.array(x.shape[-2:])
    return torch.nn.functional.interpolate(
        x, scale_factor=scales, mode='nearest'
    )


def flatten(x):
    n = 1
    for d in x.shape[1:]:
        n *= d
    return x.view(x.shape[0], n)


def L1_norm(x):
    x_sum = x.sum()
    return x / x_sum


def gaussian_pdf(x, mu, sigma_sq):
    return (1/torch.sqrt(2*np.pi*sigma_sq)) * torch.exp((-1/(2*sigma_sq)) * torch.norm((x-mu), 2, 1)**2)