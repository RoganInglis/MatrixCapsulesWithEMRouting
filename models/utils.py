import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# TODO - make sure all functions have been fully tested
# TODO - sort out interchangeable use of 'size' and 'shape'; change to 'shape'
# TODO - make sure use of dense_shape, full_shape and patches_shape is consistent


def conv_out_size(in_size, kernel_size, strides, rates, padding):
    in_size = np.array(in_size)
    k = np.array(kernel_size)
    s = np.array(strides)
    r = np.array(rates)

    k_dash = k + (k - 1) * (r - 1)

    if padding is 'VALID':
        p = np.array([0, 0])
    else:
        # Skipping the factor of 1/2 followed by the factor of 2 here as p is not needed for anything else here
        p = k_dash - 1

    out_size = (in_size + p - k_dash)/s + 1

    return list(out_size.astype(np.int32))


def conv_in_size(out_size, kernel_size, strides, rates, padding):
    # NOTE - the desired output from this is ambiguous in some cases (strides > kernel_size/2 (?))
    # Convert sizes to numpy arrays
    out_size = np.array(out_size)
    k = np.array(kernel_size)
    s = np.array(strides)
    r = np.array(rates)

    k_dash = k + (k - 1)*(r - 1)

    if padding is 'VALID':
        p = np.array([0, 0])
    else:
        # Skipping the factor of 1/2 followed by the factor of 2 here as p is not needed for anything else here
        p = (k_dash - 1)

    in_size = (out_size - 1)*s + k_dash - p

    return list(in_size.astype(np.int32))


def get_correct_conv_param(param):
    if type(param) is int:
        param = [1, param, param, 1]
    elif len(param) == 2:
        param = [1, *param, 1]
    elif len(param) == 4:
        param = list(param)
    else:
        raise ValueError('Incorrect parameter format')
    return param


def get_correct_ksizes_strides_rates(ksizes, strides, rates):
    """
    Take int, list or tuple values for ksizes, strides and rates and convert to correct form
    :param ksizes: int, list or tuple
    :param strides: int, list or tuple
    :param rates: int, list or tuple
    :return: ksizes, strides, rates in correct form (4 element list)
    """
    ksizes = get_correct_conv_param(ksizes)
    strides = get_correct_conv_param(strides)
    rates = get_correct_conv_param(rates)
    return ksizes, strides, rates


def save_mnist_as_image(mnist_batch, outdir, name="image"):
    # If outdir doesn't exist then create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for i, image in enumerate(tqdm(mnist_batch, desc="Saving images", leave=False, ncols=100)):
        image = np.squeeze(image)
        plt.imsave("{}/{}_{}.png".format(outdir, name, i), image)
