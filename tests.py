from models import utils
import numpy as np
import math


def conv_out_size_test():

    in_size = [12, 10]
    kernel_size = [3, 4]
    strides = [2, 1]
    padding = 'valid'

    out_size = utils.conv_out_size(in_size, kernel_size, strides, padding)

    print(out_size)
    print(np.zeros(in_size))
    print(np.zeros(kernel_size))
    print(np.zeros(out_size))


def get_conv_slices_test():
    kernel_size = [3, 3]
    padding = 'valid'
    if padding is 'same':
        pad = [[int(math.floor((kernel_size[0] - 1) / 2)), int(math.ceil((kernel_size[0] - 1) / 2))],
               [int(math.floor((kernel_size[1] - 1) / 2)), int(math.ceil((kernel_size[1] - 1) / 2))]]
    else:
        pad = [[0, 0], [0, 0]]
    in_height = 5
    in_width = 5
    strides = [1, 1]

    conv_pose = np.pad(np.reshape(np.arange(in_height*in_width) + 1, [in_height, in_width]), pad, 'constant')

    pad = [[int(math.floor((kernel_size[0] - 1) / 2)), int(math.ceil((kernel_size[0] - 1) / 2))],
           [int(math.floor((kernel_size[1] - 1) / 2)), int(math.ceil((kernel_size[1] - 1) / 2))]]

    slices = utils.get_conv_caps_slices(conv_pose, kernel_size, pad, padding, in_height, in_width, strides)

    print(conv_pose)
    print(*[slice for slice in slices], sep='\n')


if __name__ == '__main__':
    get_conv_slices_test()