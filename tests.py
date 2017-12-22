from models import utils
import numpy as np
import math
import tensorflow as tf
import time


def test_out(pass_condition, name):
    if pass_condition:
        print('Passed: ' + name)
        return True
    else:
        print('Failed: ' + name)
        return False


def conv_out_size_test():

    in_size = [12, 12]
    kernel_size = [3, 3]
    strides = [2, 2]
    rates = [1, 1]
    padding = 'VALID'

    out_size = utils.conv_out_size(in_size, kernel_size, strides, rates, padding)

    print(out_size)
    print(np.zeros(in_size))
    print(np.zeros(kernel_size))
    print(np.zeros(out_size))


def get_conv_slices_test():
    kernel_size = [3, 3]
    padding = 'same'
    if padding is 'same':
        pad = [[0, 0],
               [int(math.floor((kernel_size[0] - 1) / 2)), int(math.ceil((kernel_size[0] - 1) / 2))],
               [int(math.floor((kernel_size[1] - 1) / 2)), int(math.ceil((kernel_size[1] - 1) / 2))],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0],
               [0, 0]]
    else:
        pad = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    in_height = 5
    in_width = 5
    strides = [1, 1]

    conv_pose = np.pad(np.reshape(np.arange(in_height*in_width) + 1, [1, in_height, in_width, 1, 1, 1, 1, 1]), pad, 'constant')

    slices = utils.extract_conv_caps_patches(conv_pose, kernel_size, padding, in_height, in_width, strides)

    print(conv_pose)
    print(*[x for x in slices], sep='\n new array \n')


def extract_image_patches_nd_test():
    input_shape1 = [1, 12, 12, 1]
    kernel_size = [5, 5]
    ksizes = [1, *kernel_size, 1]
    strides = [2, 2]  # Will need to change out conditions if changing this, but unlikely changing this would break extract_image_slices_nd
    padding = 'VALID'  # Will need to change out conditions if changing this, but unlikely changing this would break extract_image_slices_nd

    input_tensor1 = tf.reshape(tf.constant(list(range(np.prod(input_shape1)))), input_shape1)

    out1 = utils.extract_image_patches_nd(input_tensor1, ksizes, [1, *strides, 1], [1, 1, 1, 1], padding)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        input_tensor1_out, out1_out = sess.run([input_tensor1, out1])

        out_size = out1_out.shape[4:]

        for i in range(out_size[0]):
            for j in range(out_size[1]):
                print(np.reshape(out1_out[:, :, :, :, i, j], out1_out.shape[1:3]))

        # Expected results:
        #out_shape = [1, 2, 2, 2, 2, 2]
        input_tensor_np = np.reshape(np.arange(np.prod(input_shape1)), input_shape1)
        input_tensor1_0_all_all_0_expected = input_tensor_np[0, :, :, 0]

        input_tensor1_0_all_all_1_expected = input_tensor_np[0, :, :, 1]

        out1_0_all_all_0_0_0_0_expected = input_tensor1_0_all_all_0_expected[0:kernel_size[0], 0:kernel_size[1]]

        out1_0_all_all_1_1_1_0_expected = input_tensor1_0_all_all_1_expected[1:1 + kernel_size[0], 1:1 + kernel_size[1]]

        # True results:
        input_tensor1_0_all_all_0_true = input_tensor1_out[0, :, :, 0]

        input_tensor1_0_all_all_1_true = input_tensor1_out[0, :, :, 1]

        out1_0_all_all_0_0_0_0_true = out1_out[0, :, :, 0, 0, 0]

        out1_0_all_all_1_1_1_0_true = out1_out[0, :, :, 1, 1, 1]

        condition = (np.all(input_tensor1_0_all_all_0_expected == input_tensor1_0_all_all_0_true)) and \
                    (np.all(input_tensor1_0_all_all_1_expected == input_tensor1_0_all_all_1_true)) and \
                    (np.all(out1_0_all_all_0_0_0_0_expected == out1_0_all_all_0_0_0_0_true)) and \
                    (np.all(out1_0_all_all_1_1_1_0_expected == out1_0_all_all_1_1_1_0_true))

        return test_out(condition, 'extract_image_slices_nd_test')


def expand_dims_nd_v_reshape_speed_test():
    a = tf.ones([4, 5, 3])

    b = tf.reshape(a, [4, 1, 5, 3, 1])
    c = utils.expand_dims_nd(a, [1, 3])

    n_repeats = 10000

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        for _ in range(n_repeats):
            b_out = sess.run(b)
        mean_reshape_time = (time.time() - start_time)/n_repeats

        start_time = time.time()
        for _ in range(n_repeats):
            c_out = sess.run(c)
        mean_expand_dims_nd_time = (time.time() - start_time)/n_repeats

        print("Mean reshape time: {} \n Mean expand_dims_nd time: {}".format(mean_reshape_time, mean_expand_dims_nd_time))


def get_reverse_conv_2d_mask_test():
    in_size = [3, 3]
    ksizes = [1, 3, 3, 1]
    strides = [1, 1, 1, 1]
    padding = 'SAME'

    mask = utils.get_reverse_conv_2d_mask(in_size, ksizes, strides, padding=padding)

    print(mask)


def sparse_where_test():
    # RESULT: tf.where DOESNT WORK FOR SPARSE TENSORS
    indices = tf.constant([[1, 1], [0, 0]], dtype=tf.int64)
    values = tf.constant([1, -1])
    dense_shape = tf.constant([2, 2], dtype=tf.int64)
    sparse_tensor = tf.SparseTensor(indices, values, dense_shape)
    zeros = tf.zeros([2, 2], dtype=tf.int32)
    normal_tensor = tf.constant([[1, 0], [0, -1]])

    y = tf.constant(5, shape=[2, 2])

    masked_tensor = tf.where(tf.sparse_tensor_to_dense(sparse_tensor, validate_indices=False) > zeros,
                             tf.sparse_tensor_to_dense(sparse_tensor, validate_indices=False), y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sparse_tensor_np, normal_tensor_np, masked_tensor_np = sess.run([sparse_tensor, normal_tensor, masked_tensor])

        print(sparse_tensor_np)
        print(masked_tensor_np)


def conv_in_size_test():
    out_size = [5, 5]
    kernel_size = [3, 3]
    strides = [2, 2]
    rates = [1, 1]
    padding = 'VALID'

    in_size = utils.conv_in_size(out_size, kernel_size, strides, rates, padding)

    expected_in_size = [12, 12]

    return test_out(in_size == expected_in_size, 'conv_in_size_test')


if __name__ == '__main__':
    #get_conv_slices_test()
    #extract_image_patches_nd_test()
    #expand_dims_nd_v_reshape_speed_test()
    #get_reverse_conv_2d_mask_test()
    #sparse_where_test()
    #conv_in_size_test()
    conv_out_size_test()