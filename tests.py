import numpy as np
import math
import tensorflow as tf
import time

from models import utils
from models import tf_ops


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


def extract_image_patches_nd_test():
    input_shape1 = [1, 12, 12, 1]
    kernel_size = [5, 5]
    ksizes = [1, *kernel_size, 1]
    strides = [2, 2]  # Will need to change out conditions if changing this, but unlikely changing this would break extract_image_slices_nd
    padding = 'VALID'  # Will need to change out conditions if changing this, but unlikely changing this would break extract_image_slices_nd

    input_tensor1 = tf.reshape(tf.constant(list(range(np.prod(input_shape1)))), input_shape1)

    out1 = tf_ops.extract_image_patches_nd(input_tensor1, ksizes, [1, *strides, 1], [1, 1, 1, 1], padding)

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
    c = tf_ops.expand_dims_nd(a, [1, 3])

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


def get_full_indices_test():
    kernel_rows = 3
    kernel_cols = 3
    in_capsules = 32
    out_rows = 6
    out_cols = 6
    out_capsules = 32
    capsule_dim1 = 4
    capsule_dim2 = 4
    strides = [1, 2, 2, 1]
    rates = [1, 1, 1, 1]
    in_rows = 12
    in_cols = 12
    k_dash = [3, 3]
    p_rows = 1
    p_cols = 1

    indices = []
    # Construct for first batch element, in capsule and out capsule
    start = time.time()
    for i_k in range(kernel_rows):
        for j_k in range(kernel_cols):
            for c_i in range(in_capsules):
                for i_o in range(out_rows):
                    for j_o in range(out_cols):
                        # Need to take into account strides, rates, padding
                        # Can't have padding on the outside as we need only indices within the original
                        # image. Can switch it to the other side of the kernel as the rest of the full
                        # array should be zeros anyway.
                        # If padding is on top/left we need to switch it to the bottom/right by adding k_dash to the index
                        # If padding is on the bottom/right, need to switch it to the top/left by subtracting k_dash from index
                        row = i_o * strides[1] + i_k * rates[
                            1] - p_rows
                        if row < 0:
                            row = row + k_dash[0]
                        elif row > in_rows - 1:
                            row = row - k_dash[0]

                        col = j_o * strides[2] + j_k * rates[
                            2] - p_cols
                        if col < 0:
                            col = col + k_dash[1]
                        elif col > in_cols - 1:
                            col = col - k_dash[1]

                        indices.append(np.array([row, col, c_i, i_o, j_o]))
    np_indices = np.stack(indices)

    np_indices = np.repeat(np_indices, out_capsules * capsule_dim1 * capsule_dim2, axis=0)
    capsule_dim2_indices = np.tile(np.arange(capsule_dim2), out_capsules * capsule_dim1)
    capsule_dim1_indices = np.tile(np.repeat(np.arange(capsule_dim1), capsule_dim2), out_capsules)
    out_capsules_indices = np.repeat(np.arange(out_capsules), capsule_dim2 * capsule_dim1)
    extra_indices = np.transpose(np.stack([out_capsules_indices, capsule_dim1_indices, capsule_dim2_indices], axis=0))
    extra_indices = np.tile(extra_indices, [kernel_rows * kernel_cols * in_capsules * out_rows * out_cols, 1])
    np_indices = np.concatenate([np_indices, extra_indices], axis=1)
    print("Reconstruction indices np time: {}s".format(time.time() - start))

    indices = []
    # Construct for first batch element, in capsule and out capsule
    # TODO - can do this more efficiently by doing loops separately and combining (currently adds ~35s to the graph construction time)
    start = time.time()  # TODO - for debugging; remove
    for i_k in range(kernel_rows):
        for j_k in range(kernel_cols):
            for c_i in range(in_capsules):
                for i_o in range(out_rows):
                    for j_o in range(out_cols):
                        for c_o in range(out_capsules):
                            for c_d1 in range(capsule_dim1):
                                for c_d2 in range(capsule_dim2):
                                    # Need to take into account strides, rates, padding
                                    # Can't have padding on the outside as we need only indices within the original
                                    # image. Can switch it to the other side of the kernel as the rest of the full
                                    # array should be zeros anyway.
                                    # If padding is on top/left we need to switch it to the bottom/right by adding k_dash to the index
                                    # If padding is on the bottom/right, need to switch it to the top/left by subtracting k_dash from index
                                    row = i_o * strides[1] + i_k * rates[
                                        1] - p_rows  # TODO - something related to strides here is causing incorrect kernel reconstruction when strides>1
                                    if row < 0:
                                        row = row + k_dash[0]
                                    elif row > in_rows - 1:
                                        row = row - k_dash[0]

                                    col = j_o * strides[2] + j_k * rates[
                                        2] - p_cols  # TODO - something related to strides here is causing incorrect kernel reconstruction when strides>1
                                    if col < 0:
                                        col = col + k_dash[1]
                                    elif col > in_cols - 1:
                                        col = col - k_dash[1]

                                    indices.append(np.array([row, col, c_i, i_o, j_o, c_o, c_d1, c_d2]))
    np_loop_indices = np.stack(indices)
    print("Reconstruction indices loop time: {}s".format(time.time() - start))  # TODO - for debugging; remove

    condition = np.all(np.equal(np_indices, np_loop_indices))
    return test_out(condition, 'get_full_indices_test')


def repeat_test():
    a_np = np.random.uniform(size=[3, 6, 5, 7])
    a = tf.constant(a_np)
    b1 = tf_ops.repeat(a, 3, axis=0)
    b2 = tf_ops.repeat(a, 5, axis=1)
    b3 = tf_ops.repeat(a, 7, axis=3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        b1_np, b2_np, b3_np = sess.run([b1, b2, b3])

    b1_np_repeat = np.repeat(a_np, 3, axis=0)
    b2_np_repeat = np.repeat(a_np, 5, axis=1)
    b3_np_repeat = np.repeat(a_np, 7, axis=3)

    condition = (np.all(np.equal(b1_np, b1_np_repeat))) and (np.all(np.equal(b2_np, b2_np_repeat))) and (np.all(np.equal(b3_np, b3_np_repeat)))
    return test_out(condition, 'repeat_test')


def tf_zero_div_test():
    a = tf.zeros([1, 2])
    b = tf.zeros_like(a)

    div = tf.divide(a, b)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        div_out = sess.run(div)
    print(div_out)


def gather_nd_test():
    a = tf.random_normal([2, 3, 3, 2])
    indices = tf.constant([[0, 0, 0, 0],
                           [1, 2, 2, 1]])

    b = tf.gather_nd(a, indices)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a_np, b_np = sess.run([a, b])

        print(a_np)
        print(b_np)


def sparse_values_op_test():
    indices = tf.constant([[1, 1], [0, 0]], dtype=tf.int64)
    values = tf.constant([1., 2.])
    dense_shape = tf.constant([2, 2], dtype=tf.int64)
    sparse_tensor = tf.SparseTensor(indices, values, dense_shape)

    log_sparse_tensor = tf.SparseTensor(sparse_tensor.indices, tf.log(sparse_tensor.values), sparse_tensor.dense_shape)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        log_sparse_tensor_np = sess.run(log_sparse_tensor)

        print(log_sparse_tensor_np)


def indices_reshape_test():
    indices = tf.constant([[1, 1], [0, 0]], dtype=tf.int64)
    values = tf.constant([1., 2.])
    dense_shape = tf.constant([2, 2], dtype=tf.int64)
    sparse_tensor = tf.SparseTensor(indices, values, dense_shape)

    reshape_tensor = tf.sparse_reshape(sparse_tensor, [4])
    reshape_indices = tf.reshape(indices, [4])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        reshape_tensor_indices_np, reshape_indices_np = sess.run([reshape_tensor.indices, reshape_indices])

    print(reshape_tensor_indices_np)
    print(reshape_indices_np)


def sparse_get_dims_test():
    indices = tf.constant([[1, 1], [0, 0]], dtype=tf.int64)
    values = tf.constant([1., 2.])
    dense_shape = tf.constant([2, 2], dtype=tf.int64)
    sparse_tensor = tf.SparseTensor(indices, values, dense_shape)
    print(sparse_tensor.get_shape().ndims)
    print(sparse_tensor.dense_shape)
    print('test')


def permute_gather_test():
    a = tf.constant([7, 8, 9, 10, 11, 12])
    b = [5, 4, 1, 2, 3, 0]

    c = tf.gather(a, b)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        c_np = sess.run(c)

        print(c_np)


def sparse_reduce_sum_nd_test():
    indices = tf.constant([[0, 0, 0],
                           [0, 1, 0],
                           [1, 1, 1]], dtype=tf.int64)
    values = tf.constant([1., 2., 3.])
    dense_shape = tf.constant([2, 2, 2], dtype=tf.int64)
    sparse_tensor = tf.SparseTensor(indices, values, dense_shape)

    reduce_sum_tensor = tf_ops.sparse_reduce_sum_nd(sparse_tensor, axis=[1], keep_dims=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        reduce_sum_tensor_np = sess.run(reduce_sum_tensor)

    np_tensor = np.zeros([2, 2, 2])
    np_tensor[0, 0, 0] = 1.
    np_tensor[0, 1, 0] = 2.
    np_tensor[1, 1, 1] = 3.
    np_sum_tensor = np.sum(np_tensor, axis=1, keepdims=True)

    condition = np.all(np.equal(reduce_sum_tensor_np, np_sum_tensor))
    test_out(condition, 'sparse_reduce_sum_nd_test')


if __name__ == '__main__':
    #extract_image_patches_nd_test()
    #expand_dims_nd_v_reshape_speed_test()
    #get_reverse_conv_2d_mask_test()
    #sparse_where_test()
    #conv_in_size_test()
    #conv_out_size_test()
    #get_full_indices_test()
    #repeat_test()
    #tf_zero_div_test()
    #gather_nd_test()
    #sparse_values_op_test()
    #indices_reshape_test()
    #sparse_get_dims_test()
    #permute_gather_test()
    sparse_reduce_sum_nd_test()
