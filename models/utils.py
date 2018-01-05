import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import math
import time


# TODO - make sure all functions have been fully tested
# TODO - sort out interchangeable use of 'size' and 'shape'; change to 'shape'
# Define eps - small constant for safe division/log
div_eps = 1e-8
log_eps = 1e-8


def safe_divide(x, y, name=None):
    if name is None:
        scope_name = 'safe_divide'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Want to clamp any values of y that are less than div_eps to div_eps
        y_eps = tf.where(tf.greater(tf.abs(y), div_eps), y, tf.sign(y) * div_eps * tf.ones_like(y))

        z = tf.divide(x, y_eps, name=name)
        return z


def safe_log(x, name=None):
    if name is None:
        scope_name = 'safe_log'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Want to clam any values of x less than log_eps to log_eps
        x_eps = tf.maximum(x, log_eps)
        y = tf.log(x_eps)
        return y


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


def get_shape_list(input_tensor):
    """
    Get the shape (as a list) of a tensor with dynamic batch dimension first
    :param input_tensor:
    :return: shape
    """
    shape = [tf.shape(input_tensor)[0], *input_tensor.get_shape().as_list()[1:]]  # TODO - use this function everywhere below where possible
    return shape


def repeat(input_tensor, repeats, axis=0, name=None):
    """
    TensorFlow version of the numpy.repeat function
    :param input_tensor: TensorFlow tensor
    :param repeats: Number of times to repeat each element
    :param axis: axis along which to repeat
    :param name: name of the op
    :return: repeat_tensor:
    """
    if name is None:
        scope_name = 'repeat'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        batch_size = tf.shape(input_tensor)[0]
        input_shape = [batch_size, *input_tensor.get_shape().as_list()[1:]]

        # First need to transpose so axis is the final dim
        perm = list(range(len(input_shape)))
        perm[-1], perm[axis] = perm[axis], perm[-1]
        input_tensor = tf.transpose(input_tensor, perm)

        # Then need to add an extra dimension to tile along
        input_tensor = tf.expand_dims(input_tensor, axis=[-1])

        # Tile along newly created dimension
        input_tensor = tf.tile(input_tensor, [*[1 for _ in range(input_tensor.get_shape().ndims - 1)], repeats])

        # Reshape and remove extra dimension
        repeat_shape = input_shape
        repeat_shape[-1], repeat_shape[axis] = repeat_shape[axis], repeat_shape[-1]
        repeat_shape[-1] *= repeats
        repeat_tensor = tf.reshape(input_tensor, repeat_shape)

        # Transpose back to original orientation
        repeat_tensor = tf.transpose(repeat_tensor, perm)

        return repeat_tensor


def sparse_reduce_sum_nd(sparse_tensor, axis=None, keep_dims=False, name=None):
    """
    Version of sparse_reduce_sum that works with tensors >5D. This assumes that dim 0 is batch size and axis is 3
    (or 4?)D max.
    :param sparse_tensor:
    :param axis:
    :param keep_dims:
    :param name:
    :return: full_tensor:
    """
    if name is None:
        scope_name = 'sparse_reduce_sum_nd'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Make sure axis is list
        axis = list(axis)

        # Need to sparse reshape to get around 5D limitation of sparse reduce sum
        # Transpose so that axis is/are first + (leaving batch as zeroth) dims of tensor
        perm = list(range(sparse_tensor.get_shape().ndims))
        full_shape = sparse_tensor.dense_shape
        new_axis = axis
        for i, ax in enumerate(axis):
            # i + 1 index here as we want to leave batch as dim 0
            perm[i + 1], perm[ax] = perm[ax], perm[i + 1]
            new_axis[i] = i + 1

        # Transpose sparse tensor so that non batch or axis dims are last
        sparse_tensor = tf.sparse_transpose(sparse_tensor, perm)

        full_perm_shape = tf.gather(full_shape, perm)
        full_perm_shape_axis = tf.gather(full_shape, perm[:len(axis) + 1])
        if len(axis) + 1 < len(perm):
            full_perm_shape_condensed = tf.cast(tf.reduce_prod(tf.gather(full_shape, perm[len(axis) + 1:]), keep_dims=True), tf.int64)
            condensed_shape = tf.concat([full_perm_shape_axis, full_perm_shape_condensed], axis=0)
        else:
            condensed_shape = full_perm_shape_axis

        # Reshape to combine other dims
        sparse_tensor = tf.sparse_reshape(sparse_tensor, condensed_shape)

        # Do sparse reduce sum on resulting sparse tensor to get the required dense tensor
        full_tensor = tf.sparse_reduce_sum(sparse_tensor, axis=new_axis, keep_dims=True)

        # Reshape to regain condensed dimensions
        new_shape = tf.scatter_update(tf.Variable(full_perm_shape), tf.constant(new_axis, dtype=tf.int32),
                                      tf.ones([len(new_axis)], dtype=full_perm_shape.dtype))

        full_tensor = tf.reshape(full_tensor, new_shape)

        # Transpose back to original orientation
        full_tensor = tf.transpose(full_tensor, perm)

        # Squeeze over axis if keep dims is False
        if not keep_dims:
            full_tensor = tf.squeeze(full_tensor, axis)

        return full_tensor


def gather_nd_nd(params, indices, name=None):
    """
    Version of gather_nd that gets around the 5D limitation by reshaping before and after performing standard gather_nd.
    Need to double check this works for gathering
    :param params:
    :param indices:
    :param name:
    :return:
    """
    tf.gather_nd()
    return out_tensor


def reduce_sumsparse(input_tensor, strides, rates=(1, 1, 1, 1), padding='SAME', in_size=None,
                     axis=None, keep_dims=False, name=None):
    """
    First converts a tensor of image patches to a sparse tensor, where the image patches retain their original position
    in the image, before doing a sparse reduce sum and converting back to a full tensor
    :param input_tensor: Tensor of image patches, indexed by separate dimension(s)
    :param strides:
    :param rates:
    :param padding:
    :param in_size:
    :param axis:
    :param keep_dims:
    :param name:
    :return:
    """
    if name is None:
        scope_name = 'reduce_sumsparse'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Convert input tensor to sparse
        sparse_tensor = patches_to_sparse(input_tensor, strides, rates, padding, in_size)

        # Do sparse reduce sum
        full_tensor = sparse_reduce_sum_nd(sparse_tensor, axis=axis, keep_dims=keep_dims)

        return full_tensor


def reduce_logsumexpsparse(input_tensor, strides, rates=(1, 1, 1, 1), padding='SAME', in_size=None,
                           axis=None, keep_dims=False, name=None):
    """
    First converts a tensor of image patches to a sparse tensor before performing sparse_reduce_logsumexp and returning
    the resulting full tensor
    :param input_tensor:
    :param strides:
    :param rates:
    :param padding:
    :param in_size:
    :param axis:
    :param keep_dims:
    :param name:
    :return:
    """
    if name is None:
        scope_name = 'reduce_logsumexpsparse'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Get indices for conversion from full to dense
        # TODO - complete this
        # Convert to full tensor containing zeros
        sparse_tensor = patches_to_full(input_tensor, strides, rates, padding, in_size)

        # Do sparse reduce logsumexp on resulting full tensor
        full_tensor = sparse_reduce_logsumexp(sparse_tensor, axis=axis, keep_dims=keep_dims)

        return full_tensor


def sparse_reduce_logsumexp(sparse_tensor, axis=None, keep_dims=False, name=None):
    """
    Sparse version of tf.reduce_logsumexp that take a SparseTensor as input and returns a dense tensor. The design of
    this function takes advantage some properties that may be specific to this situation so care should be taken using
    this in other projects.
    :param sparse_tensor:
    :param axis:
    :param keep_dims:
    :param name:
    :return: full_tensor:
    """
    if name is None:
        scope_name = 'sparse_reduce_logsumexp'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        raw_max = tf.sparse_reduce_max(sparse_tensor, axis=axis, keep_dims=keep_dims)

        conditional = tf.where(tf.is_finite(raw_max), raw_max, tf.zeros_like(raw_max))
        my_max = tf.stop_gradient(conditional)

        # Exp
        # Convert sparse_tensor to full so that we can subtract my_max
        sparse_full = tf.sparse_tensor_to_dense(sparse_tensor, validate_indices=False)

        # Subtract my_max
        exp_values = tf.exp(sparse_full, - my_max)

        # Extract only required values from exp_values (according to sparse_tensor.indices)
        exp_values = gather_nd_nd(exp_values, sparse_tensor.indices)

        # Convert back to sparse
        exp_sparse = tf.SparseTensor(sparse_tensor.indices, exp_values, sparse_tensor.dense_shape)

        # Sum
        sum_sparse = sparse_reduce_sum_nd(exp_sparse, axis=axis, keep_dims=True)

        # Log
        full_tensor = tf.log(sum_sparse) + my_max

        return full_tensor


def extract_image_patches_nd(input_tensor, ksizes, strides, rates=(1, 1, 1, 1), padding='SAME', return_sparse=False,
                             name=None):
    """
    Extension of tf.extract_image_patches to tensors of arbitrary dimensions
    :param input_tensor: Tensor with at least 4 dimensions - [batch_size, image_height, image_width, depth, ...]
    :param ksizes: Int, Tuple or List containing 4D kernel size - [1, kernel_rows, kernel_cols, 1]
    :param strides: Int, Tuple or List containing 4D strides - [1, stride_rows, stride_cols, 1]
    :param rates: Int, Tuple or List containing rates for dilated kernels - [1, rate_rows, rate_cols, 1]
    :param padding: 'SAME' or 'VALID' specifying the padding to use
    :param name: Name of the extract_image_patches op
    :return: patches: Tensor with shape [batch_size, kernel_rows, kernel_cols, depth, out_rows, out_cols, ...]
    """
    if name is None:
        scope_name = 'extract_image_patches_nd'
    else:
        scope_name = name
    ksizes, strides, rates = get_correct_ksizes_strides_rates(ksizes, strides, rates)
    with tf.variable_scope(scope_name):
        # Get shape of input_tensor
        input_shape = input_tensor.get_shape().as_list()
        batch_size = tf.shape(input_tensor)[0]

        # Reshape input_tensor to 4D for use with tf.extract_image_patches
        input_tensor = tf.reshape(input_tensor, shape=[batch_size, input_shape[1], input_shape[2], np.prod(input_shape[3:])])

        # Extract image patches
        patches = tf.extract_image_patches(input_tensor, ksizes, strides, rates, padding, name=name)
        # Reshape and transpose resulting tensor
        patches_shape = patches.get_shape().as_list()
        patches_shape = [batch_size, patches_shape[1], patches_shape[2], ksizes[1], ksizes[2], *input_shape[3:]]
        patches = tf.reshape(patches, patches_shape)

        permutation = [0, 3, 4, 5, 1, 2, *range(6, len(patches_shape))]
        patches = tf.transpose(patches, permutation)

        if return_sparse:
            patches = patches_to_sparse(patches, strides, rates, padding)

        return patches


def get_dense_indices(patch_shape, in_size, strides, rates, padding):
    """
    Get the dense indices of a kernel patch array as a numpy array
    :param patch_shape: list of ints [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2]
    :param in_size:
    :param strides:
    :param rates:
    :param padding
    :return: indices
    """
    batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2 = patch_shape

    in_rows, in_cols = in_size

    k_dash = [kernel_rows + (kernel_rows - 1) * (rates[1] - 1), kernel_cols + (kernel_cols - 1) * (rates[2] - 1)]
    if padding is 'VALID':
        p_rows = 0
        p_cols = 0
    else:
        # TODO - padding computation (and potentially patch placement below) needs to mirror tf.extract_image_patches, in which there are some subtleties; check here: https://github.com/RLovelett/eigen/blob/ebc657d1bc26aebd77ac9ecc817def4d92120b77/unsupported/Eigen/CXX11/src/Tensor/TensorImagePatch.h#L151 for original code and transcribe here
        p_rows = math.floor((k_dash[0] - 1) / 2)
        p_cols = math.floor((k_dash[1] - 1) / 2)

    indices = []
    # Construct for first batch element, out capsule, capsule dim 1 and capsule dim 2
    start = time.time()
    # TODO - can speed this up by taking in_capsules out of the loop and combining later
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
                        row = i_o * strides[1] + i_k * rates[1] - p_rows
                        if row < 0:
                            row = row + k_dash[0]
                        elif row > in_rows - 1:
                            row = row - k_dash[0]

                        col = j_o * strides[2] + j_k * rates[2] - p_cols
                        if col < 0:
                            col = col + k_dash[1]
                        elif col > in_cols - 1:
                            col = col - k_dash[1]

                        indices.append(np.array([row, col, c_i, i_o, j_o]))
    np_indices = np.stack(indices)

    # Convert indices to constant tensor with dtype tf.int64
    indices = tf.constant(np_indices, dtype=tf.int64)

    # Repeat computed indices over out capsules, capsule dim 1 and capsule dim 2
    indices = repeat(indices, out_capsules * capsule_dim1 * capsule_dim2, axis=0)

    # Get indices for out capsules, capsule dim 1 and capsule dim 2 and repeat/tile to correct size
    capsule_dim2_indices = tf.tile(tf.range(capsule_dim2), [out_capsules * capsule_dim1])
    capsule_dim1_indices = tf.tile(repeat(tf.range(capsule_dim1), capsule_dim2), [out_capsules])
    out_capsules_indices = repeat(tf.range(out_capsules), capsule_dim2 * capsule_dim1)

    # Concatenate the indices just computed and tile over the previously computed indices
    extra_indices = tf.transpose(tf.stack([out_capsules_indices, capsule_dim1_indices, capsule_dim2_indices], axis=0))
    extra_indices = tf.tile(extra_indices, [kernel_rows * kernel_cols * in_capsules * out_rows * out_cols, 1])

    # Concatenate the two sets of indices
    indices = tf.concat([indices, tf.cast(extra_indices, tf.int64)], axis=1)

    indices_per_batch = kernel_rows * kernel_cols * in_capsules * out_rows * out_cols * out_capsules * capsule_dim1 * capsule_dim2

    # Extend indices over batch - Should be done within the graph so we can use variable batch size
    batch_indices = tf.cast(tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, indices_per_batch]),
                                       [batch_size * indices_per_batch, 1]), dtype=tf.int64)
    indices = tf.concat([batch_indices, tf.tile(indices, [batch_size, 1])], axis=1)

    print("Reconstruction indices time: {}s".format(time.time() - start))

    return indices


def patches_to_sparse(input_tensor, strides, rates=(1, 1, 1, 1), padding='SAME', in_size=None, name=None):
    """
    Convert a dense tensor containing image patches to a sparse tensor for which the image patches retain their original
    indices
    :param input_tensor: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
    :param strides:
    :param rates:
    :param padding:
    :param in_size:
    :param name: Name for the op
    :return: sparse_patches: SparseTensor with shape [batch_size, im_rows, im_cols, ...]
    """
    if name is None:
        scope_name = 'patches_to_sparse'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Get required shapes
        batch_size = tf.shape(input_tensor)[0]
        shape_list = input_tensor.get_shape().as_list()
        kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2 = shape_list[1:]

        # Compute input shape
        if in_size is None:
            in_rows, in_cols = conv_in_size([out_rows, out_cols], [kernel_rows, kernel_cols], list(strides[1:3]),
                                            list(rates[1:3]), padding)
        else:
            in_rows, in_cols = in_size

        # Get dense indices
        indices = get_dense_indices([batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2],
                                    [in_rows, in_cols], strides, rates, padding)

        # Reshape input_tensor to 1D
        values = tf.reshape(input_tensor,
                            shape=[batch_size*kernel_rows*kernel_cols*in_capsules*out_rows*out_cols*out_capsules*capsule_dim1*capsule_dim2])

        # Create constant tensor containing dense shape
        # TODO - might be a better way to do this but this works
        dense_shape = tf.shape(input_tensor, out_type=tf.int64)
        dense_shape = tf.add(dense_shape,
                             tf.constant([0, in_rows - kernel_rows, in_cols - kernel_cols, 0, 0, 0, 0, 0, 0],
                                         dtype=tf.int64))

        # Create sparse tensor
        sparse_patches = tf.SparseTensor(indices, values, dense_shape)
        return sparse_patches


def patches_to_full(input_tensor, strides, rates=(1, 1, 1, 1), padding='SAME', in_size=None, name=None):
    """
    Converts a tensor of image patches to a full tensor in which the image patches are embedded within an array of zeros
    with the same shape as the original image
    :param input_tensor: Tensor containing image patches with shape [batch_size, kernel_rows, kernel_cols, in_capsules,
                         out_rows, out_cols, out_capsules, ?, ?]
    :param strides:
    :param rates:
    :param padding:
    :param in_size:
    :param name:
    :return:
    """
    if name is None:
        scope_name = 'patches_to_full'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # First convert to sparse
        sparse_patches = patches_to_sparse(input_tensor, strides, rates, padding, in_size=in_size)

        # Then convert sparse to dense
        dense_patches = tf.sparse_tensor_to_dense(sparse_patches, validate_indices=False)

        # This seems to lose the shape so reset shape
        full_shape = get_patches_full_shape(input_tensor, strides, rates, padding, in_size=in_size)
        dense_patches = tf.reshape(dense_patches, full_shape)

        return dense_patches


def full_to_patches(full_tensor, indices, dense_shape, name=None):
    """
    Converts a full tensor of patches (i.e. with patches in their original position with the rest of the image padded
    with zeros and each patch image indexed by separate dimension(s))
    Assumes dim 0 is batch size and dims 1 & 2 are the patch/kernel dims
    :param full_tensor: Full tensor containing patches padded to full size and indexed by separate dimension(s)
    :param indices: Indices of the patches within the full tensor
    :param dense_shape: Shape of the resulting dense patches tensor (not the same as dense_shape for sparse tensors)
    :param name: Name for the op
    :return: patches_tensor: Tensor containing only patches, not padded to full size
    """
    if name is None:
        scope_name = 'full_to_patches'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # TODO - This is a hacky way to do this, only need to because of the 5D limit of gather_nd. If this limit is raised to 9 or higher then we can remove everything here up to the 'Extract patches' comment
        values = tf.zeros(dense_shape)
        full_shape = get_shape_list(full_tensor)
        sparse_tensor = tf.SparseTensor(indices, tf.reshape(values, [-1]), tf.shape(full_tensor, out_type=tf.int64))

        # Get condensed shape
        full_condensed_shape = [*full_shape[0:3], np.prod(full_shape[3:])]

        # Reshape sparse tensor so that we can get the reshaped indices
        reshape_sparse_tensor = tf.sparse_reshape(sparse_tensor, full_condensed_shape)

        # Get reshaped indices
        indices = reshape_sparse_tensor.indices

        # Reshape full tensor
        full_tensor = tf.reshape(full_tensor, full_condensed_shape)

        # Extract patches
        patches_tensor = tf.gather_nd(full_tensor, indices)

        # Reshape
        patches_tensor = tf.reshape(patches_tensor, dense_shape)

        return patches_tensor


def get_patches_full_shape(patches_tensor, strides, rates=(1, 1, 1, 1), padding='SAME', in_size=None, name=None):
    """
    Get the eqivalent full shape for a tensor containing image patches
    :param patches_tensor:
    :param strides:
    :param rates:
    :param padding:
    :param in_size:
    :param name:
    :return:
    """
    if name is None:
        scope_name = 'get_patches_full_shape'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        batch_size = tf.shape(patches_tensor)[0]
        shape = patches_tensor.get_shape().as_list()
        kernel_size = shape[1:3]
        in_capsules = shape[3]
        out_size = shape[4:6]
        out_capsules = shape[6]
        remaining_dims = shape[7:]

        if in_size is None:
            in_size = conv_in_size(out_size, kernel_size, strides[1:3], rates[1:3], padding)

        full_shape = [batch_size, *in_size, in_capsules, *out_size, out_capsules, *remaining_dims]

        return full_shape


def expand_dims_nd(input_tensor, axis=None, name=None):
    """
    Extension of tf.expand_dims to multiple dimensions so that more than one dimension can be added at a time in the
    same way that more than one dimension can be squeezed at a time using tf.squeeze. This is very marginally faster
    than using tf.reshape
    :param input_tensor: Tensor to be expanded
    :param axis: None, Int, Tuple or List specifying which axes to be expanded using the same logic as tf.expand_dims
                 NOTE - be careful of the order of the dimensions here as this may change the results
    :param name: Name for the op/ops
    :return:
    """
    axis = list(axis)
    if len(axis) == 1:
        input_tensor = tf.expand_dims(input_tensor, axis=axis[0], name=name)
    else:
        with tf.variable_scope('expand_dims_nd'):
            for dim in axis:
                input_tensor = tf.expand_dims(input_tensor, axis=dim, name=name)

    return input_tensor


def convcaps_affine_transform(in_pose, in_activation, out_capsules, kernel_size, strides, rates=(1, 1, 1, 1),
                              padding='SAME'):
    """
    Creates the TensorFlow graph for the convolutional affine transform performed prior to routing in a convolutional
    capsule layer. This also extracts image patches from and reshapes in_activation in order to keep the code and graph
    clean.
    :param in_pose: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, in_rows, in_cols, in_capsules]
    :param out_capsules: Int specifying the number of output capsules
    :param kernel_size: Int, Tuple or List specifying the size of the convolution kernel (assuming square kernel if int)
    :param strides: Int, Tuple or List specifying the strides for the convolution (assuming equal over dimensions if int)
    :param rates:
    :param padding: 'valid' or 'same' specifying padding to use in the same way as tf.nn.conv2d
    :return: vote: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
    """
    # TODO - this implementation is quite memory inefficient due to tiling, extracting image patches and converting to sparse then full, should look into using depthwise separable convolution (although this may make things tricky in the e-step sum), otherwise there should really be a new tensorflow op implemented for this
    # Sort out different possible kernel_size and strides formats
    ksizes, strides, rates = get_correct_ksizes_strides_rates(kernel_size, strides, rates)
    with tf.variable_scope('convcaps_affine_transform'):
        # Get required shape values
        batch_size = tf.shape(in_pose)[0]
        shape_list = in_pose.get_shape().as_list()
        in_rows = shape_list[1]
        in_cols = shape_list[2]
        in_capsules = shape_list[3]
        pose_size = shape_list[4]

        # Compute output im grid size
        out_size = conv_out_size([in_rows, in_cols], ksizes[1:3], strides[1:3], rates[1:3], padding)

        # Create convolutional matmul kernel and tile over batch and out_size (as we need the same kernel to be
        # multiplied by each patch of conv_pose for each element in the batch)
        kernel = tf.Variable(tf.truncated_normal([1, *ksizes[1:3], in_capsules, 1, 1, out_capsules, pose_size, pose_size]), name='kernel')
        tf.summary.histogram('kernel', kernel)
        kernel = tf.tile(kernel, [batch_size, 1, 1, 1, *out_size, 1, 1, 1])

        # Re-organise in_pose so performing matmul with kernel computes the required convolutional affine transform
        conv_pose = tf.reshape(in_pose, [batch_size, in_rows, in_cols, in_capsules, 1, pose_size, pose_size])

        # Get patches from conv_pose and concatenate over out_size in correct dimensions so that new shape is:
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], 1, pose_size, pose_size]
        conv_pose = extract_image_patches_nd(conv_pose, ksizes, strides, rates=rates, padding=padding,
                                             name='extract_pose_patches')

        # Tile over out_capsules: need to be multiplying the same input tensor by different weights for each out capsule
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], out_capsules, pose_size, pose_size]
        conv_pose = tf.tile(conv_pose, [1, 1, 1, 1, 1, 1, out_capsules, 1, 1])

        vote = tf.matmul(kernel, conv_pose)

        # Get patches from in_activation and expand dims
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1]]
        activation = extract_image_patches_nd(in_activation, ksizes, strides, rates=rates, padding=padding,
                                              name='extract_activation_patches')
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], 1, 1, 1]
        activation = tf.reshape(activation, [batch_size, *ksizes[1:3], in_capsules, *out_size, 1, 1, 1])
        #tf.summary.histogram('convcaps_affine_activation', activation)

        # Convert vote and activation patches to full tensors
        #vote = patches_to_full(vote, strides, rates, padding, in_size=[in_rows, in_cols])
        #activation = patches_to_full(activation, strides, rates, padding, in_size=[in_rows, in_cols])

        return vote, activation


def caps_affine_transform(in_pose, in_activation, out_capsules, coord_addition=True):
    """
    Creates the TensorFlow graph for the affine transform performed prior to routing in a capsule layer. This also
    reshapes in_activation in order to keep the code and graph clean.
    :param in_pose: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, in_rows, in_cols, in_capsules]
    :param out_capsules: Int, the number of output capsules
    :param coord_addition: Bool, whether to do coordinate addition
    :return: vote: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, 1, 1, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, 1, 1, 1, 1, 1]
    """
    with tf.variable_scope('caps_affine_transform'):
        # Get required shape values
        batch_size = tf.shape(in_pose)[0]
        shape_list = in_pose.get_shape().as_list()
        in_rows = shape_list[1]
        in_cols = shape_list[2]
        in_capsules = shape_list[3]
        pose_size = shape_list[4]

        # Create matmul weights and tile over batch, in_rows and in_columns (as we need the same weights to be
        # multiplied by each batch element and because we need to share transformation matrices over the whole image)
        weights = tf.Variable(tf.truncated_normal([1, 1, 1, in_capsules, 1, 1, out_capsules, pose_size, pose_size]), name='weights')
        tf.summary.histogram('weights', weights)
        weights = tf.tile(weights, [batch_size, in_rows, in_cols, 1, 1, 1, 1, 1, 1])

        # Re-organise in_pose so performing matmul with kernel computes the required convolutional affine transform
        pose = tf.reshape(in_pose, [batch_size, in_rows, in_cols, in_capsules, 1, 1, 1, pose_size, pose_size])

        # Tile over out_capsules: need to be multiplying the same input tensor by different weights for each out capsule
        # [batch_size, in_rows, in_cols, in_capsules, 1, 1, out_capsules, pose_size, pose_size]
        pose = tf.tile(pose, [1, 1, 1, 1, 1, 1, out_capsules, 1, 1])

        vote = tf.matmul(weights, pose)

        # Do coordinate addition
        if coord_addition:
            vote = coordinate_addition(vote)

        # Expand dims of activation
        # [batch_size, in_rows, in_cols, in_capsules, 1, 1, 1, 1, 1]
        activation = expand_dims_nd(in_activation, [4, 5, 6, 7, 8])

        return vote, activation


def coordinate_addition(vote):
    """
    Creates the TensorFlow graph to do coordinate addition as described in the paper for vote tensor
    :param vote: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, 1, 1, out_capsules, pose_size, pose_size]
    :return: vote: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, 1, 1, out_capsules, pose_size, pose_size]
                   for which the scaled coordinates of each capsule has been added to the first two dimensions of the vote
                   (row -> vote[:, :, :, :, :, :, :, 0, 0] and col -> vote[:, :, :, :, :, :, :, 0, 1])
    """
    with tf.variable_scope('coordinate_addition'):
        in_rows, in_cols = vote.get_shape().as_list()[1:3]

        # Get grids of size [in_rows, in_cols] containing the scaled row and column coordinates
        col_coord, row_coord = tf.meshgrid(list(np.arange(in_rows) / (in_rows - 1)),
                                           list(np.arange(in_cols) / (in_cols - 1)))

        # Expand dimensions and concatenate so they can be added to vote
        # [1, in_rows, in_cols, 1, 1, 1, 1, 1, 1]
        row_coord = expand_dims_nd(row_coord, [0, 3, 4, 5, 6, 7, 8])
        col_coord = expand_dims_nd(col_coord, [0, 3, 4, 5, 6, 7, 8])
        # [1, in_rows, in_cols, 1, 1, 1, 1, 1, 2]
        coords = tf.concat([row_coord, col_coord], axis=-1)
        full_coords = tf.concat([coords, tf.zeros_like(coords)], axis=-1)

        # Add coordinates to vote
        vote += full_coords

        return vote


def m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp, conv=False):
    """
    Creates the TensorFlow graph for the M-Step of the EM Routing algorithm
    :param r: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1].
              The assignment probabilities for each vote (aka data point) to each Gaussian (for which the data point is
              within it's receptive field if we are doing conv -> conv routing) with mean described by pose of next
              layer.
    :param in_activation: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, 1, 1, 1]
    :param in_vote: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param beta_v: tf.Variable
    :param beta_a: tf.Variable
    :param inverse_temp: Scalar
    :return: mean: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
                   The re-computed means of the Gaussians (output poses of the layer)
             variance: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
                       The re-computed variances of each dimension of each Gaussian (i.e. diagonal of covariance matrix)
             activation: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
                         The updated activations for each output capsule
    """
    with tf.variable_scope('m_step'):
        # Update r (assignment probabilities)
        # In this step for each higher-level capsule, c, we multiply the activation, a, with the assignment probs/
        # responsibilities, r, to get the adjusted assignment probs, r'. We need to consider only capsules from the
        # previous layer that are within the receptive field of the capsules in the current layer
        rp = tf.multiply(r, in_activation, name='r_update_mul')

        # Update means (out_poses)
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        rp_reduce_sum = tf.reduce_sum(rp, axis=[1, 2, 3], keep_dims=True)
        
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
        mean = safe_divide(tf.reduce_sum(tf.multiply(rp, in_vote, name='mean_mul'), axis=[1, 2, 3], keep_dims=True),
                           rp_reduce_sum)

        # Update variances (same shape as mean)
        diff_vote_mean = tf.subtract(in_vote, mean, name='vote_mean_sub')
        variance = safe_divide(tf.reduce_sum(tf.multiply(rp, tf.square(diff_vote_mean, name='diff_vote_mean_square'),
                                                         name='variance_mul'), axis=[1, 2, 3], keep_dims=True),
                               rp_reduce_sum)

        # Compute cost (same shape as mean)
        cost_h = tf.multiply(tf.add(beta_v, safe_log(tf.sqrt(variance), name='log_stdd'), name='add_beta_log_stdd'),
                             rp_reduce_sum, name='cost_h_mul')

        # Compute new activations
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        activation = tf.nn.sigmoid(inverse_temp*(tf.subtract(beta_a, tf.reduce_sum(cost_h, axis=[-2, -1],
                                                                                   keep_dims=True))))

        tf.summary.histogram('rp', rp)
        tf.summary.histogram('rp_reduce_sum', rp_reduce_sum)
        tf.summary.histogram('cost_h', cost_h)
        tf.summary.histogram('mean', mean)
        tf.summary.histogram('variance', variance)
        tf.summary.histogram('activation', activation)
        if not conv:
            tf.summary.image('activation', tf.squeeze(activation, axis=[1, 2, 3, 4, 8]), max_outputs=1)

        return mean, variance, activation


def e_step(mean, variance, activation, in_vote, logspace=True, **sparse_args):
    """
    Creates the TensorFlow graph for the E-Step of the EM Routing algorithm
    :param mean: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param variance: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param activation: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
    :param in_vote: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param logspace: Bool, whether to do this step in log space or not (as written in the paper)
    :param sparse_args:
    :return: r: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
    """
    with tf.variable_scope('e_step'):
        if logspace:
            # Compute log(P): the log probability of each in_vote (data point)
            a = safe_log(2*math.pi*variance)  # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
            b = safe_divide(tf.square(tf.subtract(in_vote, mean, name='b_sub')), variance)  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]

            log_p_pre_sum = tf.add(a, b, name='log_p_sub')  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]

            log_p = -0.5 * tf.reduce_sum(log_p_pre_sum, axis=[-2, -1], keep_dims=True)  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]

            # Clip log_p to be <= 0 (i.e. 0 < p < 1)
            log_p = tf.where(tf.less(log_p, 0.), log_p, tf.zeros_like(log_p))

            # Compute updated r (assignment probability/responsibility)
            # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
            log_p_activation = safe_log(activation) + log_p

            if sparse_args['sparse']:
                log_p_activation_sum = tf.reduce_logsumexp(log_p_activation, axis=[4, 5, 6], keep_dims=True)

                # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
                r = tf.exp(log_p_activation - log_p_activation_sum)
            else:
                log_p_activation_sum = tf.reduce_logsumexp(log_p_activation, axis=[4, 5, 6], keep_dims=True)

                # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
                r = tf.exp(log_p_activation - log_p_activation_sum)

            tf.summary.histogram('log_p', log_p)
            tf.summary.histogram('log_p_activation', log_p_activation)
        else:
            a = tf.sqrt(tf.reduce_prod(2*math.pi*variance, axis=[-2, -1], keep_dims=True))
            b = 0.5 * tf.reduce_sum(safe_divide(tf.square(in_vote - mean), variance), axis=[-2, -1], keep_dims=True)

            p = safe_divide(tf.exp(-b), a)

            # Clip p to between 0 and 1 if safe divide has caused this not to be the case
            p = tf.where(tf.greater(p, 1.), tf.ones_like(p), p)
            p = tf.where(tf.less(p, 0.), tf.zeros_like(p), p)

            activation_p = activation * p

            if sparse_args['sparse']:
                activation_p_reduce_sum = reduce_sumsparse(activation_p, sparse_args['strides'], sparse_args['rates'],
                                                           sparse_args['padding'], sparse_args['in_size'],
                                                           axis=[4, 5, 6], keep_dims=True)

                # Get patch size and dense shape for conversion back to dense patches later
                dense_shape = get_shape_list(activation_p)

                # Convert activation_p to full for compatibility with activation_p_reduce_sum
                activation_p = patches_to_full(activation_p, sparse_args['strides'], sparse_args['rates'],
                                               sparse_args['padding'], sparse_args['in_size'])

                # Compute new r
                r = safe_divide(activation_p, activation_p_reduce_sum)

                # Convert r back to patches (since activation_p is patches this is valid)
                indices = get_dense_indices(dense_shape, sparse_args['in_size'], sparse_args['strides'],
                                            sparse_args['rates'], sparse_args['padding'])  # TODO - might be more efficient to  pass the indices out from reduce_sumsparse as they are already in there
                r = full_to_patches(r, indices, dense_shape)

            else:
                activation_p_reduce_sum = tf.reduce_sum(activation_p, axis=[4, 5, 6], keep_dims=True)

                # Compute new r
                r = safe_divide(activation_p, activation_p_reduce_sum)

            tf.summary.histogram('p', p)
            tf.summary.histogram('activation_p', activation_p)

        tf.summary.histogram('a', a)
        tf.summary.histogram('b', b)
        tf.summary.histogram('r', r)

        return r


def em_routing(in_vote, in_activation, n_routing_iterations=3, init_inverse_temp=0.1, final_inverse_temp=0.9,
               ksizes=None, strides=(1, 1, 1, 1), rates=(1, 1, 1, 1), padding='SAME', in_size=None, conv=False):
    """
    Creates the TensorFlow graph for the EM routing between capsules. Takes in and outputs 2D grids of capsules so that 
    it will work between convolutional layers, but it may be used for routing between non convolutional layers by using
    a 2D grid where the size of one dimension is 1 and taking kernel_rows/cols to be input_rows/cols
    :param in_vote: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size] NOW SHOULD BE FULL PATCH TENSOR
    :param in_activation: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, 1, 1, 1]
    :param n_routing_iterations: Int number of iterations to run the EM algorithm for the routing
    :param init_inverse_temp: Initial value for the inverse temperature parameter used in the M-step
    :param final_inverse_temp: Final value (on last iteration) of the inverse temperature parameter for the M-step
    :param ksizes:
    :param strides: Strides used to extract patches for in_vote and in_activation
    :param rates: Rates used to extract patches for in_vote and in_activation
    :param padding: Padding used to extract patches for in_vote and in_activation
    :param in_size: Shape of the input image from which patches have been extracted. Can be inferred if None but ambiguous in some cases so best if explicit
    :param conv: bool whether this is routing between convolutional capsules or not
    :return: pose: Tensor with shape [batch_size, out_rows, out_cols, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, out_rows, out_cols, out_capsules]
    """
    with tf.variable_scope('em_routing'):
        # Get required shapes
        batch_size = tf.shape(in_vote)[0]
        shape_list = in_vote.get_shape().as_list()
        if conv:
            kernel_size = ksizes[1:3]
        else:
            kernel_size = [shape_list[1], shape_list[2]]
        in_capsules = shape_list[3]
        out_size = [shape_list[4], shape_list[5]]
        out_capsules = shape_list[6]

        # Create R constant and initialise as uniform probability (stores the probability of each vote under each
        # Gaussian with mean corresponding to output pose)
        r = tf.ones([batch_size, *kernel_size, in_capsules, *out_size, out_capsules, 1, 1], name="R") / (np.prod(out_size) * out_capsules)

        # Create beta parameters
        beta_v = tf.Variable(tf.ones([1, 1, 1, 1, 1, 1, out_capsules, 1, 1]), name="beta_v")
        beta_a = tf.Variable(-0.5 * tf.ones([1, 1, 1, 1, 1, 1, out_capsules, 1, 1]), name="beta_a")
        tf.summary.histogram('beta_v', beta_v)
        tf.summary.histogram('beta_a', beta_a)

        # Initialise inverse temperature parameter and compute how much to increment by for each routing iteration
        inverse_temp = init_inverse_temp
        if n_routing_iterations > 1:
            inverse_temp_increment = (final_inverse_temp - init_inverse_temp) / (n_routing_iterations - 1)
        else:
            inverse_temp_increment = 0  # Cant increment anyway in this case

        #inverse_temp_tf = tf.abs(tf.Variable(init_inverse_temp))
        #inverse_temp_increment_tf = tf.abs(tf.Variable(inverse_temp_increment))
        #tf.summary.scalar('inverse_temp', inverse_temp_tf)
        #tf.summary.scalar('inverse_temp_increment', inverse_temp_increment_tf)

        """
        If we are doing routing between convolutional capsules we need to send the correct parameters to the e-step
        so that we can convert the a*p tensor to a sparse tensor and do a sparse reduce sum. Otherwise we would
        be computing the wrong sum since all the tensors of conv patches do not keep the patches in their original 
        position in the image but we need to take their original position into account when doing the sum. E.g.
        for a 1D image and 1D patches for a single input and output 1D capsule:
        image --------------------------- [[1, 2, 3, 4]]
        patches (of size 2)-------------- [[1, 2],
                                           [2, 3],
                                           [3, 4]]
        patches in original position ---- [[1, 2, x, x],
                                           [x, 2, 3, x],
                                           [x, x, 3, 4]] 
        incorrect sum over output ------- [[6, 9]]
        correct sum over output --------- [[1, 4, 6, 4]]
        sum over input works either way:- [[3],
                                           [5],
                                           [7]]
        """
        if conv:
            strides = get_correct_conv_param(strides)
            rates = get_correct_conv_param(rates)
            if in_size is None:
                in_size = conv_in_size(out_size, kernel_size, strides[1:3], rates[1:3], padding)
            sparse_args = {'sparse': True,
                           'strides': strides,
                           'rates': rates,
                           'padding': padding,
                           'in_size': in_size}
            #r = patches_to_full(r, strides, rates, padding, in_size=in_activation.get_shape().as_list()[1:3])
        else:
            sparse_args = {'sparse': False}

        # TODO - remove this summary once debugged?
        tf.summary.image('in_activation_00', tf.reshape(tf.reshape(in_activation[0],
                                                                   shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 0, 0],
                                                        shape=[1, *in_activation.get_shape().as_list()[1:3], 1]), max_outputs=1)
        if conv:
            tf.summary.image('in_activation_01', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 0, 1],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]), max_outputs=1)
            tf.summary.image('in_activation_10', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 1, 0],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]), max_outputs=1)
            tf.summary.image('in_activation_02', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 0, 2],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]), max_outputs=1)
            tf.summary.image('in_activation_20', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 2, 0],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]), max_outputs=1)
            tf.summary.image('in_activation_22', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 2, 2],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]), max_outputs=1)


        # TODO - should we definitely be stopping the gradient of the beta_v, beta_a, vote and/or activations here?
        in_vote_stopped = tf.stop_gradient(in_vote)
        in_activation_stopped = tf.stop_gradient(in_activation)
        beta_v_stopped = tf.stop_gradient(beta_v)
        beta_a_stopped = tf.stop_gradient(beta_a)

        # Do routing iterations
        for routing_iteration in range(n_routing_iterations - 1):
            with tf.variable_scope("routing_iteration_{}".format(routing_iteration + 1)):
                # Do M-Step to get Gaussian means and standard deviations and update activations
                mean, variance, activation = m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp, conv)

                # Do E-Step to update R
                r = e_step(mean, variance, activation, in_vote, **sparse_args)

                # Update inverse temp
                inverse_temp += inverse_temp_increment

        with tf.variable_scope("routing_iteration_{}".format(n_routing_iterations)):
            # Do M-Step to get Gaussian means and update activations
            mean, _, activation = m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp, conv)

        # Get rid of redundant dimensions
        pose = tf.squeeze(mean, [1, 2, 3])
        activation = tf.squeeze(activation, [1, 2, 3, 7, 8])

        return pose, activation


def primarycaps_layer(input_tensor, out_capsules, pose_size):
    """
    Creates the TensorFlow graph for the PrimaryCaps layer described in 'Matrix Capsules with EM Routing'c
    :param input_tensor: Tensor with shape [batch_size, height, width, n_filters] (batch_size, 12?, 12?, 32) in paper
    :param out_capsules: Number of capsules (for each pixel)
    :param pose_size: Size of the capsule pose matrices (i.e. pose matrix will be pose_size x pose_size)
    :return: pose: Tensor with shape [batch_size, in_rows, in_cols, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, height, width, out_capsules]
    """
    with tf.variable_scope('PrimaryCaps'):
        # Get required shape values
        batch_size = tf.shape(input_tensor)[0]
        shape_list = input_tensor.get_shape().as_list()
        in_rows = shape_list[1]
        in_cols = shape_list[2]
        in_channels = shape_list[3]

        # Affine transform to create capsule pose matrices and activations
        # Create weights and tile them over batch in preparation for matmul op as we need to use the same weights for
        # each element in the batch
        weights = tf.Variable(tf.truncated_normal([1, 1, 1, out_capsules, in_channels, (pose_size ** 2 + 1)]), name='weights')
        weights = tf.tile(weights, [batch_size, in_rows, in_cols, 1, 1, 1])

        # Expand input tensor for matmul op and tile input over out_capsules for matmul op as we need to multiply the
        # input by separate weights for each output capsule
        input_tensor = tf.reshape(input_tensor, [batch_size, in_rows, in_cols, 1, 1, in_channels])  # [batch_size, in_rows, in_cols, 1, 1, in_channels]
        input_tensor = tf.tile(input_tensor, [1, 1, 1, out_capsules, 1, 1])  # [batch_size, in_rows, in_cols, out_capsules, 1, in_channels]

        # Do matmul to get flattened primarycaps pose matrices and then reshape so pose matrices are square
        pose_activation = tf.matmul(input_tensor, weights)  # [batch_size, in_rows, in_cols, out_capsules, 1, (pose_size**2 + 1)]

        # Get pose
        pose = pose_activation[:, :, :, :, :, :pose_size ** 2]  # [batch_size, in_rows, in_cols, out_capsules, 1, pose_size**2]
        pose = tf.reshape(pose, [batch_size, in_rows, in_cols, out_capsules, pose_size, pose_size])

        # Get activation
        activation = pose_activation[:, :, :, :, :, pose_size ** 2]  # [batch_size, in_rows, in_cols, out_capsules, 1, 1]
        activation = tf.reshape(activation, [batch_size, in_rows, in_cols, out_capsules])  # [batch_size, in_rows, in_cols, out_capsules]
        activation = tf.nn.sigmoid(activation)

        return pose, activation


def convcaps_layer(in_pose, in_activation, out_capsules, kernel_size, strides=1, rates=(1, 1, 1, 1), padding='SAME',
                   n_routing_iterations=3, init_inverse_temp=0.1, final_inverse_temp=0.9):
    """
    Creates the TensorFlow graph for a convolutional capsule layer as specified in 'Matrix Capsules with EM Routing'.
    In this layer we first perform the convolutional affine transform between input poses to get votes for the routing.
    We then use these votes along with the input activations with the EM routing algorithm to compute the output pose
    and activations.
    :param in_pose: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, in_rows, in_cols, in_capsules]
    :param out_capsules: Int number of output capsules in the layer
    :param kernel_size: Int, Tuple or List specifying the size of the convolution kernel (assuming square kernel if int)
    :param strides: Int, Tuple or List specifying the strides for the convolution (assuming equal over dimensions if int)
    :param rates:
    :param padding: 'valid' or 'same' specifying padding to use in the same way as tf.nn.conv2d
    :param n_routing_iterations: Number of iterations to use for the EM dynamic routing procedure
    :param init_inverse_temp: Scalar initial value for the inverse temperature parameter used for EM routing
    :param final_inverse_temp: Scalar final value for the inverse temperature parameter used for EM routing
    :return: pose: Tensor with shape [batch_size, out_rows, out_cols, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, out_rows, out_cols, out_capsules]
    """
    ksizes, strides, rates = get_correct_ksizes_strides_rates(kernel_size, strides, rates)
    with tf.variable_scope('ConvCaps'):
        in_size = in_pose.get_shape().as_list()[1:3]
        # Pose convolutional affine transform
        in_vote, in_activation = convcaps_affine_transform(in_pose, in_activation, out_capsules, ksizes, strides,
                                                           rates, padding)

        # EM Routing
        pose, activation = em_routing(in_vote, in_activation, n_routing_iterations, init_inverse_temp,
                                      final_inverse_temp, ksizes, strides, rates, padding, in_size, conv=True)

        return pose, activation


def classcaps_layer(in_pose, in_activation, n_classes, n_routing_iterations=3,
                    init_inverse_temp=0.1, final_inverse_temp=0.9):
    """
    Creates the TensorFlow graph for the class capsules layer
    :param in_pose: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, in_rows, in_cols, in_capsules]
    :param n_classes: Int, the number of classes (the number out output capsules)
    :param n_routing_iterations: Number of iterations to use for the EM dynamic routing procedure
    :param init_inverse_temp: Scalar initial value for the inverse temperature parameter used for EM routing
    :param final_inverse_temp: Scalar final value for the inverse temperature parameter used for EM routing
    :return:
    """
    with tf.variable_scope('ClassCaps'):
        # Pose affine transform
        # in_vote: Tensor with shape[batch_size, in_rows, in_cols, in_capsules, 1, 1, out_capsules, pose_size, pose_size]
        # in_activation: Tensor with shape[batch_size, in_rows, in_cols, in_capsules, 1, 1, 1, 1, 1]
        in_vote, in_activation = caps_affine_transform(in_pose, in_activation, n_classes)

        # EM Routing
        pose, activation = em_routing(in_vote, in_activation, n_routing_iterations, init_inverse_temp,
                                      final_inverse_temp)

        #tf.summary.image('classcaps_activation_image_cap_0', tf.transpose(activation, [0, 1, 3, 2]), max_outputs=1)

        pose = tf.squeeze(pose, [1, 2])  # [batch_size, n_classes, pose_size, pose_size]
        activation = tf.squeeze(activation, [1, 2])  # [batch_size, n_classes]

        return pose, activation


def spread_loss(in_activation, label, margin):
    """
    Creates the TensorFlow graph for the spread loss detailed in the paper
    :param in_activation: Tensor with shape [batch_size, n_classes]
    :param label: Tensor with shape [batch_size, n_classes] containing the one-hot class labels
    :return: Tensor containing the scalar loss
    """
    with tf.variable_scope('spread_loss'):
        label = tf.cast(label, tf.float32)

        # Get activation of the correct class
        activation_target = tf.reduce_sum(tf.multiply(in_activation, label), axis=1, keep_dims=True)  # [batch_size, 1]

        # Get activations of incorrect classes
        # Subtracting label so that the value for a_t here will become a_t - 1 which will result in
        # (margin - (a_t - (a_t - 1))) < 0 such that the loss for the a_t element will be 0.
        # Seems like a slightly inelegant solution but works
        activation_other = in_activation - label

        # Get margin loss for each incorrect class
        # [batch_size, n_classes - 1]
        l_i = tf.square(tf.maximum(0., margin - (activation_target - activation_other)))

        # Get total loss for each batch element
        # [batch_size]
        l = tf.reduce_sum(l_i, axis=1)

        # Take mean of total loss over batch
        spread_loss = tf.reduce_mean(l)

        return spread_loss


def build_capsnetem_graph(placeholders, relu_conv1_params, primarycaps_params, convcaps1_params,
                          convcaps2_params, classcaps_params, spread_loss_params, image_dim=784):
    """
    Builds the TensorFlow graph for the capsules model (named CapsNetEM here) presented in the paper 'Matrix Capsules
    with EM Routing'
    :param placeholders: Dict containing TensorFlow placeholders for the input image and labels under 'image' and 'label'
    :param relu_conv1_params: dict containing the parameters for the relu conv1 layer
    :param primarycaps_params: dict containing the parameters for the primarycaps layer
    :param convcaps1_params: dict containing the parameters for the convcaps1 layer
    :param convcaps2_params: dict containing the parameters for the convcaps2 layer
    :param classcaps_params: dict containing the parameters for the classcaps layer
    :param spread_loss_params: dict containing the parameters for the spread loss
    :param image_dim: Int dimension of the flattened image (i.e. image_dim = image_height*image_width)
    :return: loss: Scalar Tensor
             predictions: Tensor with shape [batch_size] containing predicted classes
             accuracy: Scalar Tensor
             correct: Tensor with shape [batch_size] containing 1 or 0 for correct/incorrect classification
             summaries: dict containing tensorboard summaries so that we can be selective about when we run each summary
    """
    # Initalise summaries dict - using dict so that we can merge only select summaries; don't want image summaries all
    # the time
    summaries = {}
    summaries["general"] = []

    # Reshape flattened image tensor to 2D
    images = tf.reshape(placeholders['image'], [-1, 28, 28, 1])
    #summaries['images'] = tf.summary.image('input_images', images)
    tf.summary.image('input_images', images, max_outputs=1)

    # Create ReLU Conv1 de-rendering layer
    with tf.variable_scope('relu_conv1'):
        relu_conv1_out = tf.layers.conv2d(images, **relu_conv1_params,
                                          activation=tf.nn.relu)  # [batch_size, 12?, 12?, relu_conv1_filters]

    # Create PrimaryCaps layer
    primarycaps_pose, primarycaps_activation = primarycaps_layer(relu_conv1_out, **primarycaps_params)
    tf.summary.image('primarycaps_activation_image_cap_0', tf.expand_dims(primarycaps_activation[:, :, :, 0], 3), max_outputs=1)
    tf.summary.image('primarycaps_activation_image_cap_0', tf.expand_dims(primarycaps_activation[:, :, :, 1], 3), max_outputs=1)
    tf.summary.image('primarycaps_activation_image_cap_0', tf.expand_dims(primarycaps_activation[:, :, :, 2], 3), max_outputs=1)

    # Create ConvCaps1 layer
    #convcaps1_pose, convcaps1_activation = convcaps_layer(primarycaps_pose, primarycaps_activation, **convcaps1_params)
    #tf.summary.image('convcaps1_activation_image_cap_0', tf.expand_dims(convcaps1_activation[:, :, :, 0], 3), max_outputs=1)
    #tf.summary.image('convcaps1_activation_image_cap_0', tf.expand_dims(convcaps1_activation[:, :, :, 1], 3), max_outputs=1)
    #tf.summary.image('convcaps1_activation_image_cap_0', tf.expand_dims(convcaps1_activation[:, :, :, 2], 3), max_outputs=1)

    # Create ConvCaps2 layer
    #convcaps2_pose, convcaps2_activation = convcaps_layer(convcaps1_pose, convcaps1_activation, **convcaps2_params)
    #convcaps2_pose, convcaps2_activation = convcaps_layer(primarycaps_pose, primarycaps_activation, **convcaps2_params)
    #tf.summary.image('convcaps2_activation_image_cap_0', tf.expand_dims(convcaps2_activation[:, :, :, 0], 3), max_outputs=1)
    #tf.summary.image('convcaps2_activation_image_cap_0', tf.expand_dims(convcaps2_activation[:, :, :, 1], 3), max_outputs=1)
    #tf.summary.image('convcaps2_activation_image_cap_0', tf.expand_dims(convcaps2_activation[:, :, :, 2], 3), max_outputs=1)

    # Create Class Capsules layer
    #classcaps_pose, classcaps_activation = classcaps_layer(convcaps2_pose, convcaps2_activation, **classcaps_params)
    classcaps_pose, classcaps_activation = classcaps_layer(primarycaps_pose, primarycaps_activation, **classcaps_params)

    # Create spread loss
    loss = spread_loss(classcaps_activation, placeholders['label'], **spread_loss_params)

    # Get predictions, accuracy, correct and summaries
    with tf.name_scope("accuracy"):
        predictions = tf.argmax(classcaps_activation, axis=1)
        labels = tf.argmax(placeholders['label'], axis=1)
        correct = tf.cast(tf.equal(labels, predictions), tf.int32)
        accuracy = tf.reduce_sum(correct)/tf.shape(correct)[0]  # reduce_mean not working here for some reason
        summaries['accuracy'] = tf.summary.scalar('accuracy', accuracy)
    summaries['loss'] = tf.summary.scalar('loss', loss)

    summaries['general'].append(tf.summary.histogram('primarycaps_activation', primarycaps_activation))
    #summaries['general'].append(tf.summary.histogram('convcaps1_activation', convcaps1_activation))
    #summaries['general'].append(tf.summary.histogram('convcaps2_activation', convcaps2_activation))
    summaries['general'].append(tf.summary.histogram('classcaps_activation', classcaps_activation))
    summaries['general'].append(tf.summary.histogram('correct', correct))

    return loss, predictions, accuracy, correct, summaries


def save_mnist_as_image(mnist_batch, outdir, name="image"):
    # If outdir doesn't exist then create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for i, image in enumerate(tqdm(mnist_batch, desc="Saving images", leave=False, ncols=100)):
        image = np.squeeze(image)
        plt.imsave("{}/{}_{}.png".format(outdir, name, i), image)
