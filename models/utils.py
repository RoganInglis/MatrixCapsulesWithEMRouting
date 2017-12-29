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
div_eps = 1e-9
log_eps = 1e-9


def safe_divide(x, y, mode='ZERO', name=None):
    if name is None:
        scope_name = 'safe_divide'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        if mode == 'ZERO':
            z = tf.divide(x, y, name=name)
            z = tf.where(tf.is_finite(z), z, tf.zeros_like(z))
        elif mode == 'EPS_ZERO':
            # TODO - implement broadcasting by tiling for this to work properly
            x = tf.where(tf.greater(y, div_eps), x, tf.zeros_like(x))
            y = tf.where(tf.greater(y, div_eps), y, tf.ones_like(y))
            z = tf.divide(x, y, name=name)
        else:
            y = tf.maximum(y, div_eps)
            z = tf.divide(x, y, name=name)
        return z


def safe_log(x, mode='ZERO', name=None):
    if name is None:
        scope_name = 'safe_log'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        if mode == 'ZERO':
            x = tf.log(x)
            x = tf.where(tf.is_finite(x), x, tf.zeros_like(x))
        elif mode == 'EPS_ZERO':
            x = tf.where(tf.greater(x, log_eps), x, tf.ones_like(x))
            x = tf.log(x, name=name)
        else:
            x = tf.maximum(x, log_eps)
            x = tf.log(x, name=name)
        return x


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
    :param patch_shape: list of ints [kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2]
    :param in_size:
    :param strides:
    :param rates:
    :param padding
    :return: indices
    """
    kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2 = patch_shape

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

    # Repeat computed indices over out capsules, capsule dim 1 and capsule dim 2
    np_indices = np.repeat(np_indices, out_capsules * capsule_dim1 * capsule_dim2, axis=0)

    # Get indices for out capsules, capsule dim 1 and capsule dim 2 and repeat/tile to correct size
    capsule_dim2_indices = np.tile(np.arange(capsule_dim2), out_capsules * capsule_dim1)
    capsule_dim1_indices = np.tile(np.repeat(np.arange(capsule_dim1), capsule_dim2), out_capsules)
    out_capsules_indices = np.repeat(np.arange(out_capsules), capsule_dim2 * capsule_dim1)

    # Concatenate the indices just computed and tile over the previously computed indices
    extra_indices = np.transpose(np.stack([out_capsules_indices, capsule_dim1_indices, capsule_dim2_indices], axis=0))
    extra_indices = np.tile(extra_indices, [kernel_rows * kernel_cols * in_capsules * out_rows * out_cols, 1])

    # Concatenate the two sets of indices
    np_indices = np.concatenate([np_indices, extra_indices], axis=1)
    print("Reconstruction indices np time: {}s".format(time.time() - start))

    return np_indices


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
        """
        kernel_cols = shape_list[2]
        in_capsules = shape_list[3]
        out_rows = shape_list[4]
        out_cols = shape_list[5]
        out_capsules = shape_list[6]
        capsule_dim1 = shape_list[7]
        capsule_dim2 = shape_list[8]
        """

        # Compute input shape
        if in_size is None:
            in_rows, in_cols = conv_in_size([out_rows, out_cols], [kernel_rows, kernel_cols], list(strides[1:3]),
                                            list(rates[1:3]), padding)
        else:
            in_rows, in_cols = in_size

        # Get dense indices
        indices = get_dense_indices([kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2],
                                    [in_rows, in_cols], strides, rates, padding)

        indices_per_batch = len(indices)

        # Convert indices to constant tensor with dtype tf.int64
        indices = tf.constant(indices, dtype=tf.int64)

        # Extend indices over batch - Should be done within the graph so we can use variable batch size
        batch_indices = tf.cast(tf.reshape(tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, indices_per_batch]),
                                           [batch_size*indices_per_batch, 1]), dtype=tf.int64)
        indices = tf.concat([batch_indices, tf.tile(indices, [batch_size, 1])], axis=1)

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
        vote = patches_to_full(vote, strides, rates, padding, in_size=[in_rows, in_cols])
        activation = patches_to_full(activation, strides, rates, padding, in_size=[in_rows, in_cols])

        return vote, activation


def caps_affine_transform(in_pose, in_activation, out_capsules, coord_addition=True):
    """
    Creates the TensorFlow graph for the affine transform performed prior to routing in a capsule layer. This also
    reshapes in_activation in order to keep the code and graph clean.
    :param in_pose: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, in_rows, in_cols, in_capsules]
    :param out_capsules: Int, the number of output capsules
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


def m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp):
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
        r = tf.multiply(r, in_activation, name='r_update_mul')

        # Update means (out_poses)
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        r_reduce_sum = tf.reduce_sum(r, axis=[1, 2, 3], keep_dims=True)
        
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
        mean = safe_divide(tf.reduce_sum(tf.multiply(r, in_vote, name='mean_mul'), axis=[1, 2, 3], keep_dims=True),
                           r_reduce_sum)

        # Update variances (same shape as mean)
        diff_vote_mean = tf.subtract(in_vote, mean, name='vote_mean_sub')
        variance = safe_divide(tf.reduce_sum(tf.multiply(r, tf.square(diff_vote_mean, name='diff_vote_mean_square'),
                                                         name='variance_mul'), axis=[1, 2, 3], keep_dims=True),
                               r_reduce_sum)

        # Compute cost (same shape as mean)
        cost_h = tf.multiply(tf.add(beta_v, safe_log(tf.sqrt(variance), name='log_stdd'), name='add_beta_log_stdd'),
                             r_reduce_sum, name='cost_h_mul')

        # Compute new activations
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        activation = tf.nn.sigmoid(inverse_temp*(tf.subtract(beta_a, tf.reduce_sum(cost_h, axis=[-2, -1],
                                                                                   keep_dims=True))))
        return mean, variance, activation


def e_step(mean, variance, activation, in_vote):
    """
    Creates the TensorFlow graph for the E-Step of the EM Routing algorithm
    :param mean: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param variance: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param activation: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
    :param in_vote: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :return: r: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
    """
    with tf.variable_scope('e_step'):
        # Compute log(P): the log probability of each in_vote (data point)
        a = safe_log(2*math.pi*variance)  # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
        b = safe_divide(tf.square(tf.subtract(in_vote, mean, name='b_sub')), variance)  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]

        log_p = 0.5 * tf.subtract(-a, b, name='log_p_sub')  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]

        log_p_sum = tf.reduce_sum(log_p, axis=[-2, -1], keep_dims=True)  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]

        # Compute updated r (assignment probability/responsibility)
        # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
        log_p_activation = safe_log(activation) + log_p_sum

        log_p_activation_sum = tf.reduce_logsumexp(log_p_activation, axis=[4, 5, 6], keep_dims=True)

        # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
        r = tf.exp(log_p_activation - log_p_activation_sum)
        """
        NON LOGSUMEXP VERSION AS IN PAPER \/\/\/
        a = safe_divide(1, (tf.sqrt(tf.reduce_prod(2*math.pi*variance, axis=[-2, -1], keep_dims=True))))
        b = 0.5*tf.reduce_sum(safe_divide(tf.square(in_vote - mean), variance), axis=[-2, -1], keep_dims=True)

        p = a * tf.exp(-b)

        ap = a * p
        r = safe_divide(ap, tf.reduce_sum(ap, axis=[4, 5, 6], keep_dims=True))
        """
        return r


def em_routing(in_vote, in_activation, n_routing_iterations=3, init_inverse_temp=0.1, final_inverse_temp=0.9,
               ksizes=None, strides=(1, 1, 1, 1), rates=(1, 1, 1, 1), padding='SAME', conv=False):
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
        beta_v = tf.Variable(tf.truncated_normal([1, 1, 1, 1, 1, 1, out_capsules, 1, 1]), name="beta_v")
        beta_a = tf.Variable(tf.truncated_normal([1, 1, 1, 1, 1, 1, out_capsules, 1, 1]), name="beta_a")
        tf.summary.histogram('beta_v', beta_v)
        tf.summary.histogram('beta_a', beta_a)

        # Initialise inverse temperature parameter and compute how much to increment by for each routing iteration
        inverse_temp = init_inverse_temp
        if n_routing_iterations > 1:
            inverse_temp_increment = (final_inverse_temp - init_inverse_temp) / (n_routing_iterations - 1)
        else:
            inverse_temp_increment = 0  # Cant increment anyway in this case

        # If we are doing routing between convolutional capsules we need to convert r patches to full tensors
        if conv:
            strides = get_correct_conv_param(strides)
            rates = get_correct_conv_param(rates)
            r = patches_to_full(r, strides, rates, padding, in_size=in_activation.get_shape().as_list()[1:3])

        # TODO - remove this summary once debugged?
        tf.summary.image('in_activation_00', tf.reshape(tf.reshape(in_activation[0],
                                                                   shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 0, 0],
                                                        shape=[1, *in_activation.get_shape().as_list()[1:3], 1]))
        if conv:
            tf.summary.image('in_activation_01', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 0, 1],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]))
            tf.summary.image('in_activation_10', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 1, 0],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]))
            tf.summary.image('in_activation_02', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 0, 2],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]))
            tf.summary.image('in_activation_20', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 2, 0],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]))
            tf.summary.image('in_activation_22', tf.reshape(tf.reshape(in_activation[0],
                                                                       shape=[*in_activation.get_shape().as_list()[1:6]])[:, :, 0, 2, 2],
                                                            shape=[1, *in_activation.get_shape().as_list()[1:3], 1]))


        # TODO - should we definitely be stopping the gradient of the beta_v, beta_a, vote and/or activations here?
        in_vote_stopped = tf.stop_gradient(in_vote)
        in_activation_stopped = tf.stop_gradient(in_activation)
        beta_v_stopped = tf.stop_gradient(beta_v)
        beta_a_stopped = tf.stop_gradient(beta_a)

        # Do routing iterations
        for routing_iteration in range(n_routing_iterations - 1):
            with tf.variable_scope("routing_iteration_{}".format(routing_iteration)):
                # Do M-Step to get Gaussian means and standard deviations and update activations
                mean, std_dev, activation = m_step(r, in_activation_stopped, in_vote_stopped,
                                                   beta_v_stopped, beta_a_stopped, inverse_temp)
                tf.summary.histogram('em_routing_mean_{}'.format(routing_iteration), mean)
                tf.summary.histogram('em_routing_std_dev_{}'.format(routing_iteration), std_dev)
                tf.summary.histogram('em_routing_activation_{}'.format(routing_iteration), activation)

                # Do E-Step to update R
                r = e_step(mean, std_dev, activation, in_vote_stopped)
                tf.summary.histogram('em_routing_r_{}'.format(routing_iteration), r)

                # Update inverse temp
                inverse_temp += inverse_temp_increment

        with tf.variable_scope("routing_iteration_{}".format(n_routing_iterations)):
            # Do M-Step to get Gaussian means and update activations
            mean, _, activation = m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp)

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
        # Pose convolutional affine transform
        in_vote, in_activation = convcaps_affine_transform(in_pose, in_activation, out_capsules, ksizes, strides,
                                                           rates, padding)

        # EM Routing
        pose, activation = em_routing(in_vote, in_activation, n_routing_iterations, init_inverse_temp,
                                      final_inverse_temp, ksizes, strides, rates, padding, conv=True)

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

        tf.summary.image('classcaps_activation_image_cap_0', tf.transpose(activation, [0, 1, 3, 2]))

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

    # Create ReLU Conv1 de-rendering layer
    with tf.variable_scope('relu_conv1'):
        relu_conv1_out = tf.layers.conv2d(images, **relu_conv1_params,
                                          activation=tf.nn.relu)  # [batch_size, 12?, 12?, relu_conv1_filters]

    # Create PrimaryCaps layer
    primarycaps_pose, primarycaps_activation = primarycaps_layer(relu_conv1_out, **primarycaps_params)
    tf.summary.image('primarycaps_activation_image_cap_0', tf.expand_dims(primarycaps_activation[:, :, :, 0], 3))
    tf.summary.image('primarycaps_activation_image_cap_0', tf.expand_dims(primarycaps_activation[:, :, :, 1], 3))
    tf.summary.image('primarycaps_activation_image_cap_0', tf.expand_dims(primarycaps_activation[:, :, :, 2], 3))

    # Create ConvCaps1 layer
    convcaps1_pose, convcaps1_activation = convcaps_layer(primarycaps_pose, primarycaps_activation, **convcaps1_params)
    tf.summary.image('convcaps1_activation_image_cap_0', tf.expand_dims(convcaps1_activation[:, :, :, 0], 3))
    tf.summary.image('convcaps1_activation_image_cap_0', tf.expand_dims(convcaps1_activation[:, :, :, 1], 3))
    tf.summary.image('convcaps1_activation_image_cap_0', tf.expand_dims(convcaps1_activation[:, :, :, 2], 3))

    # Create ConvCaps2 layer
    convcaps2_pose, convcaps2_activation = convcaps_layer(convcaps1_pose, convcaps1_activation, **convcaps2_params)
    tf.summary.image('convcaps2_activation_image_cap_0', tf.expand_dims(convcaps2_activation[:, :, :, 0], 3))
    tf.summary.image('convcaps2_activation_image_cap_0', tf.expand_dims(convcaps2_activation[:, :, :, 1], 3))
    tf.summary.image('convcaps2_activation_image_cap_0', tf.expand_dims(convcaps2_activation[:, :, :, 2], 3))

    # Create Class Capsules layer
    classcaps_pose, classcaps_activation = classcaps_layer(convcaps2_pose, convcaps2_activation, **classcaps_params)
    #classcaps_pose, classcaps_activation = classcaps_layer(primarycaps_pose, primarycaps_activation, **classcaps_params)

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
    summaries['general'].append(tf.summary.histogram('convcaps1_activation', convcaps1_activation))
    summaries['general'].append(tf.summary.histogram('convcaps2_activation', convcaps2_activation))
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
