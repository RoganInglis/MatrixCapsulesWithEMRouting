import math
import tensorflow as tf
import numpy as np
from models import utils


# TODO - make sure all functions have been fully tested
# TODO - sort out interchangeable use of 'size' and 'shape'; change to 'shape'
# TODO - make sure use of dense_shape, full_shape and patches_shape is consistent
# Define eps - small constant for safe division/log
div_small_eps = 1e-12
div_zero_eps = 1e-12
div_big_eps = 1e12
log_eps = 1e-12


def safe_divide(x, y, name=None):
    if name is None:
        scope_name = 'safe_divide'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Want to clamp any values of y that are less than div_small_eps to div_small_eps
        #y_eps = tf.where(tf.greater(tf.abs(y), div_small_eps), y, tf.sign(y) * div_small_eps * tf.ones_like(y))
        y = tf.where(tf.greater(tf.abs(y), div_small_eps), y, tf.sign(y) * div_small_eps + y)  # Testing just adding eps to small y here in an attempt to preserve gradient

        z = tf.divide(x, y, name=name)

        return z


def safe_log(x, name=None):
    if name is None:
        scope_name = 'safe_log'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Want to clamp any values of x less than log_eps to log_eps
        x = tf.maximum(x, log_eps)
        y = tf.log(x)
        return y


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


def sparse_reduce_sum_nd(sparse_tensor, dense_shape, axis=None, keep_dims=False, mode='FAST', name=None):
    """
    Version of sparse_reduce_sum that works with tensors >5D. This assumes that dim 0 is batch size and axis is 3
    (or 4?)D max.
    :param sparse_tensor:
    :param dense_shape
    :param axis:
    :param keep_dims:
    :param mode:
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
        if mode is 'FAST':
            # Convert sparse tensor to dense
            dense_tensor = tf.sparse_tensor_to_dense(sparse_tensor, validate_indices=False)

            # Rehape sparse_full to regain shape info
            dense_tensor = tf.reshape(dense_tensor, dense_shape)

            # Do standard reduce sum
            full_tensor = tf.reduce_sum(dense_tensor, axis=axis, keep_dims=keep_dims)
        else:
            # Need to sparse reshape to get around 5D limitation of sparse reduce sum
            # Transpose so that axis is/are first + (leaving batch as zeroth) dims of tensor
            perm = list(range(sparse_tensor.get_shape().ndims))
            dense_perm_shape = dense_shape
            new_axis = axis
            for i, ax in enumerate(axis):
                # i + 1 index here as we want to leave batch as dim 0
                perm[i + 1], perm[ax] = perm[ax], perm[i + 1]
                dense_perm_shape[i + 1], dense_perm_shape[ax] = dense_perm_shape[ax], dense_perm_shape[i + 1]
                new_axis[i] = i + 1

            # Transpose sparse tensor so that non batch or axis dims are last
            sparse_tensor = tf.sparse_transpose(sparse_tensor, perm)

            if len(axis) + 1 < len(perm):
                condensed_shape = [*dense_perm_shape[:len(axis) + 1], np.prod(dense_perm_shape[len(axis) + 1:])]
            else:
                condensed_shape = dense_perm_shape

            # Reshape to combine other dims
            sparse_tensor = tf.sparse_reshape(sparse_tensor, condensed_shape)

            # Do sparse reduce sum on resulting sparse tensor to get the required dense tensor
            full_tensor = tf.sparse_reduce_sum(sparse_tensor, axis=new_axis, keep_dims=True)

            # Reshape to regain condensed dimensions
            new_shape = dense_perm_shape
            for ax in new_axis:
                new_shape[ax] = 1

            full_tensor = tf.reshape(full_tensor, new_shape)

            # Transpose back to original orientation
            full_tensor = tf.transpose(full_tensor, perm)

            # Squeeze over axis if keep dims is False
            if not keep_dims:
                full_tensor = tf.squeeze(full_tensor, axis)

        return full_tensor


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
        dense_shape = get_patches_full_shape(input_tensor, strides, rates, padding, in_size)

        # Do sparse reduce sum
        full_tensor = sparse_reduce_sum_nd(sparse_tensor, dense_shape, axis=axis, keep_dims=keep_dims)

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
        # Convert to sparse tensor
        sparse_tensor = patches_to_sparse(input_tensor, strides, rates, padding, in_size)
        dense_shape = get_patches_full_shape(input_tensor, strides, rates, padding, in_size)

        # Do sparse reduce logsumexp on resulting sparse tensor
        full_tensor = sparse_reduce_logsumexp(sparse_tensor, dense_shape, axis=axis, keep_dims=keep_dims)

        return full_tensor


def sparse_reduce_logsumexp(sparse_tensor, dense_shape, axis=None, keep_dims=False, mode='FAST', name=None):
    """
    Sparse version of tf.reduce_logsumexp that take a SparseTensor as input and returns a dense tensor. The design of
    this function takes advantage some properties that may be specific to this situation so care should be taken using
    this in other projects.
    :param sparse_tensor:
    :param dense_shape:
    :param axis:
    :param keep_dims:
    :param mode:
    :param name:
    :return: full_tensor:
    """
    if name is None:
        scope_name = 'sparse_reduce_logsumexp'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        if mode is 'FAST':
            # First convert sparse to dense for reduce max (assumes tensor is no negative?)
            dense_tensor = tf.sparse_tensor_to_dense(sparse_tensor, validate_indices=False)

            # Rehape sparse_full to regain shape info
            dense_tensor = tf.reshape(dense_tensor, dense_shape)

            # Now compute raw max using standard rather than sparse reduce max
            raw_max = tf.reduce_max(dense_tensor, axis=axis, keep_dims=True)

            # Compute my_max
            conditional = tf.where(tf.is_finite(raw_max), raw_max, tf.zeros_like(raw_max))
            my_max = tf.stop_gradient(conditional)

            # Subtract my_max
            exp_values = tf.exp(dense_tensor - my_max)
        else:
            raw_max = tf.sparse_reduce_max(sparse_tensor, axis=axis, keep_dims=keep_dims)

            # Reshape to regain shape info for raw_max
            reduced_shape = dense_shape.copy()
            for ax in axis:
                reduced_shape[ax] = 1
            raw_max = tf.reshape(raw_max, reduced_shape)

            conditional = tf.where(tf.is_finite(raw_max), raw_max, tf.zeros_like(raw_max))
            my_max = tf.stop_gradient(conditional)

            # Exp
            # Convert sparse_tensor to full so that we can subtract my_max
            dense_tensor = tf.sparse_tensor_to_dense(sparse_tensor, validate_indices=False)

            # Rehape sparse_full to regain shape info
            dense_tensor = tf.reshape(dense_tensor, dense_shape)

            # Subtract my_max
            exp_values = tf.exp(dense_tensor - my_max)

        # Extract only required values from exp_values (according to sparse_tensor.indices)
        new_dense_shape = tf.reduce_prod(sparse_tensor.dense_shape, keep_dims=True)
        reshape_sparse_tensor = tf.sparse_reshape(sparse_tensor, new_dense_shape)
        exp_values = tf.reshape(exp_values, [-1])
        exp_values = tf.gather_nd(exp_values, reshape_sparse_tensor.indices)

        # Convert back to sparse
        exp_sparse = tf.SparseTensor(sparse_tensor.indices, exp_values, sparse_tensor.dense_shape)

        # Sum
        sum_sparse = sparse_reduce_sum_nd(exp_sparse, dense_shape, axis=axis, keep_dims=True)

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
    ksizes, strides, rates = utils.get_correct_ksizes_strides_rates(ksizes, strides, rates)
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
            in_rows, in_cols = utils.conv_in_size([out_rows, out_cols], [kernel_rows, kernel_cols], list(strides[1:3]),
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


def fast_patches_to_sparse(input_tensor, dense_indices, shape_list, strides, rates=(1, 1, 1, 1), padding='SAME', in_size=None, name=None):
    """
    Convert a dense tensor containing image patches to a sparse tensor for which the image patches retain their original
    indices. Compared to patches_to_sparse, fast_patches_to_sparse takes the dense_indices and shape_list as additional
    inputs so that repeated calls to the function for the same shapes does not do repeated computation
    :param input_tensor: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
    :param dense_indices:
    :param shape_list:
    :param strides:
    :param rates:
    :param padding:
    :param in_size:
    :param name: Name for the op
    :return: sparse_patches: SparseTensor with shape [batch_size, im_rows, im_cols, ...]
    """
    if name is None:
        scope_name = 'fast_patches_to_sparse'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Get required shapes
        batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2 = shape_list

        # Compute input shape
        if in_size is None:
            in_rows, in_cols = utils.conv_in_size([out_rows, out_cols], [kernel_rows, kernel_cols], list(strides[1:3]),
                                            list(rates[1:3]), padding)
        else:
            in_rows, in_cols = in_size

        # Reshape input_tensor to 1D
        flat_shape = [batch_size*kernel_rows*kernel_cols*in_capsules*out_rows*out_cols*out_capsules*capsule_dim1*capsule_dim2]
        values = tf.reshape(input_tensor, shape=flat_shape)

        # Create constant tensor containing dense shape
        # TODO - might be a better way to do this but this works. Might also be able to pass the dense_shape in so these lines don't have to be recomputed, although tf.SparseTensor does require this in a specific format
        dense_shape = tf.shape(input_tensor, out_type=tf.int64)
        dense_shape = tf.add(dense_shape,
                             tf.constant([0, in_rows - kernel_rows, in_cols - kernel_cols, 0, 0, 0, 0, 0, 0],
                                         dtype=tf.int64))

        # Create sparse tensor
        sparse_patches = tf.SparseTensor(dense_indices, values, dense_shape)
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


def sparse_patches_to_full(sparse_tensor, full_shape, strides, rates=(1, 1, 1, 1), padding='SAME', in_size=None, name=None):
    """
    Converts a sparse tensor of image patches to a full tensor in which the image patches are embedded within an array of zeros
    with the same shape as the original image
    :param sparse_tensor: SparseTensor containing image patches with shape [batch_size, in_rows, in_cols, in_capsules,
                         out_rows, out_cols, out_capsules, ?, ?]
    :param full_shape:
    :param strides:
    :param rates:
    :param padding:
    :param in_size:
    :param name:
    :return:
    """
    if name is None:
        scope_name = 'sparse_patches_to_full'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # Then convert sparse to dense
        dense_patches = tf.sparse_tensor_to_dense(sparse_tensor, validate_indices=False)

        # This seems to lose the shape so reset shape
        #full_shape = get_patches_full_shape(input_tensor, strides, rates, padding, in_size=in_size)
        dense_patches = tf.reshape(dense_patches, full_shape)

        return dense_patches


def full_to_patches(full_tensor, indices, patches_shape, name=None):
    """
    Converts a full tensor of patches (i.e. with patches in their original position with the rest of the image padded
    with zeros and each patch image indexed by separate dimension(s))
    Assumes dim 0 is batch size and dims 1 & 2 are the patch/kernel dims
    :param full_tensor: Full tensor containing patches padded to full size and indexed by separate dimension(s)
    :param indices: Indices of the patches within the full tensor
    :param patches_shape: Shape of the resulting dense patches tensor
    :param name: Name for the op
    :return: patches_tensor: Tensor containing only patches, not padded to full size
    """
    if name is None:
        scope_name = 'full_to_patches'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        # TODO - This is a hacky way to do this, only need to because of the 5D limit of gather_nd. If this limit is raised to 9 or higher then we can remove everything here before the 'Extract patches' comment
        values = tf.zeros(patches_shape)
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
        patches_tensor = tf.reshape(patches_tensor, patches_shape)

        return patches_tensor


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
    if name is None:
        scope_name = 'expand_dims_nd'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        axis = list(axis)
        if len(axis) == 1:
            input_tensor = tf.expand_dims(input_tensor, axis=axis[0], name=name)
        else:
            for dim in axis:
                input_tensor = tf.expand_dims(input_tensor, axis=dim, name=name)

        return input_tensor


def get_shape_list(input_tensor, name=None):
    """
    Get the shape (as a list) of a tensor with dynamic batch dimension first
    :param input_tensor:
    :return: shape
    """
    if name is None:
        scope_name = 'get_shape_list'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        shape = [tf.shape(input_tensor)[0], *input_tensor.get_shape().as_list()[1:]]  # TODO - use this function everywhere where possible
        return shape


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
            in_size = utils.conv_in_size(out_size, kernel_size, strides[1:3], rates[1:3], padding)

        full_shape = [batch_size, *in_size, in_capsules, *out_size, out_capsules, *remaining_dims]

        return full_shape


def fast_get_patches_full_shape(patches_shape, strides, rates=(1, 1, 1, 1), padding='SAME', in_size=None, name=None):
    """
    Get the eqivalent full shape for a tensor containing image patches from the patches shape
    :param patches_shape:
    :param strides:
    :param rates:
    :param padding:
    :param in_size:
    :param name:
    :return:
    """
    if name is None:
        scope_name = 'fast_get_patches_full_shape'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        batch_size = patches_shape[0]
        kernel_size = patches_shape[1:3]
        in_capsules = patches_shape[3]
        out_size = patches_shape[4:6]
        out_capsules = patches_shape[6]
        remaining_dims = patches_shape[7:]

        if in_size is None:
            in_size = utils.conv_in_size(out_size, kernel_size, strides[1:3], rates[1:3], padding)

        full_shape = [batch_size, *in_size, in_capsules, *out_size, out_capsules, *remaining_dims]

        return full_shape


def get_dense_indices(patch_shape, in_size, strides, rates, padding, name=None):
    """
    Get the dense indices of a kernel patch array as a numpy array
    :param patch_shape: list of ints [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2]
    :param in_size:
    :param strides:
    :param rates:
    :param padding
    :return: indices
    """
    if name is None:
        scope_name = 'get_dense_indices'
    else:
        scope_name = name
    with tf.variable_scope(scope_name):
        batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, capsule_dim1, capsule_dim2 = patch_shape

        in_rows, in_cols = in_size

        # Compute padding
        k_dash = [kernel_rows + (kernel_rows - 1) * (rates[1] - 1),
                  kernel_cols + (kernel_cols - 1) * (rates[2] - 1)]
        if padding is 'VALID':
            p_rows = (out_rows - 1) * strides[1] + k_dash[0] - in_rows
            p_cols = (out_cols - 1) * strides[2] + k_dash[1] - in_cols
        elif padding is 'SAME':
            p_rows = ((out_rows - 1) * strides[1] + k_dash[0] - in_rows) // 2
            p_cols = ((out_cols - 1) * strides[2] + k_dash[1] - in_cols) // 2
        else:
            raise ValueError('Unexpected padding')
        p_rows, p_cols = max(0, p_rows), max(0, p_cols)

        # check https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/array_grad.py#L575 for
        # implementation of extract_image_patches gradient where then have to do the inverse as here
        indices = []
        # Construct for first batch element, out capsule, capsule dim 1 and capsule dim 2
        # TODO - can speed this up by taking in_capsules out of the loop and combining later, could also try doing this by combining np arrays
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

        return indices
