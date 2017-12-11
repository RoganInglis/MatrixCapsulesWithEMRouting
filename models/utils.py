import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import math


# Define eps - small constant for safe division/log
div_eps = 1e-6  # TODO - with these > 0 we dont get NaNs but the loss just seems to increase linearly
log_eps = 1e-6


def safe_divide(x, y, name=None):
    with tf.variable_scope('safe_divide'):
        y = tf.maximum(y, div_eps)
        return tf.divide(x, y, name=name)


def safe_log(x, name=None):
    with tf.variable_scope('safe_log'):
        x = tf.maximum(x, log_eps)
        return tf.log(x, name=name)


def conv_out_size(in_size, kernel_size, strides, padding):
    if padding is 'VALID':
        p = [0, 0]
    else:
        # Skipping the factor of 1/2 followed by the factor of 2 here as p is not needed for anything else here
        p = [kernel_size[0] - 1, kernel_size[1] - 1]

    out_size = [(in_size[0] + p[0] - kernel_size[0]) // strides[0] + 1,
                (in_size[1] + p[1] - kernel_size[1]) // strides[1] + 1]
    return out_size


def extract_image_patches_nd(input_tensor, ksizes, strides, rates=(1, 1, 1, 1), padding='SAME', name=None):
    """
    Extension of tf.extract_image_patches to tensors of arbitrary dimensions
    :param input_tensor: Tensor with at least 4 dimensions - [batch_size, image_height, image_width, depth, ...]
    :param ksizes: Tuple or List containing 4D kernel size - [1, kernel_rows, kernel_cols, 1]
    :param strides: Tuple or List containing 4D strides - [1, stride_rows, stride_cols, 1]
    :param rates: Tuple or List containing rates for dilated kernels - [1, rate_rows, rate_cols, 1]
    :param padding: 'SAME' or 'VALID' specifying the padding to use
    :param name: Name of the extract_image_patches op
    :return: patches: Tensor with shape [batch_size, kernel_rows, kernel_cols, depth, out_rows, out_cols, ...]
    """
    with tf.variable_scope('extract_image_patches_nd'):
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

        return patches


def reduce_logconv2d_transposeexp(input_tensor, ksizes, strides, rates=(1, 1, 1, 1), padding='SAME', name=None):
    """
    This function takes an input tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1] and
    returns a tensor with shape [batch_size, in_rows, in_cols, 1, out_rows_per_patch, out_cols_per_patch, out_capsules, 1, 1]
    for which
    :param input_tensor:
    :param ksizes:
    :param strides:
    :param rates:
    :param padding:
    :param name:
    :return:
    """


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


def convcaps_affine_transform(in_pose, in_activation, out_capsules, kernel_size, strides, padding):
    """
    Creates the TensorFlow graph for the convolutional affine transform performed prior to routing in a convolutional
    capsule layer. This also extracts image patches from and reshapes in_activation in order to keep the code and graph
    clean.
    :param in_pose: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, in_rows, in_cols, in_capsules]
    :param out_capsules: Int specifying the number of output capsules
    :param kernel_size: Int, Tuple or List specifying the size of the convolution kernel (assuming square kernel if int)
    :param strides: Int, Tuple or List specifying the strides for the convolution (assuming equal over dimensions if int)
    :param padding: 'valid' or 'same' specifying padding to use in the same way as tf.nn.conv2d
    :return: vote: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
    """
    # Sort out different possible kernel_size and strides formats
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(strides, int):
        strides = [strides, strides]

    with tf.variable_scope('convcaps_affine_transform'):
        # Get required shape values
        batch_size = tf.shape(in_pose)[0]
        shape_list = in_pose.get_shape().as_list()
        in_rows = shape_list[1]
        in_cols = shape_list[2]
        in_capsules = shape_list[3]
        pose_size = shape_list[4]

        # Compute output im grid size
        out_size = conv_out_size([in_rows, in_cols], kernel_size, strides, padding)

        # Create convolutional matmul kernel and tile over batch and out_size (as we need the same kernel to be
        # multiplied by each patch of conv_pose for each element in the batch)
        kernel = tf.Variable(tf.random_normal([1, *kernel_size, in_capsules, 1, 1, out_capsules, pose_size, pose_size]),
                             name='kernel')
        kernel = tf.tile(kernel, [batch_size, 1, 1, 1, *out_size, 1, 1, 1])

        # Re-organise in_pose so performing matmul with kernel computes the required convolutional affine transform
        conv_pose = tf.reshape(in_pose, [batch_size, in_rows, in_cols, in_capsules, 1, pose_size, pose_size])

        # Get patches from conv_pose and concatenate over out_size in correct dimensions so that new shape is:
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], 1, pose_size, pose_size]
        conv_pose = extract_image_patches_nd(conv_pose, [1, *kernel_size, 1], [1, *strides, 1], padding=padding, name='extract_pose_patches')

        # Tile over out_capsules: need to be multiplying the same input tensor by different weights for each out capsule
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], out_capsules, pose_size, pose_size]
        conv_pose = tf.tile(conv_pose, [1, 1, 1, 1, 1, 1, out_capsules, 1, 1])

        vote = tf.matmul(kernel, conv_pose)

        # Get patches from in_activation and expand dims
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1]]
        activation = extract_image_patches_nd(in_activation, [1, *kernel_size, 1], [1, *strides, 1], padding=padding, name='extract_activation_patches')
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], 1, 1, 1]
        activation = tf.reshape(activation, [batch_size, *kernel_size, in_capsules, *out_size, 1, 1, 1])
        #tf.summary.histogram('convcaps_affine_activation', activation)

        return vote, activation


def caps_affine_transform(in_pose, in_activation, out_capsules):
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
        weights = tf.Variable(tf.random_normal([1, 1, 1, in_capsules, 1, 1, out_capsules, pose_size, pose_size]),
                             name='weights')
        weights = tf.tile(weights, [batch_size, in_rows, in_cols, 1, 1, 1, 1, 1, 1])

        # Re-organise in_pose so performing matmul with kernel computes the required convolutional affine transform
        pose = tf.reshape(in_pose, [batch_size, in_rows, in_cols, in_capsules, 1, 1, 1, pose_size, pose_size])

        # Tile over out_capsules: need to be multiplying the same input tensor by different weights for each out capsule
        # [batch_size, in_rows, in_cols, in_capsules, 1, 1, out_capsules, pose_size, pose_size]
        pose = tf.tile(pose, [1, 1, 1, 1, 1, 1, out_capsules, 1, 1])

        vote = tf.matmul(weights, pose)

        # Do coordinate addition
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
        col_coord, row_coord = tf.meshgrid(list(np.arange(in_rows) / in_rows), list(np.arange(in_cols) / in_cols))

        # Expand dimensions and concatenate so they can be added to vote
        # [1, in_rows, in_cols, 1, 1, 1, 1, 1, 1]
        row_coord = expand_dims_nd(row_coord, [0, 3, 4, 5, 6, 7, 8])
        col_coord = expand_dims_nd(col_coord, [0, 3, 4, 5, 6, 7, 8])
        # [1, in_rows, in_cols, 1, 1, 1, 1, 1, 2]
        coords = tf.concat([row_coord, col_coord], axis=-1)
        full_coords = tf.concat([coords, tf.zeros(coords.get_shape())], axis=-1)

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
    :param beta_v: Scalar TensorFlow variable
    :param beta_a: Scalar TensorFlow variable
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
        r = tf.multiply(r, in_activation, name='r_update_mul')  # TODO - ~1e-4

        # Update means (out_poses)
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        r_reduce_sum = tf.reduce_sum(r, axis=[1, 2, 3], keep_dims=True)
        
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
        mean = safe_divide(tf.reduce_sum(tf.multiply(r, in_vote, name='mean_mul'), axis=[1, 2, 3], keep_dims=True),
                           r_reduce_sum)  # TODO - mean_mul has zeros here in the same places that the -infs seem to appear below

        # Update variances (same shape as mean)
        diff_vote_mean = tf.subtract(in_vote, mean, name='vote_mean_sub')
        variance = safe_divide(tf.reduce_sum(tf.multiply(r, tf.square(diff_vote_mean, name='diff_vote_mean_square'),
                                                         name='variance_mul'), axis=[1, 2, 3], keep_dims=True),
                               r_reduce_sum)  # TODO - real issue seems to be here?

        # Compute cost (same shape as mean)
        cost_h = tf.multiply(tf.add(beta_v, 0.5*safe_log(variance, name='log_variance'), name='add_beta_log_variance'),
                             r_reduce_sum, name='cost_h_mul')  # TODO - -inf appears here first

        # Compute new activations
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        activation = tf.nn.sigmoid(inverse_temp*(tf.subtract(beta_a, tf.reduce_sum(cost_h, axis=[-2, -1], keep_dims=True))))
        """
        r = tf.multiply(r, in_activation, name='mul_1')  # TODO - ~1e-4

        # Update means (out_poses)
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        r_reduce_sum = tf.reduce_sum(r, axis=[1, 2, 3], keep_dims=True, name='reduce_sum_2')

        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]             TODO - NO SAFE DIVIDE OR LOG EPS, 1 ROUTING ITERATION                   - SAFE DIVIDE AND LOG EPS 1E-6
        mul_3 = tf.multiply(r, in_vote, name='mul_3')                                             # TODO - zeros in same places -infs appear                                -

        reduce_sum_4 = tf.reduce_sum(mul_3, axis=[1, 2, 3], keep_dims=True, name='reduce_sum_4')  # TODO - zeros in same places -infs appear                                -

        mean = safe_divide(reduce_sum_4, r_reduce_sum, name='safe_divide_5')                      # TODO - zeros in same places -infs appear                                -

        # Update variances (same shape as mean)
        diff_vote_mean = tf.subtract(in_vote, mean, name='sub_6')                                 # TODO - zeros in same places -infs appear                                -

        square_7 = tf.square(diff_vote_mean, name='square_7')                                     # TODO - zeros in same places -infs appear, values around e-02 -> e-08    -

        mul_8 = tf.multiply(r, square_7, name='mul_8')                                            # TODO - zeros in same places -infs appear, values around e-05 -> e-11    -

        reduce_sum_9 = tf.reduce_sum(mul_8, axis=[1, 2, 3], keep_dims=True, name='reduce_sum_9')  # TODO - zeros in same places -infs appear values around e-2/e-3          -

        variance = safe_divide(reduce_sum_9, r_reduce_sum, name='safe_divide_10')                 # TODO - zeros in same places -infs appear, values around e-1/e-2         -

        # Compute cost (same shape as mean)
        safe_log_11 = safe_log(variance, name='safe_log_11')                                      # TODO - -infs appear here due to log of 0s in variance                   -

        add_12 = tf.add(beta_v, 0.5 * safe_log_11, name='add_12')

        cost_h = tf.multiply(add_12, r_reduce_sum, name='cost_h_mul_13')

        # Compute new activations
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        reduce_sum_14 = tf.reduce_sum(cost_h, axis=[-2, -1], keep_dims=True, name='reduce_sum_14')

        sub_15 = tf.subtract(beta_a, reduce_sum_14, 'sub_15')

        activation = tf.nn.sigmoid(inverse_temp * sub_15, name='sigmoid_16')
        """
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
        # TODO - SEEMS TO AT LEAST PARTLY WORK WITH ONLY 1 ROUTING ITERATION, I.E. NO E-STEP, SO THERE MUST BE AN ISSUE HERE - MOST LIKELY WITH THE FINAL OP NOT CONSIDERING RECEPTIVE FIELDS
        # Compute log(P): the log probability of each in_vote (data point)
        a = 0.5*safe_log(2*math.pi*variance)  # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
        b = 0.5*safe_divide(tf.square(tf.subtract(in_vote, mean, name='b_sub')), variance)  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]

        log_p = tf.subtract(-a, b, name='log_p_sub')  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]
        #log_p = log_p - (tf.reduce_max(log_p, axis=[-2, -1], keep_dims=True) - tf.log(10.))  # TODO - this line apparently helps with stability in the implementation credited in the readme, still getting NaN with it though?

        log_p_sum = tf.reduce_sum(log_p, axis=[-2, -1], keep_dims=True)  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]

        # Compute updated r (assignment probability/responsibility)
        # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
        log_p_activation = safe_log(activation) + log_p_sum  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]

        log_p_activation_sum = tf.reduce_logsumexp(log_p_activation, axis=[4, 5, 6], keep_dims=True)  # [batch_size, kernel_rows, kernel_cols, in_capsules, 1, 1, 1, 1, 1]  TODO - want to do this sum only over out_rows and out_cols for which the element specified by kernel_row, kernel column is within the receptive field

        # [batch_size, ]
        r = tf.exp(log_p_activation - log_p_activation_sum)  # TODO - Check bottom of page 5 of paper. Should we be considering receptive fields here? Could not considering receptive fields here lead to diffusion of small values at edges of variance? should we be summing over input capsules or output capsules

        return r


def em_routing(in_vote, in_activation, n_routing_iterations=3, init_inverse_temp=0.1, final_inverse_temp=0.9):
    """
    Creates the TensorFlow graph for the EM routing between capsules. Takes in and outputs 2D grids of capsules so that 
    it will work between convolutional layers, but it may be used for routing between non convolutional layers by using
    a 2D grid where the size of one dimension is 1 and taking kernel_rows/cols to be input_rows/cols
    :param in_vote: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, 1, 1, 1]
    :param n_routing_iterations: Int number of iterations to run the EM algorithm for the routing
    :param init_inverse_temp: Initial value for the inverse temperature parameter used in the M-step
    :param final_inverse_temp: Final value (on last iteration) of the inverse temperature parameter for the M-step
    :return: pose: Tensor with shape [batch_size, out_rows, out_cols, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, out_rows, out_cols, out_capsules]
    """
    with tf.variable_scope('em_routing'):
        # Get required shapes
        batch_size = tf.shape(in_vote)[0]
        shape_list = in_vote.get_shape().as_list()
        kernel_size = [shape_list[1], shape_list[2]]
        in_capsules = shape_list[3]
        out_size = [shape_list[4], shape_list[5]]
        out_capsules = shape_list[6]

        # Create R constant and initialise as uniform probability (stores the probability of each vote under each
        # Gaussian with mean defined by the output pose)
        r = tf.ones([batch_size, *kernel_size, in_capsules, *out_size, out_capsules, 1, 1], name="R")/(np.prod(out_size)*out_capsules)

        # Create beta parameters
        beta_v = tf.Variable(tf.random_normal([1, 1, 1, 1, *out_size, out_capsules, 1, 1]), name="beta_v")  # TODO - should the betas have different values for each output capsule? seems to learn much slower if they do, may end up with higher accuracy though? (not fully tested yet)
        beta_a = tf.Variable(tf.random_normal([1, 1, 1, 1, *out_size, out_capsules, 1, 1]), name="beta_a")
        tf.summary.histogram('beta_v', beta_v)
        tf.summary.histogram('beta_a', beta_a)

        # Initialise inverse temperature parameter and compute how much to increment by for each routing iteration
        inverse_temp = init_inverse_temp
        if n_routing_iterations > 1:
            inverse_temp_increment = (final_inverse_temp - init_inverse_temp)/(n_routing_iterations - 1)
        else:
            inverse_temp_increment = 0  # Cant increment anyway in this case

        # TODO - should we be stopping the gradient of the mean and/or activations here? doesnt seem to make much difference for primarycaps -> classcaps network
        #in_vote_stopped = tf.stop_gradient(in_vote)
        #in_activation_stopped = tf.stop_gradient(in_activation)
        in_vote_stopped = in_vote
        in_activation_stopped = in_activation

        # Do routing iterations
        for routing_iteration in range(n_routing_iterations - 1):
            with tf.variable_scope("routing_iteration_{}".format(routing_iteration)):
                # Do M-Step to get Gaussian means and standard deviations and update activations
                mean, std_dev, activation = m_step(r, in_activation_stopped, in_vote_stopped, beta_v, beta_a, inverse_temp)
                # tf.summary.histogram('em_routing_activation', activation)

                # Do E-Step to update R
                r = e_step(mean, std_dev, activation, in_vote)

                # Update inverse temp
                inverse_temp += inverse_temp_increment

        with tf.variable_scope("routing_iteration_{}".format(n_routing_iterations)):
            # Do M-Step to get Gaussian means and standard deviations and update activations
            mean, std_dev, activation = m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp)

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
        weights = tf.Variable(tf.random_normal([1, in_rows, in_cols, out_capsules, in_channels, (pose_size ** 2 + 1)]),
                              name='weights')
        weights = tf.tile(weights, [batch_size, 1, 1, 1, 1, 1])

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


def convcaps_layer(in_pose, in_activation, out_capsules, kernel_size, strides=1, padding='SAME', n_routing_iterations=3,
                   init_inverse_temp=0.1, final_inverse_temp=0.9):
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
    :param padding: 'valid' or 'same' specifying padding to use in the same way as tf.nn.conv2d
    :param n_routing_iterations: Number of iterations to use for the EM dynamic routing procedure
    :param init_inverse_temp: Scalar initial value for the inverse temperature parameter used for EM routing
    :param final_inverse_temp: Scalar final value for the inverse temperature parameter used for EM routing
    :return: pose: Tensor with shape [batch_size, out_rows, out_cols, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, out_rows, out_cols, out_capsules]
    """
    with tf.variable_scope('ConvCaps'):
        # Pose convolutional affine transform
        in_vote, in_activation = convcaps_affine_transform(in_pose, in_activation, out_capsules, kernel_size, strides, padding)

        # EM Routing
        pose, activation = em_routing(in_vote, in_activation, n_routing_iterations, init_inverse_temp, final_inverse_temp)

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
        pose, activation = em_routing(in_vote, in_activation, n_routing_iterations, init_inverse_temp, final_inverse_temp)

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

    # Create ConvCaps1 layer
    convcaps1_pose, convcaps1_activation = convcaps_layer(primarycaps_pose, primarycaps_activation, **convcaps1_params)

    # Create ConvCaps2 layer
    convcaps2_pose, convcaps2_activation = convcaps_layer(convcaps1_pose, convcaps1_activation, **convcaps2_params)

    # Create Class Capsules layer
    classcaps_pose, classcaps_activation = classcaps_layer(convcaps2_pose, convcaps2_activation, **classcaps_params)

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

    #summaries['general'].append(tf.summary.histogram('primarycaps_activation', primarycaps_activation))
    #summaries['general'].append(tf.summary.histogram('convcaps1_activation', convcaps1_activation))
    #summaries['general'].append(tf.summary.histogram('convcaps2_activation', convcaps2_activation))
    #summaries['general'].append(tf.summary.histogram('classcaps_activation', classcaps_activation))
    summaries['general'].append(tf.summary.histogram('correct', correct))

    return loss, predictions, accuracy, correct, summaries


def save_mnist_as_image(mnist_batch, outdir, name="image"):
    # If outdir doesn't exist then create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for i, image in enumerate(tqdm(mnist_batch, desc="Saving images", leave=False, ncols=100)):
        image = np.squeeze(image)
        plt.imsave("{}/{}_{}.png".format(outdir, name, i), image)