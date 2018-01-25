import math

import numpy as np
import tensorflow as tf

from models import tf_ops
from models.utils import get_correct_ksizes_strides_rates, conv_out_size, get_correct_conv_param, conv_in_size


def convcaps_affine_transform(in_pose, in_activation, out_capsules, kernel_size, strides, rates=(1, 1, 1, 1),
                              padding='SAME', summaries=False):
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
    :param summaries:
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
        kernel = tf.Variable(tf.truncated_normal([1, *ksizes[1:3], in_capsules, 1, 1, out_capsules, pose_size, pose_size], stddev=0.5), name='weights')
        if summaries:
            tf.summary.histogram('weights', kernel)

        kernel = tf.tile(kernel, [batch_size, 1, 1, 1, *out_size, 1, 1, 1])

        # Re-organise in_pose so performing matmul with kernel computes the required convolutional affine transform
        conv_pose = tf.reshape(in_pose, [batch_size, in_rows, in_cols, in_capsules, 1, pose_size, pose_size])

        # Get patches from conv_pose and concatenate over out_size in correct dimensions so that new shape is:
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], 1, pose_size, pose_size]
        conv_pose = tf_ops.extract_image_patches_nd(conv_pose, ksizes, strides, rates=rates, padding=padding,
                                                    name='extract_pose_patches')

        # Tile over out_capsules: need to be multiplying the same input tensor by different weights for each out capsule
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], out_capsules, pose_size, pose_size]
        conv_pose = tf.tile(conv_pose, [1, 1, 1, 1, 1, 1, out_capsules, 1, 1])

        vote = tf.matmul(kernel, conv_pose)

        # Get patches from in_activation and expand dims
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1]]
        activation = tf_ops.extract_image_patches_nd(in_activation, ksizes, strides, rates=rates, padding=padding,
                                              name='extract_activation_patches')
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], 1, 1, 1]
        activation = tf.reshape(activation, [batch_size, *ksizes[1:3], in_capsules, *out_size, 1, 1, 1])

        return vote, activation


def caps_affine_transform(in_pose, in_activation, out_capsules, coord_addition=True, summaries=False):
    """
    Creates the TensorFlow graph for the affine transform performed prior to routing in a capsule layer. This also
    reshapes in_activation in order to keep the code and graph clean.
    :param in_pose: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, in_rows, in_cols, in_capsules]
    :param out_capsules: Int, the number of output capsules
    :param coord_addition: Bool, whether to do coordinate addition
    :param summaries:
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
        weights = tf.Variable(tf.truncated_normal([1, 1, 1, in_capsules, 1, 1, out_capsules, pose_size, pose_size], stddev=0.5), name='weights')
        if summaries:
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
        activation = tf_ops.expand_dims_nd(in_activation, [4, 5, 6, 7, 8])

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
        row_coord = tf_ops.expand_dims_nd(row_coord, [0, 3, 4, 5, 6, 7, 8])
        col_coord = tf_ops.expand_dims_nd(col_coord, [0, 3, 4, 5, 6, 7, 8])
        # [1, in_rows, in_cols, 1, 1, 1, 1, 1, 2]
        coords = tf.concat([row_coord, col_coord], axis=-1)
        full_coords = tf.concat([coords, tf.zeros_like(coords)], axis=-1)

        # Add coordinates to vote
        vote += full_coords

        return vote


def m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp, conv=False, summaries=False):
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
    :param conv:
    :param summaries:
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
        mean = tf_ops.safe_divide(tf.reduce_sum(tf.multiply(rp, in_vote, name='mean_mul'), axis=[1, 2, 3],
                                                keep_dims=True),
                                  rp_reduce_sum, name='safe_divide_mean')

        # Update variances (same shape as mean)
        diff_vote_mean = tf.subtract(in_vote, mean, name='vote_mean_sub')
        variance = tf_ops.safe_divide(tf.reduce_sum(tf.multiply(rp,
                                                                tf.square(diff_vote_mean, name='diff_vote_mean_square'),
                                                                name='variance_mul'), axis=[1, 2, 3], keep_dims=True),
                                      rp_reduce_sum, name='safe_divide_variance')  # TODO - getting very high values with this. Is it possible to reduce with sensible initialisations?

        # Compute cost (same shape as mean)
        cost_h = tf.multiply(tf.add(beta_v, 0.5 * tf_ops.safe_log(variance, name='log_stdd'), name='add_beta_log_stdd'),
                             rp_reduce_sum, name='cost_h_mul')

        # Compute new activations
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        activation = tf.nn.sigmoid(inverse_temp*(tf.subtract(beta_a, tf.reduce_sum(cost_h, axis=[-2, -1],
                                                                                   keep_dims=True))), name='activation_sigmoid')

        if summaries:
            tf.summary.histogram('rp', rp)
            tf.summary.histogram('rp_reduce_sum', rp_reduce_sum)
            tf.summary.histogram('cost_h', cost_h)
            tf.summary.histogram('mean', mean)
            tf.summary.histogram('variance', variance)
            tf.summary.histogram('activation', activation)

        # TODO - add image summary bool optional param?
        if not conv:
            tf.summary.image('activation', tf.squeeze(activation, axis=[1, 2, 3, 4, 8]), max_outputs=1)
        else:
            tf.summary.image('activation', tf.expand_dims(tf.squeeze(activation, axis=[1, 2, 3, 7, 8])[:, :, :, 0], 3),
                             max_outputs=1)  # TODO - should unstack/concatenate so that we can display the activations of all capsules at once

        return mean, variance, activation


def e_step(mean, variance, activation, in_vote, logspace=True, summaries=False, **sparse_args):
    """
    Creates the TensorFlow graph for the E-Step of the EM Routing algorithm
    :param mean: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param variance: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param activation: Tensor with shape [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
    :param in_vote: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]
    :param logspace: Bool, whether to do this step in log space or not (as written in the paper)
    :param summaries:
    :param sparse_args:
    :return: r: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
    """
    with tf.variable_scope('e_step'):
        if logspace:
            # TODO - room for optimisation here. There are two identical (I think) calls to patches to sparse, within reduce_logsumexpsparse and within patches_to_full. These also each contain calls of get_dense_indices, which is also called direct within this function; are these all identical calls and if so can we use only one? (even if it makes the code a little bit messier). There are also what look like multiple calls to code to get the shape of tensors with the same shape?
            # TODO - should pass in result of get_
            # Compute log(P): the log probability of each in_vote (data point)
            a = tf_ops.safe_log(2*math.pi*variance, name='safe_log_a')  # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
            b = tf_ops.safe_divide(tf.square(tf.subtract(in_vote, mean, name='b_sub'), name='vote_mean_square'),
                                   variance, name='safe_divide_b')  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]

            log_p_pre_sum = tf.add(a, b, name='log_p_sub')  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size]

            log_p = -0.5 * tf.reduce_sum(log_p_pre_sum, axis=[-2, -1], keep_dims=True)  # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]

            # Compute updated r (assignment probability/responsibility)
            # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
            log_p_activation = tf_ops.safe_log(activation) + log_p

            if sparse_args['sparse']:
                # Get dense_indices and dense_shape
                patches_shape = sparse_args['patches_shape']
                dense_shape = sparse_args['dense_shape']
                dense_indices = sparse_args['dense_indices']

                # Convert log_p_activation to sparse
                sparse_log_p_activation = tf_ops.fast_patches_to_sparse(log_p_activation, dense_indices, patches_shape,
                                                                        sparse_args['strides'], sparse_args['rates'],
                                                                        sparse_args['padding'], sparse_args['in_size'])

                log_p_activation_sum = tf_ops.sparse_reduce_logsumexp(sparse_log_p_activation, dense_shape,
                                                                      axis=[4, 5, 6], keep_dims=True)

                # Convert log_p_activation to full for compatibility with log_p_activation_sum
                log_p_activation = tf.sparse_tensor_to_dense(sparse_log_p_activation, validate_indices=False)
                log_p_activation = tf.reshape(log_p_activation, dense_shape)

                # Compute new r
                r = tf.exp(log_p_activation - log_p_activation_sum)

                # Convert r back to patches (since activation_p is patches this is valid)
                r = tf_ops.full_to_patches(r, dense_indices, patches_shape)
            else:
                log_p_activation_sum = tf.reduce_logsumexp(log_p_activation, axis=[4, 5, 6], keep_dims=True)

                # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
                r = tf.exp(log_p_activation - log_p_activation_sum)

            if summaries:
                tf.summary.histogram('log_p', log_p)
                tf.summary.histogram('log_p_activation', log_p_activation)
        else:
            a = tf_ops.safe_log(2 * math.pi * variance)
            b = tf_ops.safe_divide(tf.square(in_vote - mean), variance, name='safe_divide_b')

            a_b = tf.add(a, b)

            a_b_sum = 0.5 * tf.reduce_sum(a_b, axis=[-2, -1], keep_dims=True)  # TODO - doing this sum gives numbers ~100/2 which then produces very small numbers in the next exp

            p = tf.exp(-a_b_sum, name='exp_p')  # TODO - very small numbers here

            # Clip to between 0 and 1 if safe divide has caused this not to be the case
            #p = tf.clip_by_value(p, 0., 1.)  # TODO - test without this (and pay attention to values in tensorboard)

            activation_p = tf.multiply(activation, p, name='activation_p_mul')

            if sparse_args['sparse']:
                activation_p_reduce_sum = tf_ops.reduce_sumsparse(activation_p, sparse_args['strides'], sparse_args['rates'],
                                                                  sparse_args['padding'], sparse_args['in_size'],
                                                                  axis=[4, 5, 6], keep_dims=True)

                # Get patch size and dense shape for conversion back to dense patches later
                dense_shape = tf_ops.get_shape_list(activation_p)

                # Convert activation_p to full for compatibility with activation_p_reduce_sum
                activation_p = tf_ops.patches_to_full(activation_p, sparse_args['strides'], sparse_args['rates'],
                                                      sparse_args['padding'], sparse_args['in_size'])

                # Compute new r
                r = tf_ops.safe_divide(activation_p, activation_p_reduce_sum)

                # Convert r back to patches (since activation_p is patches this is valid)
                indices = tf_ops.get_dense_indices(dense_shape, sparse_args['in_size'], sparse_args['strides'],
                                                   sparse_args['rates'], sparse_args['padding'])  # TODO - refactor as for logspace version to use passed in indices etc.
                r = tf_ops.full_to_patches(r, indices, dense_shape)

            else:
                activation_p_reduce_sum = tf.reduce_sum(activation_p, axis=[4, 5, 6], keep_dims=True)

                # Compute new r
                r = tf_ops.safe_divide(activation_p, activation_p_reduce_sum)

            # Clip r to between 0 and 1 if this is not the case
            #tf.clip_by_value(r, 0., 1.)  # TODO - test without this

            if summaries:
                tf.summary.histogram('p', p)
                tf.summary.histogram('activation_p', activation_p)

        if summaries:
            tf.summary.histogram('a', a)
            tf.summary.histogram('b', b)
            tf.summary.histogram('r', r)
        return r


def em_routing(in_vote, in_activation, n_routing_iterations=3, init_beta_v=1., init_beta_a=-0.5,
               init_inverse_temp=0.1, final_inverse_temp=0.9,
               ksizes=None, strides=(1, 1, 1, 1), rates=(1, 1, 1, 1), padding='SAME', in_size=None, conv=False,
               summaries=False):
    """
    Creates the TensorFlow graph for the EM routing between capsules. Takes in and outputs 2D grids of capsules so that
    it will work between convolutional layers, but it may be used for routing between non convolutional layers by using
    a 2D grid where the size of one dimension is 1 and taking kernel_rows/cols to be input_rows/cols
    :param in_vote: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, pose_size, pose_size] NOW SHOULD BE FULL PATCH TENSOR
    :param in_activation: Tensor with shape [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, 1, 1, 1]
    :param n_routing_iterations: Int number of iterations to run the EM algorithm for the routing
    :param init_beta_v:
    :param init_beta_a:
    :param init_inverse_temp: Initial value for the inverse temperature parameter used in the M-step
    :param final_inverse_temp: Final value (on last iteration) of the inverse temperature parameter for the M-step
    :param ksizes:
    :param strides: Strides used to extract patches for in_vote and in_activation
    :param rates: Rates used to extract patches for in_vote and in_activation
    :param padding: Padding used to extract patches for in_vote and in_activation
    :param in_size: Shape of the input image from which patches have been extracted. Can be inferred if None but ambiguous in some cases so best if explicit
    :param conv: bool whether this is routing between convolutional capsules or not
    :param summaries:
    :return: pose: Tensor with shape [batch_size, out_rows, out_cols, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, out_rows, out_cols, out_capsules]
    """
    with tf.variable_scope('em_routing'):
        # Get required shapes
        shape_list = tf_ops.get_shape_list(in_vote)
        batch_size = shape_list[0]
        if conv:
            kernel_size = ksizes[1:3]
        else:
            kernel_size = [shape_list[1], shape_list[2]]
        in_capsules = shape_list[3]
        out_size = [shape_list[4], shape_list[5]]
        out_capsules = shape_list[6]

        # Create R constant and initialise as uniform probability (stores the probability of each vote under each
        # Gaussian with mean corresponding to output pose)
        patches_shape = [batch_size, *kernel_size, in_capsules, *out_size, out_capsules, 1, 1]
        r = tf.divide(tf.ones(patches_shape, name="R"),
                      np.prod(out_size) * out_capsules)

        # Create beta parameters
        beta_v = tf.Variable(init_beta_v * tf.ones([1, 1, 1, 1, 1, 1, out_capsules, 1, 1]), name="beta_v")
        beta_a = tf.Variable(init_beta_a * tf.ones([1, 1, 1, 1, 1, 1, out_capsules, 1, 1]), name="beta_a")
        if summaries:
            tf.summary.histogram('beta_v', beta_v)
            tf.summary.histogram('beta_a', beta_a)

        # Initialise inverse temperature parameter and compute how much to increment by for each routing iteration
        inverse_temp = init_inverse_temp
        if n_routing_iterations > 1:
            inverse_temp_increment = (final_inverse_temp - init_inverse_temp) / (n_routing_iterations - 1)
        else:
            inverse_temp_increment = 0  # Cant increment anyway in this case

        """
        If we are doing routing between convolutional capsules we need to send the correct parameters to the e-step
        so that we can convert the a*p tensor to a sparse tensor and do a sparse reduce sum. Otherwise we would
        be computing the wrong sum since all the tensors of conv patches do not keep the patches in their original 
        position in the image but we need to take their original position into account when doing the sum. Example in 
        readme.
        """
        if conv:
            strides = get_correct_conv_param(strides)
            rates = get_correct_conv_param(rates)
            if in_size is None:
                in_size = conv_in_size(out_size, kernel_size, strides[1:3], rates[1:3], padding)
            dense_shape = tf_ops.fast_get_patches_full_shape(patches_shape, strides, rates, padding, in_size)
            dense_shape[6] = out_capsules
            dense_indices = tf_ops.get_dense_indices(patches_shape, in_size, strides, rates, padding)

            sparse_args = {'sparse': True,
                           'strides': strides,
                           'rates': rates,
                           'padding': padding,
                           'in_size': in_size,
                           'patches_shape': patches_shape,
                           'dense_shape': dense_shape,
                           'dense_indices': dense_indices}
        else:
            sparse_args = {'sparse': False}

        # Do routing iterations
        for routing_iteration in range(n_routing_iterations - 1):
            with tf.variable_scope("routing_iteration_{}".format(routing_iteration + 1)):
                # Do M-Step to get Gaussian means and standard deviations and update activations
                mean, variance, activation = m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp, conv,
                                                    summaries)

                # Do E-Step to update R
                r = e_step(mean, variance, activation, in_vote, logspace=True, summaries=summaries, **sparse_args)

                # Update inverse temp
                inverse_temp += inverse_temp_increment

        with tf.variable_scope("routing_iteration_{}".format(n_routing_iterations)):
            # Do M-Step to get Gaussian means and update activations
            mean, _, activation = m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp, conv, summaries)

        # Get rid of redundant dimensions
        pose = tf.squeeze(mean, [1, 2, 3])
        activation = tf.squeeze(activation, [1, 2, 3, 7, 8])

        return pose, activation


def spread_loss(in_activation, label, margin):
    """
    Creates the TensorFlow graph for the spread loss detailed in the paper
    :param in_activation: Tensor with shape [batch_size, n_classes]
    :param label: Tensor with shape [batch_size, n_classes] containing the one-hot class labels
    :param margin:
    :return: Tensor containing the scalar loss
    """
    with tf.variable_scope('spread_loss'):
        # TODO - make this conditional for efficiency if label is already float32?
        label = tf.cast(label, tf.float32)

        # Get activation of the correct class
        activation_target = tf.reduce_sum(tf.multiply(in_activation, label), axis=1, keep_dims=True)  # [batch_size, 1]

        # Get activations of incorrect classes
        # Subtracting label so that the value for a_t here will become a_t - 1 which will result in
        # (margin - (a_t - (a_t - 1))) <= 0 (since 0 < margin < 1) such that the loss for the a_t element will be 0.
        # Seems like a slightly inelegant solution but works
        activation_non_target = in_activation - label

        # Get margin loss for each incorrect class
        # [batch_size, n_classes - 1]
        l_i = tf.square(tf.maximum(0., margin - (activation_target - activation_non_target)))

        # Get total loss for each batch element
        # [batch_size]
        l = tf.reduce_sum(l_i, axis=1)

        # Take mean of total loss over batch
        spread_loss = tf.reduce_mean(l)

        return spread_loss