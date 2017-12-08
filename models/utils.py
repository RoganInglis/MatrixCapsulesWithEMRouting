import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import math


def conv_out_size(in_size, kernel_size, strides, padding):
    if padding is 'valid':
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


def expand_dims_nd(input_tensor, axis=None, name=None):
    """
    Extension of tf.expand_dims to multiple dimensions so that more than one dimension can be added at a time in the
    same way that more than one dimension can be squeezed at a time using tf.squeeze. This is very marginally faster
    than using tf.reshape
    :param input_tensor: Tensor to be expanded
    :param axis: None, Int, Tuple or List specifying which axes to be expanded using the same logic as tf.expand_dims
    :param name: Name for the op/ops
    :return:
    """
    axis = list(axis)
    if len(axis) == 1:
        input_tensor = tf.expand_dims(input_tensor, axis=axis[0], name=name)
    else:
        # Sort list such that we expand higher dimensions first unless expanding out the end
        axis_new = list(set(axis) - set(range(input_tensor.shape.ndims)))
        axis_existing = list(set(axis) - set(axis_new))
        axis_existing.sort(reverse=True)
        axis_new.sort()
        axis = [*axis_new, *axis_existing]
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
        conv_pose = extract_image_patches_nd(conv_pose, [1, *kernel_size, 1], [1, *strides, 1], padding=padding)

        # Tile over out_capsules: need to be multiplying the same input tensor by different weights for each out capsule
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], out_capsules, pose_size, pose_size]
        conv_pose = tf.tile(conv_pose, [1, 1, 1, 1, 1, 1, out_capsules, 1, 1])

        vote = tf.matmul(kernel, conv_pose)

        # Get patches from in_activation and expand dims
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1]]
        activation = extract_image_patches_nd(in_activation, [1, *kernel_size, 1], [1, *strides, 1], padding=padding)
        # [batch_size, kernel_size[0], kernel_size[1], in_capsules, out_size[0], out_size[1], 1, 1, 1]
        activation = expand_dims_nd(activation, [6, 7, 8])

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

        # Create matmul weights and tile over batch (as we need the same kernel to be multiplied by each batch element)
        weights = tf.Variable(tf.random_normal([1, in_rows, in_cols, in_capsules, 1, 1, out_capsules, pose_size, pose_size]),
                             name='weights')
        weights = tf.tile(weights, [batch_size, 1, 1, 1, 1, 1, 1, 1, 1])

        # Re-organise in_pose so performing matmul with kernel computes the required convolutional affine transform
        pose = expand_dims_nd(in_pose, [4, 5, 6])

        # Tile over out_capsules: need to be multiplying the same input tensor by different weights for each out capsule
        # [batch_size, in_rows, in_cols, in_capsules, 1, 1, out_capsules, pose_size, pose_size]
        pose = tf.tile(pose, [1, 1, 1, 1, 1, 1, out_capsules, 1, 1])

        vote = tf.matmul(weights, pose)

        # Expand dims of activation
        # [batch_size, in_rows, in_cols, in_capsules, 1, 1, 1, 1, 1]
        activation = expand_dims_nd(in_activation, [4, 5, 6, 7, 8])

        return vote, activation


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
        r = tf.multiply(r, in_activation)

        # Update means (out_poses)
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        r_reduce_sum = tf.reduce_sum(r, axis=[1, 2, 3], keep_dims=True)
        
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, pose_size, pose_size]
        mean = tf.divide(tf.reduce_sum(tf.multiply(r, in_vote), axis=[1, 2, 3], keep_dims=True),
                         r_reduce_sum)

        # Update variances (same shape as mean)
        variance = tf.divide(tf.reduce_sum(tf.multiply(r, tf.square(in_vote - mean)), axis=[1, 2, 3], keep_dims=True),
                             r_reduce_sum)

        # Compute cost (same shape as mean)
        cost_h = tf.multiply(tf.add(beta_v, tf.log(tf.sqrt(variance))), r_reduce_sum)

        # Compute new activations
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        activation = tf.nn.sigmoid(inverse_temp*(tf.subtract(beta_a, tf.reduce_sum(cost_h, axis=[-2, -1], keep_dims=True))))

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
        # Compute p: the probability density of each in_vote (data point)
        # [batch_size, 1, 1, 1, out_rows, out_cols, out_capsules, 1, 1]
        a = tf.divide(1, tf.sqrt(tf.reduce_prod(2*math.pi*variance, axis=[-2, -1], keep_dims=True)))
        # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
        b = -0.5*tf.reduce_sum(tf.divide(tf.square(tf.subtract(in_vote, mean)), variance), axis=[-2, -1], keep_dims=True)

        # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
        p = tf.multiply(a, tf.exp(b))

        # Compute updated r (assignment probability/responsibility)
        # [batch_size, kernel_rows, kernel_cols, in_capsules, out_rows, out_cols, out_capsules, 1, 1]
        r = tf.divide(tf.multiply(activation, p), tf.reduce_sum(tf.multiply(activation, p), axis=[4, 5, 6], keep_dims=True))  # TODO - double check reduce sum dimensions here; summing over out capsules? Check bottom of page 5 of paper

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
        beta_v = tf.Variable(tf.random_normal([1]), name="beta_v")
        beta_a = tf.Variable(tf.random_normal([1]), name="beta_a")

        # Initialise inverse temperature parameter and compute how much to increment by for each routing iteration
        inverse_temp = init_inverse_temp
        inverse_temp_increment = (final_inverse_temp - init_inverse_temp)/(n_routing_iterations - 1)

        # TODO - should we be stopping the gradient of the mean and/or activations here?

        # Do routing iterations
        for routing_iteration in range(n_routing_iterations):
            with tf.variable_scope("routing_iteration_{}".format(routing_iteration)):
                # Do M-Step to get Gaussian means and standard deviations and update activations
                mean, std_dev, activation = m_step(r, in_activation, in_vote, beta_v, beta_a, inverse_temp)

                # Do E-Step to update R (only if this is not the last iteration)
                if routing_iteration < n_routing_iterations - 1:
                    r = e_step(mean, std_dev, activation, in_vote)

                    # Update inverse temp
                    inverse_temp += inverse_temp_increment

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


def convcaps_layer(in_pose, in_activation, out_capsules, kernel_size, strides=1, padding='valid', n_routing_iterations=3,
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

        # TODO - need to sort out coordinate addition somewhere

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
        n_classes = in_activation.get_shape().as_list()[1]

        eye = tf.expand_dims(tf.eye(n_classes), axis=0)
        tiled_activation = tf.tile(tf.expand_dims(in_activation, 2), [1, 1, n_classes])

        activation_masked_positive = tf.multiply(eye, tiled_activation)
        activation_masked_negative = tf.multiply(1 - eye, tiled_activation)

        activation_diff = tf.reduce_sum(activation_masked_positive, axis=2)

        spread_loss = []
        # TODO - Implement this
        return spread_loss


def build_capsnetem_graph(placeholders, image_dim=784):
    """
    Builds the TensorFlow graph for the capsules model (named CapsNetEM here) presented in the paper 'Matrix Capsules
    with EM Routing'
    :param placeholders: Dict containing TensorFlow placeholders for the input image and labels under 'image' and 'label'
    :param image_dim: Int dimension of the flattened image (i.e. image_dim = image_height*image_width)
    :return: loss:
             predictions:
             accuracy:
             correct:
             summaries:
    """
    # PARAMETERS TODO - decide which (if any) should be fixed and which should be passed as arguments
    # ReLU Conv1
    relu_conv1_kernel_size = 5
    relu_conv1_filters = 32
    relu_conv1_stride = 2

    # PrimaryCaps
    primarycaps_out_capsules = 32
    pose_size = 4

    # ConvCaps1
    convcaps1_out_capsules = 32
    convcaps1_kernel_size = 3
    convcaps1_strides = 2
    convcaps1_padding = 'SAME'
    convcaps1_n_routing_iterations = 3
    convcaps1_init_inverse_temp = 0.1
    convcaps1_final_inverse_temp = 0.9

    # ConvCaps2
    convcaps2_out_capsules = 32
    convcaps2_kernel_size = 3
    convcaps2_strides = 1
    convcaps2_padding = 'SAME'
    convcaps2_n_routing_iterations = 3
    convcaps2_init_inverse_temp = 0.1
    convcaps2_final_inverse_temp = 0.9

    # Class Capsules
    classcaps_n_classes = 10
    classcaps_n_routing_iterations = 3
    classcaps_init_inverse_temp = 0.1
    classcaps_final_inverse_temp = 0.9

    # Spread Loss
    initial_margin = 0.2
    final_margin = 0.9
    margin = initial_margin  # TODO - use tf.train.polynomial_decay(initial_margin, global_step, decay_steps, final_margin)

    # Initalise summaries dict - using dict so that we can merge only select summaries; don't want image summaries all
    # the time
    summaries = {}
    summaries["general"] = []

    # Reshape flattened image tensor to 2D
    images = tf.reshape(placeholders['image'], [-1, 28, 28, 1])
    summaries['images'] = tf.summary.image('input_images', images)

    # Create ReLU Conv1 de-rendering layer
    with tf.variable_scope('relu_conv1'):
        relu_conv1_out = tf.layers.conv2d(images, relu_conv1_filters, relu_conv1_kernel_size, relu_conv1_stride,
                                          activation=tf.nn.relu)  # [batch_size, 12?, 12?, relu_conv1_filters]

    # Create PrimaryCaps layer
    primarycaps_pose, primarycaps_activation = primarycaps_layer(relu_conv1_out, primarycaps_out_capsules, pose_size)

    # Create ConvCaps1 layer
    convcaps1_pose, convcaps1_activation = convcaps_layer(primarycaps_pose, primarycaps_activation, convcaps1_out_capsules,
                                                          convcaps1_kernel_size, convcaps1_strides, convcaps1_padding,
                                                          convcaps1_n_routing_iterations, convcaps1_init_inverse_temp,
                                                          convcaps1_final_inverse_temp)

    # Create ConvCaps2 layer
    convcaps2_pose, convcaps2_activation = convcaps_layer(convcaps1_pose, convcaps1_activation, convcaps2_out_capsules,
                                                          convcaps2_kernel_size, convcaps2_strides, convcaps2_padding,
                                                          convcaps2_n_routing_iterations, convcaps2_init_inverse_temp,
                                                          convcaps2_final_inverse_temp)

    # Create Class Capsules layer
    classcaps_pose, classcaps_activation = classcaps_layer(convcaps2_pose, convcaps2_activation, classcaps_n_classes,
                                                           classcaps_n_routing_iterations, classcaps_init_inverse_temp,
                                                           classcaps_final_inverse_temp)

    # Create spread loss
    loss = spread_loss(classcaps_activation, placeholders['label'], margin)

    # TODO - Complete this (and update function params when known)
    predictions=accuracy=correct=summaries=None
    return loss, predictions, accuracy, correct, summaries


def save_mnist_as_image(mnist_batch, outdir, name="image"):
    # If outdir doesn't exist then create it
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for i, image in enumerate(tqdm(mnist_batch, desc="Saving images", leave=False, ncols=100)):
        image = np.squeeze(image)
        plt.imsave("{}/{}_{}.png".format(outdir, name, i), image)