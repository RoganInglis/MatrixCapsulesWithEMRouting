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
        p = [kernel_size[0] - 1, kernel_size[1] - 1]

    out_size = [(in_size[0] + p[0] - kernel_size[0]) // strides[0] + 1,
                (in_size[1] + p[1] - kernel_size[1]) // strides[1] + 1]
    return out_size


def get_conv_caps_slices(conv_pose, kernel_size, pad, padding, in_height, in_width, strides):
    """
    slice_list = [conv_pose[:, i:i + kernel_size[0], j:j + kernel_size[1], :, :, :, :, :]
                  for i in range(pad[0][0], in_height + pad[0][0] - 1, strides[0])
                  for j in range(pad[1][0], in_width + pad[1][0] - 1, strides[1])]
    """
    if padding is 'same':
        width_max = in_width
        height_max = in_height
    else:
        width_max = in_width - kernel_size[1] + 1
        height_max = in_height - kernel_size[0] + 1
    slice_list = [conv_pose[:, i:i + kernel_size[0], j:j + kernel_size[1], :, :, :, :, :]
                  for j in range(0, width_max, strides[1])
                  for i in range(0, height_max, strides[0])]
    return slice_list


def convcaps_affine_transform(in_pose, n_capsules, kernel_size, strides, padding):
    """
    Creates the TensorFlow graph for the convolutional affine transform performed prior to routing in a convolutional
    capsule layer
    :param in_pose: Tensor with shape [batch_size, in_height, in_width, in_capsules, pose_size, pose_size]
    :param n_capsules: Int specifying the number of output capsules
    :param kernel_size: Int, Tuple or List specifying the size of the convolution kernel (assuming square kernel if int)
    :param strides: Int, Tuple or List specifying the strides for the convolution (assuming equal over dimensions if int)
    :param padding: 'valid' or 'same' specifying padding to use in the same way as tf.nn.conv2d
    :return: vote: Tensor with shape [batch_size, kernel_height, kernel_width, in_capsules, out_height, out_width, n_capsules, pose_size, pose_size] ##### TODO - this may not be correct, double check further into implementation #####
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
        in_height = shape_list[1]
        in_width = shape_list[2]
        in_capsules = shape_list[3]
        pose_size = shape_list[4]

        # Compute output im grid size
        out_size = conv_out_size([in_height, in_width], kernel_size, strides, padding)

        # Create convolutional matmul kernel and tile over batch
        kernel = tf.Variable(tf.random_normal([1, *kernel_size, in_capsules, *out_size, n_capsules, pose_size, pose_size]),
                             name='kernel')
        kernel = tf.tile(kernel, [batch_size, 1, 1, 1, 1, 1, 1, 1, 1])

        # Re-organise in_pose so performing matmul with kernel computes the required convolutional affine transform
        conv_pose = tf.reshape(in_pose, [batch_size, in_height, in_width, in_capsules, 1, 1, pose_size, pose_size])

        # Pad pose if required
        pad = [[int(math.floor((kernel_size[0] - 1) / 2)), int(math.ceil((kernel_size[0] - 1) / 2))],
               [int(math.floor((kernel_size[1] - 1) / 2)), int(math.ceil((kernel_size[1] - 1) / 2))]]
        if padding is 'same':
            paddings = tf.constant([[0, 0], [pad[0][0], pad[0][1]], [pad[1][0], pad[1][1]], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
            conv_pose = tf.pad(conv_pose, paddings)

        conv_pose_list = get_conv_caps_slices(conv_pose, )

        vote = tf.matmul(kernel, conv_pose)
        # TODO - Implement this
        return vote


def caps_affine_transform(in_pose, n_capsules):
    with tf.variable_scope('caps_affine_transform'):
        # TODO - Implement this
        pass


def m_step():
    with tf.variable_scope('m_step'):
        # TODO - Implement this
        pass


def e_step():
    with tf.variable_scope('e_step'):
        # TODO - Implement this
        pass


def em_routing(in_vote, in_activation, n_routing_iterations):
    with tf.variable_scope('em_routing'):
        em_routing_out = []
        # TODO - Implement this
        return em_routing_out


def primarycaps_layer(input_tensor, n_capsules, pose_size):
    """
    Creates the TensorFlow graph for the PrimaryCaps layer described in 'Matrix Capsules with EM Routing'
    :param input_tensor: Tensor with shape [batch_size, height, width, n_filters] (batch_size, 12?, 12?, 32) in paper
    :param n_capsules: Number of capsules (for each pixel)
    :param pose_size: Size of the capsule pose matrices (i.e. pose matrix will be pose_size x pose_size)
    :return: pose: Tensor with shape [batch_size, in_height, in_width, n_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, height, width, n_capsules]
    """
    with tf.variable_scope('PrimaryCaps'):
        # Get required shape values
        batch_size = tf.shape(input_tensor)[0]
        shape_list = input_tensor.get_shape().as_list()
        in_height = shape_list[1]
        in_width = shape_list[2]
        in_channels = shape_list[3]

        # Affine transform to create capsule pose matrices and activations
        # Create weights and tile them over batch in preparation for matmul op as we need to use the same weights for
        # each element in the batch
        weights = tf.Variable(tf.random_normal([1, in_height, in_width, n_capsules, in_channels, (pose_size ** 2 + 1)]),
                              name='weights')
        weights = tf.tile(weights, [batch_size, 1, 1, 1, 1])

        # Expand input tensor for matmul op and tile input over n_capsules for matmul op as we need to multiply the
        # input by separate weights for each output capsule
        input_tensor = tf.reshape(input_tensor, [batch_size, in_height, in_width, 1, 1, in_channels])  # [batch_size, in_height, in_width, 1, 1, in_channels]
        input_tensor = tf.tile(input_tensor, [1, 1, 1, n_capsules, 1, 1])  # [batch_size, in_height, in_width, n_capsules, 1, in_channels]

        # Do matmul to get flattened primarycaps pose matrices and then reshape so pose matrices are square
        pose_activation = tf.matmul(input_tensor, weights)  # [batch_size, in_height, in_width, n_capsules, 1, (pose_size**2 + 1)]

        # Get pose
        pose = pose_activation[:, :, :, :, :, :pose_size ** 2]  # [batch_size, in_height, in_width, n_capsules, 1, pose_size**2]
        pose = tf.reshape(pose, [batch_size, in_height, in_width, n_capsules, pose_size, pose_size])

        # Get activation
        activation = pose_activation[:, :, :, :, :, pose_size ** 2]  # [batch_size, in_height, in_width, n_capsules, 1, 1]
        activation = tf.reshape(activation, [batch_size, in_height, in_width, n_capsules])  # [batch_size, in_height, in_width, n_capsules]
        activation = tf.nn.sigmoid(activation)

        return pose, activation


def convcaps_layer(in_pose, in_activation, n_capsules, kernel_size, strides=1, padding='valid', n_routing_iterations=3):
    """
    Creates the TensorFlow graph for a convolutional capsule layer as specified in 'Matrix Capsules with EM Routing'
    :param in_pose: Tensor with shape [batch_size, in_height, in_width, n_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, in_height, in_width, n_capsules]
    :param n_capsules: Int number of capsules in the layer
    :param kernel_size: Int, Tuple or List specifying the size of the convolution kernel (assuming square kernel if int)
    :param strides: Int, Tuple or List specifying the strides for the convolution (assuming equal over dimensions if int)
    :param padding: 'valid' or 'same' specifying padding to use in the same way as tf.nn.conv2d
    :param n_routing_iterations: Number of iterations to use for the EM dynamic routing procedure
    :return: pose: Tensor with shape [batch_size, out_height, out_width, n_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, out_height, out_width, n_capsules]
    """
    with tf.variable_scope('ConvCaps'):
        # Pose convolutional affine transform
        in_vote = convcaps_affine_transform(in_pose, n_capsules, kernel_size, strides, padding)

        # EM Routing
        pose, activation = em_routing(in_vote, in_activation, n_routing_iterations)

        return pose, activation


def classcaps_layer(in_pose, in_activation, n_classes):
    """
    Creates the TensorFlow graph for the class capsules layer
    :param in_pose: Tensor with shape [batch_size, in_height, in_width, n_capsules, pose_size, pose_size]
    :param in_activation:
    :param n_classes:
    :return:
    """
    with tf.variable_scope('ClassCaps'):
        classcaps_out = []
        # TODO - Implement this
        return classcaps_out


def spread_loss(input_tensor):
    with tf.variable_scope('spread_loss'):
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
    primarycaps_n_capsules = 32
    pose_size = 4

    # ConvCaps1
    convcaps1_n_capsules = 32
    convcaps1_kernel_size = 3
    convcaps1_strides = 2
    convcaps1_padding = 'valid'
    convcaps1_n_routing_iterations = 3

    # ConvCaps2
    convcaps2_n_capsules = 32
    convcaps2_kernel_size = 3
    convcaps2_strides = 1
    convcaps2_padding = 'valid'
    convcaps2_n_routing_iterations = 3

    # Class Capsules
    n_classes = 10

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
    primarycaps_pose, primarycaps_activation = primarycaps_layer(relu_conv1_out, primarycaps_n_capsules, pose_size)

    # Create ConvCaps1 layer
    convcaps1_pose, convcaps1_activation = convcaps_layer(primarycaps_pose, primarycaps_activation, convcaps1_n_capsules,
                                                          convcaps1_kernel_size, convcaps1_strides, convcaps1_padding,
                                                          convcaps1_n_routing_iterations)

    # Create ConvCaps2 layer
    convcaps2_pose, convcaps2_activation = convcaps_layer(convcaps1_pose, convcaps1_activation, convcaps2_n_capsules,
                                                          convcaps2_kernel_size, convcaps2_strides, convcaps2_padding,
                                                          convcaps2_n_routing_iterations)

    # Create Class Capsules layer
    classcaps_out = classcaps_layer(convcaps2_pose, convcaps2_activation, n_classes)

    # Create spread loss
    loss = spread_loss(classcaps_out)

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