import tensorflow as tf

from models.capsule_ops import convcaps_affine_transform, em_routing, caps_affine_transform
from models.utils import get_correct_ksizes_strides_rates


def primarycaps_layer(input_tensor, out_capsules, pose_size, summaries=False):
    """
    Creates the TensorFlow graph for the PrimaryCaps layer described in 'Matrix Capsules with EM Routing'
    :param input_tensor: Tensor with shape [batch_size, height, width, n_filters] (batch_size, 12?, 12?, 32) in paper
    :param out_capsules: Number of capsules (for each pixel)
    :param pose_size: Size of the capsule pose matrices (i.e. pose matrix will be pose_size x pose_size)
    :param summaries:
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
        weights = tf.Variable(tf.truncated_normal([1, 1, 1, out_capsules, in_channels, (pose_size ** 2 + 1)],
                                                  stddev=0.5), name='weights')
        if summaries:
            tf.summary.histogram('weights', weights)

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
                   n_routing_iterations=3, init_beta_v=1., init_beta_a=-0.5, init_inverse_temp=0.1, final_inverse_temp=0.9,
                   summaries=False):
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
    :param init_beta_v: Initial value for the beta_v bias parameter
    :param init_beta_a: Initial value for the beta_a bias parameter
    :param init_inverse_temp: Scalar initial value for the inverse temperature parameter used for EM routing
    :param final_inverse_temp: Scalar final value for the inverse temperature parameter used for EM routing
    :param summaries:
    :return: pose: Tensor with shape [batch_size, out_rows, out_cols, out_capsules, pose_size, pose_size]
             activation: Tensor with shape [batch_size, out_rows, out_cols, out_capsules]
    """
    ksizes, strides, rates = get_correct_ksizes_strides_rates(kernel_size, strides, rates)
    with tf.variable_scope('ConvCaps'):
        in_size = in_pose.get_shape().as_list()[1:3]

        # Pose convolutional affine transform
        in_vote, in_activation = convcaps_affine_transform(in_pose, in_activation, out_capsules, ksizes, strides,
                                                           rates, padding, summaries=summaries)

        # EM Routing
        pose, activation = em_routing(in_vote, in_activation, n_routing_iterations, init_beta_v, init_beta_a,
                                      init_inverse_temp, final_inverse_temp, ksizes, strides, rates, padding, in_size,
                                      conv=True, summaries=summaries)

        return pose, activation


def classcaps_layer(in_pose, in_activation, n_classes, n_routing_iterations=3, init_beta_v=1., init_beta_a=-0.5,
                    init_inverse_temp=0.1, final_inverse_temp=0.9, summaries=False):
    """
    Creates the TensorFlow graph for the class capsules layer
    :param in_pose: Tensor with shape [batch_size, in_rows, in_cols, in_capsules, pose_size, pose_size]
    :param in_activation: Tensor with shape [batch_size, in_rows, in_cols, in_capsules]
    :param n_classes: Int, the number of classes (the number out output capsules)
    :param n_routing_iterations: Number of iterations to use for the EM dynamic routing procedure
    :param init_beta_V:
    :param init_beta_a:
    :param init_inverse_temp: Scalar initial value for the inverse temperature parameter used for EM routing
    :param final_inverse_temp: Scalar final value for the inverse temperature parameter used for EM routing
    :param summaries:
    :return:
    """
    with tf.variable_scope('ClassCaps'):
        # Pose affine transform
        # in_vote: Tensor with shape[batch_size, in_rows, in_cols, in_capsules, 1, 1, out_capsules, pose_size, pose_size]
        # in_activation: Tensor with shape[batch_size, in_rows, in_cols, in_capsules, 1, 1, 1, 1, 1]
        in_vote, in_activation = caps_affine_transform(in_pose, in_activation, n_classes, summaries=summaries)

        # EM Routing
        pose, activation = em_routing(in_vote, in_activation, n_routing_iterations, init_beta_v, init_beta_a,
                                      init_inverse_temp, final_inverse_temp, summaries=summaries)

        # Remove redundant dimensions introduced by going from 2D to 0D in space
        pose = tf.squeeze(pose, [1, 2])  # [batch_size, n_classes, pose_size, pose_size]
        activation = tf.squeeze(activation, [1, 2])  # [batch_size, n_classes]

        return pose, activation