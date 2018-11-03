import tensorflow as tf
import numpy as np
from utils import parametric_relu


# klappt anscheinend noch nicht richtig bei batch size > 1
def upsample_with_indices(values, indices, filters=0, name="upsampling"):
    """
    max-unpooling layer for segnet. Creates a sparse tensor with shape (values[0], values[1]*2, values[2]*2, values[3])

    :param values: Input Tensor
    :param indices: Indices of max pooling
    :param name: Name of the Operation
    :return: Sparse Tensor containing values at indices
    """
    with tf.variable_scope(name):
        flat_val = tf.reshape(values, shape=[-1])
        flat_ind = tf.cast(tf.reshape(indices, shape=[-1]), tf.int32)
        flat_shape = tf.shape(flat_val) * 4

        flat_result = tf.sparse_to_dense(flat_ind, flat_shape, flat_val, validate_indices=False)

        out_shape = [tf.shape(values)[0], values.get_shape().as_list()[1] * 2, values.get_shape().as_list()[2] * 2, values.get_shape().as_list()[3]]
        result = tf.reshape(flat_result, out_shape)

        if filters == 0:
            filters = out_shape[-1]

        result = tf.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", trainable=False,
                                  kernel_initializer=bilinear_initializer(3, filters), use_bias=False)(result)
        return result


def bilinear_initializer(kernel_size, num_channels):
    """
    Bilinear kernel initializer for convolution layers

    :param kernel_size: kernel size of the convolution
    :param num_channels: number of channles/filters of the convolution
    :return: tf constant initializer for bilinear upconvolution
    """
    bilinear_kernel = np.zeros([kernel_size, kernel_size], dtype=np.float32)
    scale_factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(kernel_size):
        for y in range(kernel_size):
            bilinear_kernel[x, y] = (1 - abs(x - center) / scale_factor) * (1 - abs(y - center) / scale_factor)
    weights = np.zeros((kernel_size, kernel_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = bilinear_kernel

    return tf.constant_initializer(value=weights, dtype=tf.float32)


def bottleneck(input, filters, dropout_rate, downsampling=False, upsampling_indices=None, dilation=1, asymmetric=False, name="bottleneck"):
    """
    bottleneck block for e-net.

    :param input: input Tensor
    :param filters: number of filters of the Convolutions
    :param dropout_rate: dropout rate for the dropout layer
    :param downsampling: whether or not the block should perform downsampling
    :param dilation: dilation rate of the convolution
    :param asymmetric: whether the central convolution should be symmetric (3x3) or two asymmetric convolutions (5x1, 1x5)
    :param name: name of the block
    :return: output of the bottleneck block
    """
    with tf.variable_scope(name):
        small_filter = int(filters/4)

        if downsampling:
            conv1 = tf.layers.Conv2D(filters=small_filter, kernel_size=(2, 2), strides=(2, 2), use_bias=False, padding="same", name="conv1")(input)
            conv1 = parametric_relu(conv1, name + "_conv1_prelu")

            input, index = tf.nn.max_pool_with_argmax(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool")
            padding_size = (filters - input.get_shape().as_list()[-1])
            input = tf.pad(input, tf.constant([[0, 0], [0, 0], [0, 0], [0, padding_size]]), "CONSTANT")

        elif upsampling_indices is not None:
            conv1 = tf.layers.Conv2DTranspose(filters=small_filter, kernel_size=(2, 2), strides=(2, 2), use_bias=False, padding="same", name="conv1")(input)
            conv1 = parametric_relu(conv1, name + "_conv1_prelu")

            input = upsample_with_indices(input, upsampling_indices, filters=filters)
            input = tf.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", use_bias=False)(input)

        else:
            conv1 = tf.layers.Conv2D(filters=small_filter, kernel_size=(1, 1), use_bias=False, padding="same", name="conv1")(input)
            conv1 = parametric_relu(conv1, name + "_conv1_prelu")
        bn1 = tf.layers.BatchNormalization(name="bn1")(conv1)

        if asymmetric:
            conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(5, 1), use_bias=False, padding="same", dilation_rate=dilation, name="conv2")(bn1)
            conv2 = parametric_relu(conv2, name + "_conv2_prelu1")
            conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(1, 5), use_bias=False, padding="same", dilation_rate=dilation, name="conv2")(conv2)
            conv2 = parametric_relu(conv2, name + "_conv2_prelu2")
        else:
            conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(3, 3), use_bias=False, padding="same", dilation_rate=dilation, name="conv2")(bn1)
            conv2 = parametric_relu(conv2, name + "_conv2_prelu")

        bn2 = tf.layers.BatchNormalization(name="bn2")(conv2)
        conv3 = tf.layers.Conv2D(filters=filters, kernel_size=(1, 1), use_bias=False, padding="same", name="conv3")(bn2)

        dropout = tf.layers.Dropout(dropout_rate, name="dropout")(conv3)

        out = input + dropout
        out = parametric_relu(out, name + "_out_prelu")

        if downsampling:
            return out, index

        return out


def e_net_initializer_block(image, filters, name="initializer"):
    """
    initializer block for e-net

    :param image: input Tensor
    :param filters: total number of output filters of the block
    :param name: name of the block
    :return: output tensor of the block
    """
    with tf.variable_scope(name):
        in_filters = image.get_shape().as_list()[-1]
        conv = tf.layers.Conv2D(filters=filters-in_filters, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name="conv")(image)
        pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool")(image)
        concat = tf.concat([conv, pool], -1, name="concat")

        return concat


def non_bt_1d(input, filters, dilation_rate=1, name="non_bt_1D"):
    """
    non-bottleneck-1D block for erfnet

    :param input: input Tensor
    :param filters: number of filters of the block
    :param dilation_rate: dilation rate for second pair of convolutions
    :param name: name of the block
    :return: output tensor of the block
    """
    with tf.variable_scope(name):
        conv1 = tf.layers.Conv2D(filters=filters, kernel_size=(3, 1), padding="same", activation="relu", name="conv1")(input)
        conv2 = tf.layers.Conv2D(filters=filters, kernel_size=(1, 3), padding="same", activation="relu", name="conv2")(conv1)
        conv3 = tf.layers.Conv2D(filters=filters, kernel_size=(3, 1), padding="same", dilation_rate=dilation_rate, activation="relu", name="conv3")(conv2)
        conv4 = tf.layers.Conv2D(filters=filters, kernel_size=(1, 3), padding="same", dilation_rate=dilation_rate, activation="relu", name="conv4")(conv3)
        dropout = tf.layers.Dropout(0.3, name="dropout")(conv4)

        result = tf.nn.relu(input + dropout, name="relu")
        return result


def conv_bn(input, name, **kwargs):
    """
    convolution layer, batch normalization layer, relu layer

    :param input: input tensor
    :param name: name of the block
    :param kwargs: args for the convolution layer
    :return: output of the relu layer
    """
    with tf.variable_scope(name):
        conv = tf.layers.Conv2D(**kwargs)(input)
        bn = tf.layers.BatchNormalization()(conv)
        act = tf.nn.relu(bn)

        return act
