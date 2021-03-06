import tensorflow as tf
import numpy as np


def max_unpooling(values, indices, strides, name="max_unpooling"):
    """
    max-unpooling layer for SegNet. Creates a sparse tensor with shape values.shape * strides

    :param values: Input Tensor
    :param indices: Indices of max pooling
    :param strides: strides that were used in the corresponding max_pooling layer
    :param name: Name of the Operation
    :return: Sparse Tensor containing values at indices
    """
    with tf.variable_scope(name):
        v_shape = values.get_shape().as_list()
        indices = tf.cast(indices, tf.int32)
        out_shape = [tf.shape(values)[0] * strides[0],
                     v_shape[1] * strides[1],
                     v_shape[2] * strides[2],
                     v_shape[3] * strides[3]]

        i = tf.constant(1)
        m = tf.zeros([1] + v_shape[1:], dtype=tf.int32)
        cond = lambda i, m: i < out_shape[0]
        body = lambda i, m: [i+1, tf.concat([m, tf.fill([1] + v_shape[1:], i)], axis=0)]
        _, b = tf.while_loop(cond, body, [i, m], shape_invariants=[i.get_shape(), tf.TensorShape([None] + v_shape[1:])])
        y = indices // (out_shape[2]*out_shape[3])
        x = indices % (out_shape[2]*out_shape[3]) // out_shape[3]
        c = indices % out_shape[3]
        formatted_indices = tf.transpose(tf.stack([b, y, x, c]), (1, 2, 3, 4, 0))

        result = tf.scatter_nd(formatted_indices, values, out_shape)

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


def e_net_bottleneck(input, filters, dropout_rate, dilation=1, trainable=True, name="bottleneck"):
    """
    Standard bottleneck block for E-Net. The dilation rate is only used for the central convolution.

    :param input: Input Tensor
    :param filters: Number of filters for the convolutions
    :param dropout_rate: dropout rate for the spatial dropout layer
    :param dilation: dilation rate for the central convolution
    :param trainable: whether all variables should be trainable or fixed
    :param name: name of the block
    :return: output tensor of the block
    """
    with tf.variable_scope(name):
        small_filter = filters // 4
        conv1 = tf.layers.Conv2D(filters=small_filter, kernel_size=(1, 1), use_bias=False, padding="same", trainable=trainable, name="conv1")(input)
        conv1 = parametric_relu(conv1, trainable=trainable, name=name + "_conv1_prelu")
        bn1 = tf.layers.BatchNormalization(trainable=trainable, name="bn1")(conv1)

        conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(3, 3), padding="same", dilation_rate=dilation, trainable=trainable, name="conv2")(bn1)
        conv2 = parametric_relu(conv2, trainable=trainable, name=name + "_conv2_prelu")
        bn2 = tf.layers.BatchNormalization(trainable=trainable, name="bn2")(conv2)

        conv3 = tf.layers.Conv2D(filters=filters, kernel_size=(1, 1), use_bias=False, padding="same", trainable=trainable, name="conv3")(bn2)

        dropout = spatial_dropout(conv3, dropout_rate)

        out = input + dropout
        out = parametric_relu(out, trainable=trainable, name=name + "_out_prelu")

        return out


def asymmetric_e_net_bottleneck(input, filters, dropout_rate, dilation=1, trainable=True, name="bottleneck"):
    """
    Asymmetric bottleneck block for E-Net. The central convolution is replaced by a 5x1 and a 1x5 Convolution.

    :param input: Input Tensor
    :param filters: Number of filters for the convolutions
    :param dropout_rate: dropout rate for the spatial dropout layer
    :param dilation: dilation rate for the central convolution
    :param trainable: whether all variables should be trainable or fixed
    :param name: name of the block
    :return: output tensor of the block
    """
    with tf.variable_scope(name):
        small_filter = filters // 4
        conv1 = tf.layers.Conv2D(filters=small_filter, kernel_size=(1, 1), use_bias=False, padding="same", trainable=trainable, name="conv1")(input)
        conv1 = parametric_relu(conv1, trainable=trainable, name=name + "_conv1_prelu")
        bn1 = tf.layers.BatchNormalization(trainable=trainable, name="bn1")(conv1)

        conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(5, 1), padding="same", dilation_rate=dilation, trainable=trainable, name="conv2")(bn1)
        conv2 = parametric_relu(conv2, trainable=trainable, name=name + "_conv2_prelu1")
        conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(1, 5), padding="same", dilation_rate=dilation, trainable=trainable, name="conv2")(conv2)
        conv2 = parametric_relu(conv2, trainable=trainable, name=name + "_conv2_prelu2")
        bn2 = tf.layers.BatchNormalization(trainable=trainable, name="bn2")(conv2)

        conv3 = tf.layers.Conv2D(filters=filters, kernel_size=(1, 1), use_bias=False, padding="same", trainable=trainable, name="conv3")(bn2)

        dropout = spatial_dropout(conv3, dropout_rate)

        out = input + dropout
        out = parametric_relu(out, trainable=trainable, name=name + "_out_prelu")

        return out


def e_net_downsample(input, filters, dropout_rate, dilation=1, trainable=True, name="bottleneck"):
    """
    Downsampling bottleneck block for E-Net. Performs max pooling in the skip connections and returns the indices.

    :param input: Input Tensor
    :param filters: Number of filters for the convolutions
    :param dropout_rate: dropout rate for the spatial dropout layer
    :param dilation: dilation rate for the central convolution
    :param trainable: whether all variables should be trainable or fixed
    :param name: name of the block
    :return: output tensor of the block, max pooling indices
    """
    with tf.variable_scope(name):
        small_filter = filters // 4
        conv1 = tf.layers.Conv2D(filters=small_filter, kernel_size=(2, 2), strides=(2, 2), use_bias=False, padding="same", trainable=trainable, name="conv1")(input)

        input, index = tf.nn.max_pool_with_argmax(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool")
        padding_size = (filters - input.get_shape().as_list()[-1])
        input = tf.pad(input, tf.constant([[0, 0], [0, 0], [0, 0], [0, padding_size]]), "CONSTANT")
        conv1 = parametric_relu(conv1, trainable=trainable, name=name + "_conv1_prelu")
        bn1 = tf.layers.BatchNormalization(trainable=trainable, name="bn1")(conv1)

        conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(3, 3), padding="same", dilation_rate=dilation, trainable=trainable, name="conv2")(bn1)
        conv2 = parametric_relu(conv2, trainable=trainable, name=name + "_conv2_prelu")
        bn2 = tf.layers.BatchNormalization(trainable=trainable, name="bn2")(conv2)

        conv3 = tf.layers.Conv2D(filters=filters, kernel_size=(1, 1), use_bias=False, padding="same", trainable=trainable, name="conv3")(bn2)

        dropout = spatial_dropout(conv3, dropout_rate)

        out = input + dropout
        out = parametric_relu(out, trainable=trainable, name=name + "_out_prelu")

        return out, index


def e_net_upsampling(input, filters, dropout_rate, upsampling_indices, dilation=1, trainable=True, name="bottleneck"):
    """
    Upsampling bottleneck block for E-Net. Performs max unpooling.

    :param input: Input Tensor
    :param filters: Number of filters for the convolutions
    :param dropout_rate: dropout rate for the spatial dropout layer
    :param upsampling_indices: max pooling indices of the corresponding downsampling block
    :param dilation: dilation rate for the central convolution
    :param trainable: whether all variables should be trainable or fixed
    :param name: name of the block
    :return: output tensor of the block
    """
    with tf.variable_scope(name):
        small_filter = filters // 4
        conv1 = tf.layers.Conv2DTranspose(filters=small_filter, kernel_size=(2, 2), strides=(2, 2), use_bias=False, padding="same", trainable=trainable, name="conv1")(input)

        input = tf.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding="same", trainable=trainable, use_bias=False)(input)
        input = max_unpooling(input, upsampling_indices, strides=(1, 2, 2, 1))
        input = tf.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding="same", trainable=trainable, use_bias=False)(input)
        conv1 = parametric_relu(conv1, trainable=trainable, name=name + "_conv1_prelu")
        bn1 = tf.layers.BatchNormalization(trainable=trainable, name="bn1")(conv1)

        conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(3, 3), padding="same", dilation_rate=dilation, trainable=trainable, name="conv2")(bn1)
        conv2 = parametric_relu(conv2, trainable=trainable, name=name + "_conv2_prelu")
        bn2 = tf.layers.BatchNormalization(trainable=trainable, name="bn2")(conv2)

        conv3 = tf.layers.Conv2D(filters=filters, kernel_size=(1, 1), use_bias=False, padding="same", trainable=trainable, name="conv3")(bn2)

        dropout = spatial_dropout(conv3, dropout_rate)

        out = input + dropout
        out = parametric_relu(out, trainable=trainable, name=name + "_out_prelu")

        return out


def e_net_initializer_block(image, filters, trainable=True, name="initializer"):
    """
    initializer block for e-net

    :param image: input Tensor
    :param filters: total number of output filters of the block
    :param trainable: whether all variables should be trainable or fixed
    :param name: name of the block
    :return: output tensor of the block
    """
    with tf.variable_scope(name):
        in_filters = image.get_shape().as_list()[-1]
        conv = tf.layers.Conv2D(filters=filters-in_filters, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", trainable=trainable, name="conv")(image)
        pool = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool")(image)
        concat = tf.concat([conv, pool], -1, name="concat")

        return concat


def non_bt_1d(input, filters, dilation_rate=1, trainable=True, name="non_bt_1D"):
    """
    non-bottleneck-1D block for erfnet

    :param input: input Tensor
    :param filters: number of filters of the block
    :param dilation_rate: dilation rate for second pair of convolutions
    :param trainable: whether all variables should be trainable or fixed
    :param name: name of the block
    :return: output tensor of the block
    """
    with tf.variable_scope(name):
        conv1 = tf.layers.Conv2D(filters=filters, kernel_size=(3, 1), padding="same", activation="relu", trainable=trainable, name="conv1")(input)
        conv2 = tf.layers.Conv2D(filters=filters, kernel_size=(1, 3), padding="same", activation="relu", trainable=trainable, name="conv2")(conv1)
        conv3 = tf.layers.Conv2D(filters=filters, kernel_size=(3, 1), padding="same", dilation_rate=dilation_rate, activation="relu", trainable=trainable, name="conv3")(conv2)
        conv4 = tf.layers.Conv2D(filters=filters, kernel_size=(1, 3), padding="same", dilation_rate=dilation_rate, activation="relu", trainable=trainable, name="conv4")(conv3)
        dropout = spatial_dropout(conv4, 0.3, name="dropout")

        result = tf.nn.relu(input + dropout, name="relu")
        return result


def conv_bn(input, name, trainable=True, **kwargs):
    """
    convolution layer, batch normalization layer, relu layer in a row.

    :param input: input tensor
    :param name: name of the block
    :param trainable: whether all variables should be trainable or fixed
    :param kwargs: args for the convolution layer
    :return: output of the relu layer
    """
    with tf.variable_scope(name):
        conv = tf.layers.Conv2D(trainable=trainable, **kwargs)(input)
        bn = tf.layers.BatchNormalization(trainable=trainable)(conv)
        act = tf.nn.relu(bn)

        return act


def separable_conv_bn(input, name, trainable=True, **kwargs):
    """
    separable convolution layer, batch normalization layer, relu layer in a row.

    :param input: input tensor
    :param name: name of the block
    :param trainable: whether all variables should be trainable or fixed
    :param kwargs: args for the convolution layer
    :return: output of the relu layer
    """
    with tf.variable_scope(name):
        conv = tf.layers.separable_conv2d(inputs=input, trainable=trainable, **kwargs)
        bn = tf.layers.BatchNormalization(trainable=trainable)(conv)
        act = tf.nn.relu(bn)

        return act


def spatial_dropout(input, dropout_rate, name="dropout"):
    """
    spatial Dropout layer. Drops whole filters instead of single values.

    :param input: input tensor
    :param dropout_rate: percentage of filters that should bedropped
    :param name: name of the layer
    :return: output of the dropout layer
    """
    return tf.layers.Dropout(dropout_rate, noise_shape=[1, tf.shape(input)[1], tf.shape(input)[2], 1], name=name)(input)


def parametric_relu(x, trainable=True, name="prelu"):
    """
    prelu activation function. Like relu, but has a trainable weight for negative values. If weight is 1 prelu acts
    like linear activation, if weight is 0 it acts like relu.

    :param x: input for the activation function
    :param trainable: whether the weight should be trainable or fixed
    :param name: name for the weight. This should be a Unique name, unless the variable should be shared with
                 other prelu layers
    :return: result of the activation function
    """
    alpha = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(0.0),
                            dtype=tf.float32, trainable=trainable)

    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

