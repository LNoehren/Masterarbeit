import tensorflow as tf
from layers import e_net_initializer_block, e_net_bottleneck, e_net_downsample, e_net_upsampling, asymmetric_e_net_bottleneck


def e_net(image, *, n_classes=7, trainable=True):
    """
    E-Net model structure.
    Number of parameters in Model: 405 115

    https://arxiv.org/pdf/1606.02147.pdf

    :param image: input tensor
    :param n_classes: number of classes
    :param trainable: whether the variables of the model should be trainable or fixed
    :return: the output tensor of the model. After Softmax is applied
    """
    with tf.variable_scope("e_net"):

        init_block = e_net_initializer_block(image, filters=16, trainable=trainable)

        # encoder
        bottleneck1_0, index1 = e_net_downsample(init_block, filters=64, dropout_rate=0.01, trainable=trainable, name="bottleneck1_0")
        bottleneck1_1 = e_net_bottleneck(bottleneck1_0, filters=64, dropout_rate=0.01, trainable=trainable, name="bottleneck1_1")
        bottleneck1_2 = e_net_bottleneck(bottleneck1_1, filters=64, dropout_rate=0.01, trainable=trainable, name="bottleneck1_2")
        bottleneck1_3 = e_net_bottleneck(bottleneck1_2, filters=64, dropout_rate=0.01, trainable=trainable, name="bottleneck1_3")
        bottleneck1_4 = e_net_bottleneck(bottleneck1_3, filters=64, dropout_rate=0.01, trainable=trainable, name="bottleneck1_4")

        bottleneck2_0, index2 = e_net_downsample(bottleneck1_4, filters=128, dropout_rate=0.1, trainable=trainable, name="bottleneck2_0")
        bottleneck2_1 = e_net_bottleneck(bottleneck2_0, filters=128, dropout_rate=0.1, trainable=trainable, name="bottleneck2_1")
        bottleneck2_2 = e_net_bottleneck(bottleneck2_1, filters=128, dropout_rate=0.1, trainable=trainable, dilation=2, name="bottleneck2_2")
        bottleneck2_3 = asymmetric_e_net_bottleneck(bottleneck2_2, filters=128, dropout_rate=0.1, trainable=trainable, name="bottleneck2_3")
        bottleneck2_4 = e_net_bottleneck(bottleneck2_3, filters=128, dropout_rate=0.1, trainable=trainable, dilation=4, name="bottleneck2_4")
        bottleneck2_5 = e_net_bottleneck(bottleneck2_4, filters=128, dropout_rate=0.1, trainable=trainable, name="bottleneck2_5")
        bottleneck2_6 = e_net_bottleneck(bottleneck2_5, filters=128, dropout_rate=0.1, trainable=trainable, dilation=8, name="bottleneck2_6")
        bottleneck2_7 = asymmetric_e_net_bottleneck(bottleneck2_6, filters=128, dropout_rate=0.1, trainable=trainable, name="bottleneck2_7")
        bottleneck2_8 = e_net_bottleneck(bottleneck2_7, filters=128, dropout_rate=0.1, trainable=trainable, dilation=16, name="bottleneck2_8")

        bottleneck3_1 = e_net_bottleneck(bottleneck2_8, filters=128, dropout_rate=0.1, trainable=trainable, name="bottleneck3_1")
        bottleneck3_2 = e_net_bottleneck(bottleneck3_1, filters=128, dropout_rate=0.1, trainable=trainable, dilation=2, name="bottleneck3_2")
        bottleneck3_3 = asymmetric_e_net_bottleneck(bottleneck3_2, filters=128, dropout_rate=0.1, trainable=trainable, name="bottleneck3_3")
        bottleneck3_4 = e_net_bottleneck(bottleneck3_3, filters=128, dropout_rate=0.1, trainable=trainable, name="bottleneck3_4")
        bottleneck3_5 = e_net_bottleneck(bottleneck3_4, filters=128, dropout_rate=0.1, trainable=trainable, dilation=4, name="bottleneck3_5")
        bottleneck3_6 = e_net_bottleneck(bottleneck3_5, filters=128, dropout_rate=0.1, trainable=trainable, dilation=8, name="bottleneck3_6")
        bottleneck3_7 = asymmetric_e_net_bottleneck(bottleneck3_6, filters=128, dropout_rate=0.1, trainable=trainable, name="bottleneck3_7")
        bottleneck3_8 = e_net_bottleneck(bottleneck3_7, filters=128, dropout_rate=0.1, trainable=trainable, dilation=16, name="bottleneck3_8")

        # decoder
        bottleneck4_0 = e_net_upsampling(bottleneck3_8, filters=64, dropout_rate=0.1, upsampling_indices=index2, trainable=trainable, name="bottleneck4_0")
        bottleneck4_1 = e_net_bottleneck(bottleneck4_0, filters=64, dropout_rate=0.1, trainable=trainable, name="bottleneck4_1")
        bottleneck4_2 = e_net_bottleneck(bottleneck4_1, filters=64, dropout_rate=0.1, trainable=trainable, name="bottleneck4_2")

        bottleneck5_0 = e_net_upsampling(bottleneck4_2, filters=16, dropout_rate=0.1, upsampling_indices=index1, trainable=trainable, name="bottleneck5_0")
        bottleneck5_1 = e_net_bottleneck(bottleneck5_0, filters=16, dropout_rate=0.1, trainable=trainable, name="bottleneck5_1")

        classes = tf.layers.Conv2DTranspose(filters=n_classes, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="softmax", trainable=trainable, name="classes")(bottleneck5_1)

    return classes
