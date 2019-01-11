import tensorflow as tf
from layers import e_net_initializer_block, non_bt_1d, conv_bn


def erfnet(image, *, n_classes=7, trainable=True):
    """
    ERFNet model structure.
    Number of parameters in Model: 2 057 379

    https://ieeexplore.ieee.org/document/8063438

    :param image: input tensor
    :param n_classes: number of classes
    :param trainable: whether the variables of the model should be trainable or fixed
    :return: the output tensor of the model. After Softmax is applied
    """
    with tf.variable_scope("erfnet"):

        # encoder
        downsample1 = e_net_initializer_block(image, filters=16, trainable=trainable, name="downsample1")

        downsample2 = e_net_initializer_block(downsample1, filters=64, trainable=trainable, name="downsample2")
        non_bt_1d_1_1 = non_bt_1d(downsample2, 64, trainable=trainable, name="non_bt_1d_1_1")
        non_bt_1d_1_2 = non_bt_1d(non_bt_1d_1_1, 64, trainable=trainable, name="non_bt_1d_1_2")
        non_bt_1d_1_3 = non_bt_1d(non_bt_1d_1_2, 64, trainable=trainable, name="non_bt_1d_1_3")
        non_bt_1d_1_4 = non_bt_1d(non_bt_1d_1_3, 64, trainable=trainable, name="non_bt_1d_1_4")
        non_bt_1d_1_5 = non_bt_1d(non_bt_1d_1_4, 64, trainable=trainable, name="non_bt_1d_1_5")

        downsample3 = e_net_initializer_block(non_bt_1d_1_5, filters=128, trainable=trainable, name="downsample3")
        non_bt_1d_2_1 = non_bt_1d(downsample3, 128, dilation_rate=2, trainable=trainable, name="non_bt_1d_2_1")
        non_bt_1d_2_2 = non_bt_1d(non_bt_1d_2_1, 128, dilation_rate=4, trainable=trainable, name="non_bt_1d_2_2")
        non_bt_1d_2_3 = non_bt_1d(non_bt_1d_2_2, 128, dilation_rate=8, trainable=trainable, name="non_bt_1d_2_3")
        non_bt_1d_2_4 = non_bt_1d(non_bt_1d_2_3, 128, dilation_rate=16, trainable=trainable, name="non_bt_1d_2_4")
        non_bt_1d_2_5 = non_bt_1d(non_bt_1d_2_4, 128, dilation_rate=2, trainable=trainable, name="non_bt_1d_2_5")
        non_bt_1d_2_6 = non_bt_1d(non_bt_1d_2_5, 128, dilation_rate=4, trainable=trainable, name="non_bt_1d_2_6")
        non_bt_1d_2_7 = non_bt_1d(non_bt_1d_2_6, 128, dilation_rate=8, trainable=trainable, name="non_bt_1d_2_7")
        non_bt_1d_2_8 = non_bt_1d(non_bt_1d_2_7, 128, dilation_rate=16, trainable=trainable, name="non_bt_1d_2_8")


        # # atrous spatial pyramid pooling
        # with tf.variable_scope("aspp"):
        #     aspp_conv0 = conv_bn(non_bt_1d_2_8, filters=256, kernel_size=(1, 1), padding="same", trainable=trainable,
        #                          name="conv0")
        #     aspp_conv1 = conv_bn(non_bt_1d_2_8, filters=256, kernel_size=(3, 3), dilation_rate=6, padding="same",
        #                          trainable=trainable, name="conv1")
        #     aspp_conv2 = conv_bn(non_bt_1d_2_8, filters=256, kernel_size=(3, 3), dilation_rate=12, padding="same",
        #                          trainable=trainable, name="conv2")
        #     aspp_conv3 = conv_bn(non_bt_1d_2_8, filters=256, kernel_size=(3, 3), dilation_rate=18, padding="same",
        #                          trainable=trainable, name="conv3")
        #
        #     aspp_pool = tf.reduce_mean(non_bt_1d_2_8, axis=[1, 2], keepdims=True, name="global_avg_pool")
        #     image_level_conv = conv_bn(aspp_pool, filters=256, kernel_size=(1, 1), padding="same", trainable=trainable,
        #                                name="image_level_conv")
        #     image_level_upsample = tf.image.resize_images(image_level_conv, [non_bt_1d_2_8.get_shape().as_list()[1],
        #                                                                      non_bt_1d_2_8.get_shape().as_list()[2]])
        #
        #     aspp_concat = tf.concat([aspp_conv0, aspp_conv1, aspp_conv2, aspp_conv3, image_level_upsample], axis=-1,
        #                             name="concat")
        #     aspp_conv4 = conv_bn(aspp_concat, filters=256, kernel_size=(1, 1), padding="same", trainable=trainable,
        #                          name="conv4")


        # decoder
        up_conv1 = tf.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", trainable=trainable, name="up_conv1")(non_bt_1d_2_8)
        non_bt_1d_3_1 = non_bt_1d(up_conv1, 64, trainable=trainable, name="non_bt_1d_3_1")
        non_bt_1d_3_2 = non_bt_1d(non_bt_1d_3_1, 64, trainable=trainable, name="non_bt_1d_3_2")

        up_conv2 = tf.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", trainable=trainable, name="up_conv2")(non_bt_1d_3_2)
        non_bt_1d_4_1 = non_bt_1d(up_conv2, 16, trainable=trainable, name="non_bt_1d_4_1")
        non_bt_1d_4_2 = non_bt_1d(non_bt_1d_4_1, 16, trainable=trainable, name="non_bt_1d_4_2")

        up_conv3 = tf.layers.Conv2DTranspose(filters=n_classes, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="softmax", trainable=trainable, name="classes")(non_bt_1d_4_2)

    return up_conv3
