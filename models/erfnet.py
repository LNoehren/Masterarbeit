import tensorflow as tf
from layers import e_net_initializer_block, non_bt_1d


def erfnet(image, *, n_classes=7, trainable=True):
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

        # decoder
        up_conv1 = tf.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", trainable=trainable, name="up_conv1")(non_bt_1d_2_8)
        non_bt_1d_3_1 = non_bt_1d(up_conv1, 64, trainable=trainable, name="non_bt_1d_3_1")
        non_bt_1d_3_2 = non_bt_1d(non_bt_1d_3_1, 64, trainable=trainable, name="non_bt_1d_3_2")

        up_conv2 = tf.layers.Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", trainable=trainable, name="up_conv2")(non_bt_1d_3_2)
        non_bt_1d_4_1 = non_bt_1d(up_conv2, 16, trainable=trainable, name="non_bt_1d_4_1")
        non_bt_1d_4_2 = non_bt_1d(non_bt_1d_4_1, 16, trainable=trainable, name="non_bt_1d_4_2")

        up_conv3 = tf.layers.Conv2DTranspose(filters=n_classes, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="softmax", trainable=trainable, name="up_conv2")(non_bt_1d_4_2)

    return up_conv3
