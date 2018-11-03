import tensorflow as tf
from layers import upsample_with_indices, conv_bn


def segnet(image, *, n_classes=7):
    max_unpooling=True
    with tf.variable_scope("segnet"):

        # encoder
        conv1_1 = conv_bn(image, filters=64, kernel_size=(3, 3), padding="same", name="conv1_1")
        conv1_2 = conv_bn(conv1_1, filters=64, kernel_size=(3, 3), padding="same", name="conv1_2")
        pool1, index1 = tf.nn.max_pool_with_argmax(input=conv1_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool1")

        conv2_1 = conv_bn(pool1, filters=128, kernel_size=(3, 3), padding="same", name="conv2_1")
        conv2_2 = conv_bn(conv2_1, filters=128, kernel_size=(3, 3), padding="same", name="conv2_2")
        pool2, index2 = tf.nn.max_pool_with_argmax(input=conv2_2, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool2")

        conv3_1 = conv_bn(pool2, filters=256, kernel_size=(3, 3), padding="same", name="conv3_1")
        conv3_2 = conv_bn(conv3_1, filters=256, kernel_size=(3, 3), padding="same", name="conv3_2")
        conv3_3 = conv_bn(conv3_2, filters=256, kernel_size=(3, 3), padding="same", name="conv3_3")
        pool3, index3 = tf.nn.max_pool_with_argmax(input=conv3_3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool3")

        dropout1 = tf.layers.Dropout(0.5, name="dropout1")(pool3)

        conv4_1 = conv_bn(dropout1, filters=512, kernel_size=(3, 3), padding="same", name="conv4_1")
        conv4_2 = conv_bn(conv4_1, filters=512, kernel_size=(3, 3), padding="same", name="conv4_2")
        conv4_3 = conv_bn(conv4_2, filters=512, kernel_size=(3, 3), padding="same", name="conv4_3")
        pool4, index4 = tf.nn.max_pool_with_argmax(input=conv4_3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool4")

        dropout2 = tf.layers.Dropout(0.5, name="dropout2")(pool4)

        conv5_1 = conv_bn(dropout2, filters=512, kernel_size=(3, 3), padding="same", name="conv5_1")
        conv5_2 = conv_bn(conv5_1, filters=512, kernel_size=(3, 3), padding="same", name="conv5_2")
        conv5_3 = conv_bn(conv5_2, filters=512, kernel_size=(3, 3), padding="same", name="conv5_3")
        pool5, index5 = tf.nn.max_pool_with_argmax(input=conv5_3, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME", name="max_pool5")

        dropout3 = tf.layers.Dropout(0.5, name="dropout3")(pool5)

        # decoder
        if max_unpooling:
            upsample1 = upsample_with_indices(dropout3, index5, name="upsample1")
        else:
            upsample1 = tf.layers.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name="up_conv1")(dropout3)
        conv6_1 = conv_bn(upsample1, filters=512, kernel_size=(3, 3), padding="same", name="conv6_1")
        conv6_2 = conv_bn(conv6_1, filters=512, kernel_size=(3, 3), padding="same", name="conv6_2")
        conv6_3 = conv_bn(conv6_2, filters=512, kernel_size=(3, 3), padding="same", name="conv6_3")

        dropout4 = tf.layers.Dropout(0.5, name="dropout4")(conv6_3)

        if max_unpooling:
            upsample2 = upsample_with_indices(dropout4, index4, name="upsample2")
        else:
            upsample2 = tf.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name="up_conv2")(dropout4)
        conv7_1 = conv_bn(upsample2, filters=256, kernel_size=(3, 3), padding="same", name="conv7_1")
        conv7_2 = conv_bn(conv7_1, filters=256, kernel_size=(3, 3), padding="same", name="conv7_2")
        conv7_3 = conv_bn(conv7_2, filters=256, kernel_size=(3, 3), padding="same", name="conv7_3")

        dropout5 = tf.layers.Dropout(0.5, name="dropout5")(conv7_3)

        if max_unpooling:
            upsample3 = upsample_with_indices(dropout5, index3, name="upsample3")
        else:
            upsample3 = tf.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name="up_conv3")(dropout5)
        conv8_1 = conv_bn(upsample3, filters=128, kernel_size=(3, 3), padding="same", name="conv8_1")
        conv8_2 = conv_bn(conv8_1, filters=128, kernel_size=(3, 3), padding="same", name="conv8_2")
        conv8_3 = conv_bn(conv8_2, filters=128, kernel_size=(3, 3), padding="same", name="conv8_3")

        dropout6 = tf.layers.Dropout(0.5, name="dropout6")(conv8_3)

        if max_unpooling:
            upsample4 = upsample_with_indices(dropout6, index2, name="upsample4")
        else:
            upsample4 = tf.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name="up_conv4")(dropout6)
        conv9_1 = conv_bn(upsample4, filters=64, kernel_size=(3, 3), padding="same", name="conv9_1")
        conv9_2 = conv_bn(conv9_1, filters=64, kernel_size=(3, 3), padding="same", name="conv9_2")

        if max_unpooling:
            upsample5 = upsample_with_indices(conv9_2, index1, name="upsample5")
        else:
            upsample5 = tf.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", name="up_conv5")(conv9_2)
        conv10_1 = conv_bn(upsample5, filters=32, kernel_size=(3, 3), padding="same", name="conv10_1")
        conv10_2 = conv_bn(conv10_1, filters=32, kernel_size=(3, 3), padding="same", name="conv10_2")

    final_conv = tf.layers.Conv2D(filters=n_classes, kernel_size=(1, 1), activation="softmax", name="classes")(conv10_2)

    return final_conv
