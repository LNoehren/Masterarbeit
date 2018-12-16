import tensorflow as tf
from layers import spatial_dropout


def u_net(image, *, n_classes=7, trainable=True):
    """
    U-Net model structure.
    Number of parameters in Model: 34 513 735

    https://arxiv.org/pdf/1505.04597.pdf

    :param image: input tensor
    :param n_classes: number of classes
    :param trainable: whether the variables of the model should be trainable or fixed
    :return: the output tensor of the model. After Softmax is applied
    """
    with tf.variable_scope("u_net"):

        # encoder
        conv1_1 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv1_1")(image)
        conv1_2 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv1_2")(conv1_1)
        pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool1")(conv1_2)

        conv2_1 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv2_1")(pool1)
        conv2_2 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv2_2")(conv2_1)
        pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool2")(conv2_2)

        conv3_1 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv3_1")(pool2)
        conv3_2 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv3_2")(conv3_1)
        pool3 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool3")(conv3_2)

        conv4_1 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv4_1")(pool3)
        conv4_2 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv4_2")(conv4_1)
        pool4 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="max_pool4")(conv4_2)

        dropout1 = spatial_dropout(pool4, 0.5, name="dropout1")

        conv5_1 = tf.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv5_1")(dropout1)
        conv5_2 = tf.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv5_2")(conv5_1)

        dropout2 = spatial_dropout(conv5_2, 0.5, name="dropout2")

        # decoder
        up_conv1 = tf.layers.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", trainable=trainable, name="up_conv1")(dropout2)
        merge1 = tf.concat([up_conv1, conv4_2], axis=-1, name="merge1")

        conv6_1 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv6_1")(merge1)
        conv6_2 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv6_2")(conv6_1)

        up_conv2 = tf.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", trainable=trainable, name="up_conv2")(conv6_2)
        merge2 = tf.concat([up_conv2, conv3_2], axis=-1, name="merge2")

        conv7_1 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv7_1")(merge2)
        conv7_2 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv7_2")(conv7_1)

        up_conv3 = tf.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", trainable=trainable, name="up_conv3")(conv7_2)
        merge3 = tf.concat([up_conv3, conv2_2], axis=-1, name="merge3")

        conv8_1 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv8_1")(merge3)
        conv8_2 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv8_2")(conv8_1)

        up_conv4 = tf.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", trainable=trainable, name="up_conv4")(conv8_2)
        merge4 = tf.concat([up_conv4, conv1_2], axis=-1, name="merge4")

        conv9_1 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv9_1")(merge4)
        conv9_2 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu", trainable=trainable, name="conv9_2")(conv9_1)

        final_conv = tf.layers.Conv2D(filters=n_classes, kernel_size=(1, 1), activation="softmax", trainable=trainable, name="classes")(conv9_2)

    return final_conv
