import tensorflow as tf
from layers import separable_conv_bn, conv_bn


def deeplab_v3_plus(image, *, n_classes=7, trainable=True):
    """
    DeepLabV3+ Model structure. Xception Model is used as backbone with output stride=16. In the last two blocks of
    Xception the striding is replaced by atrous convolutions with Multi Grid = (1, 2, 4) and rates = (2, 4).

    https://arxiv.org/pdf/1802.02611.pdf

    :param image: input tensor
    :param n_classes: number of classes
    :param trainable: whether the variables of the model should be trainable or fixed
    :return: the output tensor of the Model
    """
    with tf.variable_scope("deeplab_v3_plus"):
        # Xception Model
        with tf.variable_scope("xception"):
            # entry flow
            conv0_1 = tf.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(2, 2), trainable=trainable, activation="relu", name="conv0_1")(image)
            conv0_2 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", trainable=trainable, activation="relu", name="conv0_2")(conv0_1)

            conv0_3 = separable_conv_bn(conv0_2, filters=128, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv0_3")
            conv0_4 = separable_conv_bn(conv0_3, filters=128, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv0_4")
            conv0_5 = separable_conv_bn(conv0_4, filters=128, kernel_size=(3, 3), padding="same", strides=(2, 2), trainable=trainable, name="conv0_5")
            skip_conv0 = tf.layers.Conv2D(filters=128, kernel_size=(1, 1), padding="same", strides=(2, 2), trainable=trainable, name="skip_conv1")(conv0_2)
            add0 = tf.add(skip_conv0, conv0_5, name="skip0_add")

            conv0_6 = separable_conv_bn(add0, filters=256, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv0_6")
            conv0_7 = separable_conv_bn(conv0_6, filters=256, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv0_7")
            conv0_8 = separable_conv_bn(conv0_7, filters=256, kernel_size=(3, 3), padding="same", strides=(2, 2), trainable=trainable, name="conv0_8")
            skip_conv1 = tf.layers.Conv2D(filters=256, kernel_size=(1, 1), padding="same", strides=(2, 2), trainable=trainable, name="skip_conv1")(add0)
            add1 = tf.add(skip_conv1, conv0_8, name="skip1_add")

            conv0_9 = separable_conv_bn(add1, filters=728, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv0_9")
            conv0_10 = separable_conv_bn(conv0_9, filters=728, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv0_10")
            conv0_11 = separable_conv_bn(conv0_10, filters=728, kernel_size=(3, 3), padding="same", strides=(2, 2), trainable=trainable, name="conv0_11")
            skip_conv2 = tf.layers.Conv2D(filters=728, kernel_size=(1, 1), padding="same", strides=(2, 2), trainable=trainable, name="skip_conv2")(add1)
            add2 = tf.add(skip_conv2, conv0_11, name="skip2_add")

            # middle flow
            def middle_flow_block(input, id):
                conv0 = separable_conv_bn(input, filters=728, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv{}_0".format(id))
                conv1 = separable_conv_bn(conv0, filters=728, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv{}_1".format(id))
                conv2 = separable_conv_bn(conv1, filters=728, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv{}_2".format(id))
                return tf.add(input, conv2, name="middle{}_add".format(id))

            middle1 = middle_flow_block(add2, 1)
            middle2 = middle_flow_block(middle1, 2)
            middle3 = middle_flow_block(middle2, 3)
            middle4 = middle_flow_block(middle3, 4)
            middle5 = middle_flow_block(middle4, 5)
            middle6 = middle_flow_block(middle5, 6)
            middle7 = middle_flow_block(middle6, 7)
            middle8 = middle_flow_block(middle7, 8)
            middle9 = middle_flow_block(middle8, 9)
            middle10 = middle_flow_block(middle9, 10)
            middle11 = middle_flow_block(middle10, 11)
            middle12 = middle_flow_block(middle11, 12)
            middle13 = middle_flow_block(middle12, 13)
            middle14 = middle_flow_block(middle13, 14)
            middle15 = middle_flow_block(middle14, 15)
            middle16 = middle_flow_block(middle15, 16)

            # exit_flow
            conv17_0 = separable_conv_bn(middle16, filters=728, kernel_size=(3, 3), dilation_rate=2, padding="same", trainable=trainable, name="conv17_0")
            conv17_1 = separable_conv_bn(conv17_0, filters=1024, kernel_size=(3, 3), dilation_rate=4, padding="same", trainable=trainable, name="conv17_1")
            conv17_2 = separable_conv_bn(conv17_1, filters=1024, kernel_size=(3, 3), dilation_rate=8, padding="same", trainable=trainable, name="conv17_2")
            skip_conv3 = tf.layers.Conv2D(filters=1024, kernel_size=(1, 1), dilation_rate=8, padding="same", trainable=trainable, activation="relu", name="skip_conv3")(middle16)
            add3 = tf.add(skip_conv3, conv17_2, name="skip3_add")

            conv17_3 = separable_conv_bn(add3, filters=1536, kernel_size=(3, 3), dilation_rate=4, padding="same", trainable=trainable, name="conv17_3")
            conv17_4 = separable_conv_bn(conv17_3, filters=1536, kernel_size=(3, 3), dilation_rate=8, padding="same", trainable=trainable, name="conv17_4")
            conv17_5 = separable_conv_bn(conv17_4, filters=2048, kernel_size=(3, 3), dilation_rate=16, padding="same", trainable=trainable, name="conv17_5")

        # atrous spatial pyramid pooling
        with tf.variable_scope("aspp"):
            aspp_conv0 = conv_bn(conv17_5, filters=256, kernel_size=(1, 1), padding="same", trainable=trainable, name="conv0")
            aspp_conv1 = conv_bn(conv17_5, filters=256, kernel_size=(3, 3), dilation_rate=6, padding="same", trainable=trainable, name="conv1")
            aspp_conv2 = conv_bn(conv17_5, filters=256, kernel_size=(3, 3), dilation_rate=12, padding="same", trainable=trainable, name="conv2")
            aspp_conv3 = conv_bn(conv17_5, filters=256, kernel_size=(3, 3), dilation_rate=18, padding="same", trainable=trainable, name="conv3")

            aspp_pool = tf.reduce_mean(conv17_5, axis=[1, 2], keepdims=True, name="global_avg_pool")
            image_level_conv = tf.layers.Conv2D(filters=256, kernel_size=(1, 1), padding="same", trainable=trainable, activation="relu", name="image_level_conv")(aspp_pool)
            image_level_upsample = tf.image.resize_images(image_level_conv, [conv17_5.get_shape().as_list()[1], conv17_5.get_shape().as_list()[2]])

            aspp_concat = tf.concat([aspp_conv0, aspp_conv1, aspp_conv2, aspp_conv3, image_level_upsample], axis=-1, name="concat")
            aspp_conv4 = conv_bn(aspp_concat, filters=256, kernel_size=(1, 1), padding="same", trainable=trainable, name="conv4")

        # decoder
        new_size = [aspp_conv4.get_shape().as_list()[1]*4, aspp_conv4.get_shape().as_list()[2]*4]
        upsample0 = tf.image.resize_images(aspp_conv4, new_size)
        decoder_conv0 = tf.layers.Conv2D(filters=256, kernel_size=(1, 1), padding="same", trainable=trainable, activation="relu", name="decoder_conv0")(conv0_7)
        decoder_concat = tf.concat([upsample0, decoder_conv0], axis=-1, name="decoder_concat")

        decoder_conv1 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", trainable=trainable, activation="relu", name="decoder_conv1")(decoder_concat)
        decoder_conv2 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", trainable=trainable, activation="relu", name="decoder_conv2")(decoder_conv1)

        new_size = [decoder_conv2.get_shape().as_list()[1] * 4, decoder_conv2.get_shape().as_list()[2] * 4]
        upsample1 = tf.image.resize_images(decoder_conv2, new_size)

        classes = tf.layers.Conv2D(filters=n_classes, kernel_size=(3, 3), padding="same", trainable=trainable, activation="softmax", name="classes")(upsample1)

    return classes
