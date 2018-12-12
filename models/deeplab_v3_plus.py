import tensorflow as tf
from layers import separable_conv_bn, conv_bn, spatial_dropout, non_bt_1d


def xception(image, trainable=True):
    with tf.variable_scope("xception"):
        # entry flow
        conv0_1 = tf.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", strides=(2, 2), trainable=trainable, activation="relu", name="conv0_1")(image)
        conv0_2 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", trainable=trainable, activation="relu", name="conv0_2")(conv0_1)

        def entry_flow_block(input, filters, id):
            conv0 = separable_conv_bn(input, filters=filters, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv0_{}_1".format(id))
            conv1 = separable_conv_bn(conv0, filters=filters, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv0_{}_2".format(id))
            conv2 = separable_conv_bn(conv1, filters=filters, kernel_size=(3, 3), padding="same", strides=(2, 2), trainable=trainable, name="conv0_{}_3".format(id))
            skip_conv = tf.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding="same", strides=(2, 2), trainable=trainable, name="skip_conv{}".format(id))(input)
            return tf.add(skip_conv, conv2, name="skip{}_add".format(id)), conv1

        entry1, _ = entry_flow_block(conv0_2, 128, 1)
        entry2, high_level_features = entry_flow_block(entry1, 256, 2)
        entry3, _ = entry_flow_block(entry2, 728, 3)

        # middle flow
        def middle_flow_block(input, id):
            conv0 = separable_conv_bn(input, filters=728, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv{}_0".format(id))
            conv1 = separable_conv_bn(conv0, filters=728, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv{}_1".format(id))
            conv2 = separable_conv_bn(conv1, filters=728, kernel_size=(3, 3), padding="same", trainable=trainable,  name="conv{}_2".format(id))
            # drop = spatial_dropout(conv2, 0.3, "dropout_{}".format(id))
            add = tf.add(input, conv2, name="middle{}_add".format(id))
            return add

        middle1 = middle_flow_block(entry3, 1)
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
        skip_conv4 = tf.layers.Conv2D(filters=1024, kernel_size=(1, 1), padding="same", trainable=trainable, name="skip_conv4")(middle16)
        add4 = tf.add(skip_conv4, conv17_2, name="skip4_add")

        conv17_3 = separable_conv_bn(add4, filters=1536, kernel_size=(3, 3), dilation_rate=4, padding="same", trainable=trainable, name="conv17_3")
        conv17_4 = separable_conv_bn(conv17_3, filters=1536, kernel_size=(3, 3), dilation_rate=8, padding="same", trainable=trainable, name="conv17_4")
        conv17_5 = separable_conv_bn(conv17_4, filters=2048, kernel_size=(3, 3), dilation_rate=16, padding="same", trainable=trainable, name="conv17_5")
        return conv17_5, high_level_features


def resnet101(image, trainable=True):
    with tf.variable_scope("resnet101"):
        def bottleneck_block(input, output_filter_size, id, dilation_rate=1):
            small_filter = output_filter_size // 4
            bt_conv1 = tf.layers.Conv2D(filters=small_filter, kernel_size=(1, 1), padding="same", trainable=trainable,
                                        dilation_rate=dilation_rate, activation="relu", name="conv{}_1".format(id))(input)
            bt_conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(3, 3), padding="same", trainable=trainable,
                                        dilation_rate=dilation_rate, activation="relu", name="conv{}_2".format(id))(bt_conv1)
            bt_conv3 = tf.layers.Conv2D(filters=output_filter_size, kernel_size=(1, 1), padding="same", trainable=trainable,
                                        dilation_rate=dilation_rate, name="conv{}_3".format(id))(bt_conv2)

            padding_size = (output_filter_size - input.get_shape().as_list()[-1])
            padded_input = tf.pad(input, tf.constant([[0, 0], [0, 0], [0, 0], [0, padding_size]]), "CONSTANT")

            add = tf.add(padded_input, bt_conv3, name="add{}".format(id))
            return tf.nn.relu(add, name="block{}_act".format(id))

        def downsample_block(input, output_filter_size, id):
            small_filter = output_filter_size // 4
            bt_conv1 = tf.layers.Conv2D(filters=small_filter, kernel_size=(1, 1), strides=(2, 2), padding="same",
                                        trainable=trainable, activation="relu", name="conv{}_1".format(id))(input)
            bt_conv2 = tf.layers.Conv2D(filters=small_filter, kernel_size=(3, 3), padding="same", trainable=trainable,
                                        activation="relu", name="conv{}_2".format(id))(bt_conv1)
            bt_conv3 = tf.layers.Conv2D(filters=output_filter_size, kernel_size=(1, 1), padding="same",
                                        trainable=trainable, name="conv{}_3".format(id))(bt_conv2)
            skip_conv = tf.layers.Conv2D(filters=output_filter_size, kernel_size=(1, 1), strides=(2, 2), padding="same",
                                         trainable=trainable, name="skip_conv{}".format(id))(input)
            add = tf.add(skip_conv, bt_conv3, name="add{}".format(id))
            return tf.nn.relu(add, name="block{}_act".format(id))

        # replace 7x7 conv at beginning by 3 3x3 conv
        #conv1 = tf.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", trainable=trainable, activation="relu", name="conv1")(image)
        conv1 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", trainable=trainable, activation="relu", name="conv1_1")(image)
        conv1 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", trainable=trainable, activation="relu", name="conv1_2")(conv1)
        conv1 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", trainable=trainable, activation="relu", name="conv1_3")(conv1)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=(2, 2), padding="same", name="pool1")

        # block2_1 = bottleneck_block(pool1, 256, "2_1")
        # block2_2 = bottleneck_block(block2_1, 256, "2_2")
        # block2_3 = bottleneck_block(block2_2, 256, "2_3")
        #
        # block3_1 = downsample_block(block2_3, 512, "3_1")
        # block3_2 = bottleneck_block(block3_1, 512, "3_2")
        # block3_3 = bottleneck_block(block3_2, 512, "3_3")
        # block3_4 = bottleneck_block(block3_3, 512, "3_4")
        #
        # block4_1 = downsample_block(block3_4, 1024, "4_1")
        # block4_2 = bottleneck_block(block4_1, 1024, "4_2")
        # block4_3 = bottleneck_block(block4_2, 1024, "4_3")
        # block4_4 = bottleneck_block(block4_3, 1024, "4_4")
        # block4_5 = bottleneck_block(block4_4, 1024, "4_5")
        # block4_6 = bottleneck_block(block4_5, 1024, "4_6")
        # block4_7 = bottleneck_block(block4_6, 1024, "4_7")
        # block4_8 = bottleneck_block(block4_7, 1024, "4_8")
        # block4_9 = bottleneck_block(block4_8, 1024, "4_9")
        # block4_10 = bottleneck_block(block4_9, 1024, "4_10")
        # block4_11 = bottleneck_block(block4_10, 1024, "4_11")
        # block4_12 = bottleneck_block(block4_11, 1024, "4_12")
        # block4_13 = bottleneck_block(block4_12, 1024, "4_13")
        # block4_14 = bottleneck_block(block4_13, 1024, "4_14")
        # block4_15 = bottleneck_block(block4_14, 1024, "4_15")
        # block4_16 = bottleneck_block(block4_15, 1024, "4_16")
        # block4_17 = bottleneck_block(block4_16, 1024, "4_17")
        # block4_18 = bottleneck_block(block4_17, 1024, "4_18")
        # block4_19 = bottleneck_block(block4_18, 1024, "4_19")
        # block4_20 = bottleneck_block(block4_19, 1024, "4_20")
        # block4_21 = bottleneck_block(block4_20, 1024, "4_21")
        # block4_22 = bottleneck_block(block4_21, 1024, "4_22")
        # block4_23 = bottleneck_block(block4_22, 1024, "4_23")
        #
        # block5_1 = bottleneck_block(block4_23, 2048, "5_1", dilation_rate=2)
        # block5_2 = bottleneck_block(block5_1, 2048, "5_2", dilation_rate=4)
        # block5_3 = bottleneck_block(block5_2, 2048, "5_3", dilation_rate=8)

        block2_1 = non_bt_1d(pool1, 64, trainable=True, name="block2_1")
        block2_2 = non_bt_1d(block2_1, 64, trainable=True, name="block2_2")
        block2_3 = non_bt_1d(block2_2, 64, trainable=True, name="block2_3")

        block3_1 = downsample_block(block2_3, 128, "3_1")
        block3_2 = non_bt_1d(block3_1, 128, trainable=True, name="block3_2")
        block3_3 = non_bt_1d(block3_2, 128, trainable=True, name="block3_3")
        block3_4 = non_bt_1d(block3_3, 128, trainable=True, name="block3_4")

        block4_1 = downsample_block(block3_4, 256, "4_1")
        block4_2 = non_bt_1d(block4_1, 256, trainable=True, name="block4_2")
        block4_3 = non_bt_1d(block4_2, 256, trainable=True, name="block4_3")
        block4_4 = non_bt_1d(block4_3, 256, trainable=True, name="block4_4")
        block4_5 = non_bt_1d(block4_4, 256, trainable=True, name="block4_5")
        block4_6 = non_bt_1d(block4_5, 256, trainable=True, name="block4_6")
        block4_7 = non_bt_1d(block4_6, 256, trainable=True, name="block4_7")
        block4_8 = non_bt_1d(block4_7, 256, trainable=True, name="block4_8")
        block4_9 = non_bt_1d(block4_8, 256, trainable=True, name="block4_9")
        block4_10 = non_bt_1d(block4_9, 256, trainable=True, name="block4_10")
        block4_11 = non_bt_1d(block4_10, 256, trainable=True, name="block4_11")
        block4_12 = non_bt_1d(block4_11, 256, trainable=True, name="block4_12")
        block4_13 = non_bt_1d(block4_12, 256, trainable=True, name="block4_13")
        block4_14 = non_bt_1d(block4_13, 256, trainable=True, name="block4_14")
        block4_15 = non_bt_1d(block4_14, 256, trainable=True, name="block4_15")
        block4_16 = non_bt_1d(block4_15, 256, trainable=True, name="block4_16")
        block4_17 = non_bt_1d(block4_16, 256, trainable=True, name="block4_17")
        block4_18 = non_bt_1d(block4_17, 256, trainable=True, name="block4_18")
        block4_19 = non_bt_1d(block4_18, 256, trainable=True, name="block4_19")
        block4_20 = non_bt_1d(block4_19, 256, trainable=True, name="block4_20")
        block4_21 = non_bt_1d(block4_20, 256, trainable=True, name="block4_21")
        block4_22 = non_bt_1d(block4_21, 256, trainable=True, name="block4_22")
        block4_23 = non_bt_1d(block4_22, 256, trainable=True, name="block4_23")

        block4_23 = tf.layers.Conv2D(filters=512, kernel_size=(1, 1), padding="same", trainable=trainable, activation="relu", name="filter_up2")(block4_23)

        block5_1 = non_bt_1d(block4_23, 512, trainable=True, name="block5_1", dilation_rate=2)
        block5_2 = non_bt_1d(block5_1, 512, trainable=True, name="block5_2", dilation_rate=4)
        block5_3 = non_bt_1d(block5_2, 512, trainable=True, name="block5_3", dilation_rate=8)

        return block5_3, block2_3


def deeplab_v3_plus(image, *, n_classes=7, trainable=True):
    """
    DeepLabV3+ Model structure. Xception Model is used as backbone with output stride=16. In the last two blocks of
    Xception the striding is replaced by atrous convolutions with Multi Grid = (1, 2, 4) and rates = (2, 4).
    (The number of filters in many convolutions have been reduced compared to the paper for faster runtime and to enable
    bigger batch sizes on the available hardware.)
    Spatial dropout with rate 0.3 has been added in the middle flow blocks of the Xception model to reduce overfitting.
    Number of parameters in Model (ResNet101): 57 173 175
    Number of parameters in original Model (Xception): 54 664 399
    (Number of parameters in reduced Model: 26 125 367)

    https://arxiv.org/pdf/1802.02611.pdf

    :param image: input tensor
    :param n_classes: number of classes
    :param trainable: whether the variables of the model should be trainable or fixed
    :return: the output tensor of the Model
    """
    with tf.variable_scope("deeplab_v3_plus"):
        # Backbone Model
        backbone_output, high_level_features = resnet101(image, trainable)

        # atrous spatial pyramid pooling
        with tf.variable_scope("aspp"):
            aspp_conv0 = conv_bn(backbone_output, filters=256, kernel_size=(1, 1), padding="same", trainable=trainable, name="conv0")
            aspp_conv1 = conv_bn(backbone_output, filters=256, kernel_size=(3, 3), dilation_rate=6, padding="same", trainable=trainable, name="conv1")
            aspp_conv2 = conv_bn(backbone_output, filters=256, kernel_size=(3, 3), dilation_rate=12, padding="same", trainable=trainable, name="conv2")
            aspp_conv3 = conv_bn(backbone_output, filters=256, kernel_size=(3, 3), dilation_rate=18, padding="same", trainable=trainable, name="conv3")

            aspp_pool = tf.reduce_mean(backbone_output, axis=[1, 2], keepdims=True, name="global_avg_pool")
            image_level_conv = conv_bn(aspp_pool, filters=256, kernel_size=(1, 1), padding="same", trainable=trainable, name="image_level_conv")
            image_level_upsample = tf.image.resize_images(image_level_conv, [backbone_output.get_shape().as_list()[1], backbone_output.get_shape().as_list()[2]])

            aspp_concat = tf.concat([aspp_conv0, aspp_conv1, aspp_conv2, aspp_conv3, image_level_upsample], axis=-1, name="concat")
            aspp_conv4 = conv_bn(aspp_concat, filters=256, kernel_size=(1, 1), padding="same", trainable=trainable, name="conv4")

        # decoder
        new_size = [aspp_conv4.get_shape().as_list()[1] * 4, aspp_conv4.get_shape().as_list()[2] * 4]
        upsample0 = tf.image.resize_images(aspp_conv4, new_size, align_corners=True)
        decoder_conv0 = tf.layers.Conv2D(filters=48, kernel_size=(1, 1), padding="same", trainable=trainable, activation="relu", name="decoder_conv0")(high_level_features)
        decoder_concat = tf.concat([upsample0, decoder_conv0], axis=-1, name="decoder_concat")

        decoder_conv1 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", trainable=trainable, activation="relu", name="decoder_conv1")(decoder_concat)
        decoder_conv2 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", trainable=trainable, activation="relu", name="decoder_conv2")(decoder_conv1)

        new_size = [decoder_conv2.get_shape().as_list()[1] * 4, decoder_conv2.get_shape().as_list()[2] * 4]
        upsample1 = tf.image.resize_images(decoder_conv2, new_size, align_corners=True)

        classes = tf.layers.Conv2D(filters=n_classes, kernel_size=(1, 1), padding="same", trainable=trainable, activation="softmax", name="classes")(upsample1)

    return classes
