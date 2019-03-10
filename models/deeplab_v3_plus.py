import tensorflow as tf
from layers import separable_conv_bn, conv_bn


def resnet101_original(image, trainable=True):
    with tf.variable_scope("resnet101_original"):
        def residual_block(input, filter_size, id, init_scaling=1.0, dilation_rate=1):
            small_filter = filter_size // 4
            conv1 = tf.layers.Conv2D(small_filter, kernel_size=(1, 1), padding="same", dilation_rate=dilation_rate, trainable=trainable, activation="relu",
                                     kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                     bias_initializer=tf.initializers.variance_scaling(init_scaling),
                                     name="conv_{}_1".format(id))(input)
            conv2 = tf.layers.Conv2D(small_filter, kernel_size=(3, 3), padding="same", dilation_rate=dilation_rate, trainable=trainable, activation="relu",
                                     kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                     bias_initializer=tf.initializers.variance_scaling(init_scaling),
                                     name="conv_{}_2".format(id))(conv1)
            conv3 = tf.layers.Conv2D(filter_size, kernel_size=(1, 1), padding="same", dilation_rate=dilation_rate, trainable=trainable,
                                     kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                     bias_initializer=tf.initializers.variance_scaling(init_scaling),
                                     name="conv_{}_3".format(id))(conv2)
            return tf.nn.relu(conv3 + input)

        def downsample_block(input, filter_size, id, init_scaling=1.0):
            small_filter = filter_size // 4
            shortcut = tf.layers.Conv2D(filter_size, kernel_size=(1, 1), strides=(2, 2), padding="same", trainable=trainable,
                                        kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                        bias_initializer=tf.initializers.variance_scaling(init_scaling),
                                        name="shortcut_connection_{}".format(id))(input)
            conv1 = tf.layers.Conv2D(small_filter, kernel_size=(1, 1), strides=(2, 2), padding="same", trainable=trainable, activation="relu",
                                     kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                     bias_initializer=tf.initializers.variance_scaling(init_scaling),
                                     name="conv_{}_1".format(id))(input)
            conv2 = tf.layers.Conv2D(small_filter, kernel_size=(3, 3), padding="same", trainable=trainable, activation="relu",
                                     kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                     bias_initializer=tf.initializers.variance_scaling(init_scaling),
                                     name="conv_{}_2".format(id))(conv1)
            conv3 = tf.layers.Conv2D(filter_size, kernel_size=(1, 1), padding="same", trainable=trainable,
                                     kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                     bias_initializer=tf.initializers.variance_scaling(init_scaling),
                                     name="conv_{}_3".format(id))(conv2)

            return tf.nn.relu(conv3 + shortcut)

        scale_factor = 1.0

        conv1 = tf.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding="same", trainable=trainable, activation="relu",
                                 kernel_initializer=tf.initializers.variance_scaling(scale_factor),
                                 bias_initializer=tf.initializers.variance_scaling(scale_factor),
                                 name="conv1")(image)

        scale_factor *= 0.75

        block2 = tf.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding="same", trainable=trainable, name="pool1")(conv1)
        block2 = tf.pad(block2, paddings=tf.constant([[0, 0], [0, 0], [0, 0], [96, 96]]))
        for i in range(3):
            block2 = residual_block(block2, 256, "2_{}".format(i), init_scaling=scale_factor)

        scale_factor *= 0.75

        block3 = downsample_block(block2, 512, "3_0", init_scaling=scale_factor)
        for i in range(3):
            block3 = residual_block(block3, 512, "3_{}".format(i + 1), init_scaling=scale_factor)

        scale_factor *= 0.75

        block4 = downsample_block(block3, 1024, "4_0", init_scaling=scale_factor)
        for i in range(22):
            block4 = residual_block(block4, 1024, "4_{}".format(i + 1), init_scaling=scale_factor)
            scale_factor *= 0.75

        block5 = tf.pad(block4, paddings=tf.constant([[0, 0], [0, 0], [0, 0], [512, 512]]))
        for i in range(3):
            block5 = residual_block(block5, 2048, "5_{}".format(i+1), dilation_rate=2, init_scaling=scale_factor)

        return block5, block2


def xception(image, trainable=True):
    """
    Xception model feature extractor for DeepLabV3+. The last striding is replaced by dilated Convolutions to reduce
    the output stride to 16. The dilation rates are chosen as described in DeepLabV3 with a multi-grid (1, 2, 4) and
    rates (2, 4).

    https://arxiv.org/pdf/1610.02357.pdf

    :param image: input tensor
    :param trainable:  whether the variables of the model should be trainable or fixed
    :return: extracted features.
    """
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
        middle, _ = entry_flow_block(entry2, 728, 3)

        # middle flow
        def middle_flow_block(input, id):
            conv0 = separable_conv_bn(input, filters=728, kernel_size=(3, 1), padding="same", trainable=trainable, name="conv{}_0".format(id))
            conv1 = separable_conv_bn(conv0, filters=728, kernel_size=(1, 3), padding="same", trainable=trainable, name="conv{}_1".format(id))
            conv2 = separable_conv_bn(conv1, filters=728, kernel_size=(3, 1), padding="same", trainable=trainable,  name="conv{}_2".format(id))
            conv2 = separable_conv_bn(conv2, filters=728, kernel_size=(1, 3), padding="same", trainable=trainable,  name="conv{}_3".format(id))

            add = tf.add(input, conv2, name="middle{}_add".format(id))
            return add

        for i in range(16):
            middle = middle_flow_block(middle, i)

        # exit_flow
        conv17_0 = separable_conv_bn(middle, filters=728, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv17_0")
        conv17_1 = separable_conv_bn(conv17_0, filters=1024, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv17_1")
        conv17_2 = separable_conv_bn(conv17_1, filters=1024, kernel_size=(3, 3), padding="same", trainable=trainable, name="conv17_2")
        skip_conv4 = tf.layers.Conv2D(filters=1024, kernel_size=(1, 1), padding="same", trainable=trainable, name="skip_conv4")(middle)
        add4 = tf.add(skip_conv4, conv17_2, name="skip4_add")

        conv17_3 = separable_conv_bn(add4, filters=1536, kernel_size=(3, 3), dilation_rate=2, padding="same", trainable=trainable, name="conv17_3")
        conv17_4 = separable_conv_bn(conv17_3, filters=1536, kernel_size=(3, 3), dilation_rate=2, padding="same", trainable=trainable, name="conv17_4")
        conv17_5 = separable_conv_bn(conv17_4, filters=2048, kernel_size=(3, 3), dilation_rate=2, padding="same", trainable=trainable, name="conv17_5")
        return conv17_5, high_level_features


def resnet101(image, trainable=True):
    """
    ResNet101-like feature extractor for DeeplabV3+. The bottleneck modules have been replaced with Non-bt-1D modules
    similar to ERFNet. Additionally Batch Normalisation Layers have been added after each Convolution. The parameters
    are initialized with a variance scaling initializer. The scale Factor is multiplied with 0.75 throughout the
    Network similar to the findings of this paper:
    https://arxiv.org/pdf/1803.01719.pdf

    ResNet101:
    https://arxiv.org/pdf/1512.03385.pdf

    :param image: input tensor
    :param trainable:  whether the variables of the model should be trainable or fixed
    :return: extracted features.
    """
    with tf.variable_scope("resnet101"):
        def downsample_block(input, output_filter_size, id, init_scaling):
            bt_conv1 = tf.layers.Conv2D(filters=output_filter_size, kernel_size=(1, 1), padding="same", trainable=trainable, strides=(2, 2),
                                        activation="relu", kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                        bias_initializer=tf.initializers.variance_scaling(init_scaling), name="conv{}_1".format(id))(input)
            bt_conv2 = tf.layers.Conv2D(filters=output_filter_size, kernel_size=(3, 3), padding="same", trainable=trainable,
                                        activation="relu", kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                        bias_initializer=tf.initializers.variance_scaling(init_scaling), name="conv{}_2".format(id))(bt_conv1)
            bt_conv3 = tf.layers.Conv2D(filters=output_filter_size, kernel_size=(1, 1), padding="same", trainable=trainable,
                                        kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                        bias_initializer=tf.initializers.variance_scaling(init_scaling), name="conv{}_3".format(id))(bt_conv2)
            skip_conv = tf.layers.Conv2D(filters=output_filter_size, kernel_size=(1, 1), padding="same", trainable=trainable, strides=(2, 2),
                                         kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                         bias_initializer=tf.initializers.variance_scaling(init_scaling), name="skip_conv{}".format(id))(input)
            add = tf.add(skip_conv, bt_conv3, name="add{}".format(id))
            return tf.nn.relu(add, name="block{}_act".format(id))

        def non_bt_1d(input, filters, init_scaling=1.0, dilation_rate=1, name="non_bt_1D"):
            with tf.variable_scope(name):
                conv1 = conv_bn(input, filters=filters, kernel_size=(3, 1), padding="same", trainable=trainable,
                                kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                bias_initializer=tf.initializers.variance_scaling(init_scaling), name="conv1")
                conv2 = conv_bn(conv1, filters=filters, kernel_size=(1, 3), padding="same", trainable=trainable,
                                kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                bias_initializer=tf.initializers.variance_scaling(init_scaling), name="conv2")
                conv3 = conv_bn(conv2, filters=filters, kernel_size=(3, 1), padding="same", dilation_rate=dilation_rate,
                                trainable=trainable, kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                bias_initializer=tf.initializers.variance_scaling(init_scaling), name="conv3")
                conv4 = conv_bn(conv3, filters=filters, kernel_size=(1, 3), padding="same", dilation_rate=dilation_rate,
                                trainable=trainable, kernel_initializer=tf.initializers.variance_scaling(init_scaling),
                                bias_initializer=tf.initializers.variance_scaling(init_scaling), name="conv4")

                result = tf.nn.relu(input + conv4, name="relu")
                return result

        scale_factor = 1.0

        conv1 = conv_bn(image, filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", trainable=trainable,
                        kernel_initializer=tf.initializers.variance_scaling(scale_factor),
                        bias_initializer=tf.initializers.variance_scaling(scale_factor), name="conv1_1")
        conv1 = conv_bn(conv1, filters=64, kernel_size=(3, 3), padding="same", trainable=trainable,
                        kernel_initializer=tf.initializers.variance_scaling(scale_factor),
                        bias_initializer=tf.initializers.variance_scaling(scale_factor), name="conv1_2")
        conv1 = conv_bn(conv1, filters=64, kernel_size=(3, 3), padding="same", trainable=trainable,
                        kernel_initializer=tf.initializers.variance_scaling(scale_factor),
                        bias_initializer=tf.initializers.variance_scaling(scale_factor), name="conv1_3")
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=(2, 2), padding="same", name="pool1")

        scale_factor *= 0.75

        block2_1 = non_bt_1d(pool1, 64, init_scaling=scale_factor, name="block2_1")
        block2_2 = non_bt_1d(block2_1, 64, init_scaling=scale_factor, name="block2_2")
        block2_3 = non_bt_1d(block2_2, 64, init_scaling=scale_factor, name="block2_3")

        scale_factor *= 0.75

        block3_1 = downsample_block(block2_3, 128, "3_1", init_scaling=scale_factor)
        block3_2 = non_bt_1d(block3_1, 128, init_scaling=scale_factor, name="block3_2")
        block3_3 = non_bt_1d(block3_2, 128, init_scaling=scale_factor, name="block3_3")
        block3_4 = non_bt_1d(block3_3, 128, init_scaling=scale_factor, name="block3_4")

        scale_factor *= 0.75

        block4 = downsample_block(block3_4, 256, "4_1", init_scaling=scale_factor)

        for i in range(2, 24):
            block4 = non_bt_1d(block4, 256, init_scaling=scale_factor, name="block4_{}".format(i))
            scale_factor *= 0.75

        block4 = tf.layers.Conv2D(filters=512, kernel_size=(1, 1), activation="relu", padding="same", trainable=trainable,
                                  kernel_initializer=tf.initializers.variance_scaling(scale_factor),
                                  bias_initializer=tf.initializers.variance_scaling(scale_factor), name="filter_up")(block4)

        block5_1 = non_bt_1d(block4, 512, init_scaling=scale_factor, name="block5_1", dilation_rate=2)
        block5_2 = non_bt_1d(block5_1, 512, init_scaling=scale_factor, name="block5_2", dilation_rate=4)
        block5_3 = non_bt_1d(block5_2, 512, init_scaling=scale_factor, name="block5_3", dilation_rate=8)

        return block5_3, block2_3


def deeplab_v3_plus(image, *, n_classes=7, trainable=True):
    """
    DeepLabV3+ Model structure. Either Xception or ResNet101 Model is used as backbone with output stride=16.
    In the last two blocks of Xception the striding is replaced by atrous convolutions with Multi Grid = (1, 2, 4) and
    rates = (2, 4).
    In ResNet the bottleneck modules have been replaced by Non-bt-1D Modules similar to ERFNet.
    Number of parameters in Model (modified ResNet101): 34 107 958
    Number of parameters in Model (Xception): 54 664 300

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
