import tensorflow as tf
from layers import bottleneck, e_net_initializer_block


def e_net(image, *, n_classes=7):
    with tf.variable_scope("e_net"):

        init_block = e_net_initializer_block(image, filters=16)

        # encoder
        bottleneck1_0, index1 = bottleneck(init_block, filters=64, dropout_rate=0.01, downsampling=True, name="bottleneck1_0")
        bottleneck1_1 = bottleneck(bottleneck1_0, filters=64, dropout_rate=0.01, name="bottleneck1_1")
        bottleneck1_2 = bottleneck(bottleneck1_1, filters=64, dropout_rate=0.01, name="bottleneck1_2")
        bottleneck1_3 = bottleneck(bottleneck1_2, filters=64, dropout_rate=0.01, name="bottleneck1_3")
        bottleneck1_4 = bottleneck(bottleneck1_3, filters=64, dropout_rate=0.01, name="bottleneck1_4")

        bottleneck2_0, index2 = bottleneck(bottleneck1_4, filters=128, dropout_rate=0.1, downsampling=True, name="bottleneck2_0")
        bottleneck2_1 = bottleneck(bottleneck2_0, filters=128, dropout_rate=0.1, name="bottleneck2_1")
        bottleneck2_2 = bottleneck(bottleneck2_1, filters=128, dropout_rate=0.1, dilation=2, name="bottleneck2_2")
        bottleneck2_3 = bottleneck(bottleneck2_2, filters=128, dropout_rate=0.1, asymmetric=True, name="bottleneck2_3")
        bottleneck2_4 = bottleneck(bottleneck2_3, filters=128, dropout_rate=0.1, dilation=4, name="bottleneck2_4")
        bottleneck2_5 = bottleneck(bottleneck2_4, filters=128, dropout_rate=0.1, name="bottleneck2_5")
        bottleneck2_6 = bottleneck(bottleneck2_5, filters=128, dropout_rate=0.1, dilation=8, name="bottleneck2_6")
        bottleneck2_7 = bottleneck(bottleneck2_6, filters=128, dropout_rate=0.1, asymmetric=True, name="bottleneck2_7")
        bottleneck2_8 = bottleneck(bottleneck2_7, filters=128, dropout_rate=0.1, dilation=16, name="bottleneck2_8")

        bottleneck3_1 = bottleneck(bottleneck2_8, filters=128, dropout_rate=0.1, name="bottleneck3_1")
        bottleneck3_2 = bottleneck(bottleneck3_1, filters=128, dropout_rate=0.1, dilation=2, name="bottleneck3_2")
        bottleneck3_3 = bottleneck(bottleneck3_2, filters=128, dropout_rate=0.1, asymmetric=True, name="bottleneck3_3")
        bottleneck3_4 = bottleneck(bottleneck3_3, filters=128, dropout_rate=0.1, name="bottleneck3_4")
        bottleneck3_5 = bottleneck(bottleneck3_4, filters=128, dropout_rate=0.1, dilation=4, name="bottleneck3_5")
        bottleneck3_6 = bottleneck(bottleneck3_5, filters=128, dropout_rate=0.1, dilation=8, name="bottleneck3_6")
        bottleneck3_7 = bottleneck(bottleneck3_6, filters=128, dropout_rate=0.1, asymmetric=True, name="bottleneck3_7")
        bottleneck3_8 = bottleneck(bottleneck3_7, filters=128, dropout_rate=0.1, dilation=16, name="bottleneck3_8")

        #decoder
        bottleneck4_0 = bottleneck(bottleneck3_8, filters=64, dropout_rate=0.1, upsampling_indices=index2, name="bottleneck4_0")
        bottleneck4_1 = bottleneck(bottleneck4_0, filters=64, dropout_rate=0.1, name="bottleneck4_1")
        bottleneck4_2 = bottleneck(bottleneck4_1, filters=64, dropout_rate=0.1, name="bottleneck4_2")

        bottleneck5_0 = bottleneck(bottleneck4_2, filters=16, dropout_rate=0.1, upsampling_indices=index1, name="bottleneck5_0")
        bottleneck5_1 = bottleneck(bottleneck5_0, filters=16, dropout_rate=0.1, name="bottleneck5_1")

    classes = tf.layers.Conv2DTranspose(filters=n_classes, kernel_size=(1, 1), strides=(2, 2), padding="same", activation="softmax", name="classes")(bottleneck5_1)

    return classes
