import tensorflow as tf
import numpy as np
from layers import upsample_with_indices, bilinear_initializer


input_t = tf.placeholder(tf.float32, [None, 4, 4, 4])
pool, index = tf.nn.max_pool_with_argmax(input_t, (1, 2, 2, 1), (1, 2, 2, 1), padding="SAME")

result = upsample_with_indices(pool, index)
conv = tf.layers.Conv2D(filters=4, kernel_size=(3, 3), padding="same", trainable=False,  kernel_initializer=bilinear_initializer(3, 4), use_bias=False)(result)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

array = np.random.rand(1, 4, 4, 4)
i, p, id, r, c = sess.run((input_t, pool, index, result, conv), feed_dict={input_t: array})

print("input:\n{}".format(i))
print("pool:\n{}".format(p))
print("index:\n{}".format(id))
print("upsampled:\n{}".format(r))
print("conv:\n{}".format(c))
