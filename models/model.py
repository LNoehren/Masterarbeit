import tensorflow as tf
from utils import mean_iou, weighted_categorical_cross_entropy


class Model:
    def __init__(self, width, height, n_classes, model_structure, class_weights=None):
        self.__name__ = model_structure.__name__
        self.image = tf.placeholder(dtype=tf.float32, shape=(None, width, height, 3), name="image")
        self.y_true = tf.placeholder(dtype=tf.int32, shape=(None, width, height), name="gt")
        y_true_oh = tf.one_hot(self.y_true, n_classes, name="one_hot")

        self.y_pred = model_structure(self.image, n_classes=n_classes)

        self.iou, self.class_ious = mean_iou(y_true_oh, self.y_pred)
        tf.summary.scalar('Mean_IoU', self.iou)
        for i in range(self.class_ious.get_shape().as_list()[0]):
            tf.summary.scalar('Class_{}_IoU'.format(i), self.class_ious[i])

        self.loss = weighted_categorical_cross_entropy(y_true_oh, self.y_pred, class_weights=class_weights)
        tf.summary.scalar('Cross_entropy', tf.reduce_mean(self.loss))

        self.lr = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='optimizer')
        self.train = optimizer.minimize(self.loss, name="train_op")

        self.merged = tf.summary.merge_all()

    def training(self, sess, in_image, gt, learning_rate, do_summary=False):
        if do_summary:
            _, train_loss, train_iou, summary = sess.run(
                (self.train, self.loss, self.iou, self.merged), feed_dict={self.image: in_image,
                                                                           self.y_true: gt,
                                                                           self.lr: learning_rate})
        else:
            _, train_loss, train_iou = sess.run(
                (self.train, self.loss, self.iou), feed_dict={self.image: in_image,
                                                              self.y_true: gt,
                                                              self.lr: learning_rate})
            summary = None
        return train_loss, train_iou, summary

    def validation(self, sess, in_image, gt, do_summary=False):
        if do_summary:
            prediction, val_loss, val_iou, class_ious, summary = sess.run(
                (self.y_pred, self.loss, self.iou, self.class_ious, self.merged), feed_dict={self.image: in_image,
                                                                                             self.y_true: gt})
        else:
            prediction, val_loss, val_iou, class_ious = sess.run(
                (self.y_pred, self.loss, self.iou, self.class_ious), feed_dict={self.image: in_image,
                                                                                self.y_true: gt})
            summary = None

        return prediction, val_loss, val_iou, class_ious, summary

    def inference(self, sess, in_image):
        prediction = sess.run(self.y_pred, feed_dict={self.image: in_image})

        return prediction
