import tensorflow as tf
from utils import mean_iou, weighted_categorical_cross_entropy
from sklearn.metrics import confusion_matrix
import numpy as np


class Model:
    def __init__(self, width, height, n_classes, model_structure, use_class_weights=True):
        self.__name__ = model_structure.__name__
        self.image = tf.placeholder(dtype=tf.float32, shape=(None, width, height, 3), name="image")
        self.y_true = tf.placeholder(dtype=tf.int32, shape=(None, width, height), name="gt")
        y_true_oh = tf.one_hot(self.y_true, n_classes, name="one_hot")

        self.y_pred = model_structure(self.image, n_classes=n_classes)

        self.iou, self.class_ious = mean_iou(y_true_oh, self.y_pred)
        self.loss = weighted_categorical_cross_entropy(y_true_oh, self.y_pred, use_weights=use_class_weights)

        self.lr = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='optimizer')
        self.train = optimizer.minimize(self.loss, name="train_op")

    def training(self, sess, in_image, gt, learning_rate):
        _, train_loss, train_iou = sess.run(
            (self.train, self.loss, self.iou), feed_dict={self.image: in_image,
                                                          self.y_true: gt,
                                                          self.lr: learning_rate})

        return train_loss, train_iou

    def validation(self, sess, in_image, gt):
        prediction, val_loss, val_iou, class_ious = sess.run(
            (self.y_pred, self.loss, self.iou, self.class_ious), feed_dict={self.image: in_image,
                                                                            self.y_true: gt})

        #conf_mat = confusion_matrix(np.reshape(gt, (-1)), np.reshape(np.argmax(prediction, -1), (-1)), labels=np.arange(0, prediction.shape[-1]))

        return prediction, val_loss, val_iou, class_ious

    def inference(self, sess, in_image):
        prediction = sess.run(self.y_pred, feed_dict={self.image: in_image})

        return prediction
