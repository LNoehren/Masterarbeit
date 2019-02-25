import tensorflow as tf
from utils import mean_iou, weighted_categorical_cross_entropy


class Model:
    """
    Class containing all the tensorflow variables for training, validation, inference. Model structure can be exchanged.
    The loss function for the model is a weighted categorical cross entropy, the optimizer is Adam, the metric is IoU.
    If model structure is a list an ensemble model will be build. All sub-models in an ensemble model are set to
    un-trainable and only the weight tensor combining them will be trained.
    """
    def __init__(self, width, height, n_classes, model_structure, class_weights=None, is_rgb=True):
        """
        initializes the tensorflow variables

        :param width: image width
        :param height: image height
        :param n_classes: number of classes
        :param model_structure: function that returns a prediction given an input image
        :param class_weights: class weights for the loss function
        """
        ensemble = isinstance(model_structure, list)

        self.__name__ = model_structure.__name__ if not ensemble \
            else "-".join([model.__name__ for model in model_structure])
        self.image = tf.placeholder(dtype=tf.float32, shape=(None, width, height, 3), name="image") if is_rgb else \
            tf.placeholder(dtype=tf.float32, shape=(None, width, height, 1), name="image")
        self.y_true = tf.placeholder(dtype=tf.int32, shape=(None, width, height), name="gt")
        y_true_oh = tf.one_hot(self.y_true, n_classes, name="one_hot")

        if ensemble:
            predictions = []
            for model in model_structure:
                predictions.append(model(self.image, n_classes=n_classes, trainable=False))

            predictions = tf.stack(predictions)
            # ensemble networks only work with batch size 1
            shape = tf.TensorShape([len(model_structure), 1, width, height, n_classes])
            ensemble_weights = tf.get_variable("ensemble_weights", shape, dtype=tf.float32, constraint=tf.sigmoid,
                                               initializer=tf.random_uniform_initializer, trainable=True)

            self.y_pred = tf.reduce_sum(ensemble_weights * predictions, axis=0)

        else:
            self.y_pred = model_structure(self.image, n_classes=n_classes)

        self.iou, self.class_ious = mean_iou(y_true_oh, self.y_true, self.y_pred)
        tf.summary.scalar('Mean_IoU', self.iou)
        for i in range(self.class_ious.get_shape().as_list()[0]):
            tf.summary.scalar('Class_{}_IoU'.format(i), self.class_ious[i])

        self.loss = weighted_categorical_cross_entropy(y_true_oh, self.y_pred, class_weights=class_weights)
        tf.summary.scalar('Cross_entropy', tf.reduce_mean(self.loss))

        self.lr = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate")
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, name='optimizer')
        self.train = optimizer.minimize(self.loss, name="train_op")

        self.summary = tf.summary.merge_all()

    def training(self, sess, in_image, gt, learning_rate, do_summary=False):
        """
        performs one train step

        :param sess: tensorflow session
        :param in_image: input image
        :param gt: ground truth
        :param learning_rate: learning rate for the optimizer
        :param do_summary: whether or not to return a summary for tensorboard
        :return: train loss, train IoU, summary if do_summary is true
        """
        if do_summary:
            _, train_loss, train_iou, summary = sess.run(
                (self.train, self.loss, self.iou, self.summary), feed_dict={self.image: in_image,
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
        """
        performs one validation step

        :param sess: tensorflow session
        :param in_image: input image
        :param gt: ground truth
        :param do_summary: whether or not to return a summary for tensorboard
        :return: prediction of the model, validation loss, validation IoU, list of class-IoU's,
                 summary if do_summary is true
        """
        if do_summary:
            prediction, val_loss, val_iou, class_ious, summary = sess.run(
                (self.y_pred, self.loss, self.iou, self.class_ious, self.summary), feed_dict={self.image: in_image,
                                                                                              self.y_true: gt})
        else:
            prediction, val_loss, val_iou, class_ious = sess.run(
                (self.y_pred, self.loss, self.iou, self.class_ious), feed_dict={self.image: in_image,
                                                                                self.y_true: gt})
            summary = None

        return prediction, val_loss, val_iou, class_ious, summary

    def inference(self, sess, in_image):
        """
        performs one inference step

        :param sess: tensorflow session
        :param in_image: input image
        :return: prediction of the model
        """
        prediction = sess.run(self.y_pred, feed_dict={self.image: in_image})

        return prediction
