import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.preprocessing import normalize


def get_file_list(root_path):
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_paths.append(root + '/' + file)

    return sorted(file_paths)


def read_image(path):
    img = cv2.imread(path, -1)

    if img is None:
        raise FileNotFoundError("Couldn't open " + path)

    if len(img.shape) == 3:
        img = np.flip(img, 2)

    return img


def get_image_gt(img_path):
    gt_path = img_path.replace("img", "annot")
    gt_path = gt_path.replace("leftImg8bit", "gtFine_labelIds")

    img = read_image(img_path)
    gt = read_image(gt_path)

    return img, gt


def write_image(img, path):
    if len(img.shape) == 3:
        img = np.flip(img, 2)

    cv2.imwrite(path, img)


def write_overlayed_result(net_out, img, path):
    result = transform_net_out(net_out)
    overlayed = 0.5 * result + 0.5 * img
    overlayed = np.reshape(overlayed, (512, 512, 3))
    write_image(overlayed, path)


def transform_net_out(net_out):
    class_mapping = [[105, 105, 105],   # 0: void, gray
                     [255, 0, 0],       # 1: vocal folds, red
                     [0, 0, 255],       # 2: other tissue, blue
                     [0, 255, 0],       # 3: glottal space, green
                     [128, 0, 128],     # 4: pathology, purple
                     [255, 69, 0],      # 5: surgical tool, orange
                     [255, 255, 0]]     # 6: intubation, yellow

    net_out = np.argmax(net_out, axis=-1)
    result = np.take(class_mapping, net_out, axis=0)

    return result


def one_hot_encoding(data, n_classes):
    data_flat = np.array(data).reshape(-1)
    res = np.eye(n_classes)[data_flat]
    res = res.reshape(list(data.shape) + [n_classes])
    return res


def iou(y_true, y_pred):
    tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, 1), tf.equal(y_true, 1)), tf.float32), name="iou_tp")
    fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, 1), tf.equal(y_true, 0)), tf.float32), name="iou_fp")
    fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, 0), tf.equal(y_true, 1)), tf.float32), name="iou_fn")

    div = tp + fp + fn
    return tf.cond(tf.equal(div, 0), lambda: -1.0, lambda: tp / div)


def mean_iou(y_true, y_pred):
    n_classes = y_pred.get_shape().as_list()[-1]
    y_pred = tf.one_hot(tf.argmax(y_pred, -1), n_classes)
    result = 0
    class_count = 0
    class_ious = []

    for i in range(n_classes):
        class_iou = iou(y_true[:, :, :, i], y_pred[:, :, :, i])
        class_count += tf.cond(tf.greater(class_iou, -1), lambda: 1, lambda: 0)
        result += tf.cond(tf.greater(class_iou, -1), lambda: class_iou, lambda: 0.0)
        class_ious.append(class_iou)

    return tf.divide(result, tf.cast(class_count, tf.float32), name="mean_iou"), tf.stack(class_ious)


def compute_mean_class_iou(iou_list):
    result_list = []
    for class_id in range(iou_list.shape[1]):
        class_count = 0
        class_sum = 0
        for value in iou_list[:, class_id]:
            if value > -1:
                class_sum += value
                class_count += 1
        result_list.append(class_sum/class_count)

    return result_list



def weighted_categorical_cross_entropy(y_true, y_pred, use_weights=True):
    """
    categorical cross entropy with class weights computed the same way as in the vocalfolds paper.
    Warning: The weights are only correct for the vocalfolds Dataset!

    :param y_true: not one hot encoded gt
    :param y_pred: output of network
    :return:
    """
    # correct_pred = tf.reduce_max(tf.where(y_true == 1, y_pred, tf.zeros(tf.shape(y_pred))))
    # result = -tf.log(tf.exp(correct_pred) / tf.reduce_sum(tf.exp(y_pred), axis=-1))
    result = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)
    if use_weights:
        # class_occurrences = tf.constant(
        #     [1799515, 33842753, 30357751, 8538275, 336769, 2361923, 6363305], dtype=tf.float32)
        # class_weights = tf.reduce_sum(class_occurrences) / (7 * class_occurences)
        # pre-computed weights for better performance:
        class_weights = tf.constant(
            [6.63673196, 0.35289383, 0.39340525, 1.39874843, 35.46317718, 5.05643017, 1.87683896])
        result *= tf.reduce_max(class_weights * y_true, axis=-1)

    return result


def parametric_relu(x, name):
    alpha = tf.get_variable(name, x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

    return tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)


def write_conf_mat(conf_mat, path):
    classes = ["void", "vocal folds", "other tissue", "glottal space", "pathology", "surgical tool", "intubation"]
    ious = []
    for class_id in range(conf_mat.shape[0]):
        tp = conf_mat[class_id, class_id]
        fp = np.sum(conf_mat[class_id, :]) - tp
        fn = np.sum(conf_mat[:, class_id]) - tp

        ious.append(tp / (tp+fp+fn) if tp+fp+fn > 0 else 0.0)

    with open(path, "w") as writer:
        writer.write("Confusion Matrix:\n")
        writer.write("\t")
        for class_id in range(len(classes)):
            writer.write(classes[class_id] + "\t")
        writer.write("\n")

        for y in range(conf_mat.shape[1]):
            writer.write(classes[y] + "\t")
            for x in range(conf_mat.shape[0]):
                writer.write(str(conf_mat[x, y]) + "\t")
            writer.write("\n")

        writer.write("\nClass IOU's:\n")
        for class_id in range(len(classes)):
            writer.write(classes[class_id] + ": " + str(ious[class_id]) + "\n")
