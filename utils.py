import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt


def get_file_list(root_path):
    """
    returns a sorted list of all files in the given directory and subdirectories.

    :param root_path: path to the root directory
    :return: list of files
    """
    file_paths = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            file_paths.append(root + '/' + file)

    return sorted(file_paths)


def read_image(path, transform_rgb=False):
    """
    reads the image with openCV and flips color channels to be in RGB order.

    :param path: path to the image file
    :param transform_rgb: whether or not a gray scale image should be transformed to RGB after reading
    :return: ndarray of image
    :raises FileNotFoundError if the file couldn't be read
    """
    img = cv2.imread(path, -1)

    if img is None:
        raise FileNotFoundError("Couldn't open " + path)

    if len(img.shape) == 3:
        img = np.flip(img, 2)
    else:
        if transform_rgb:
            img = np.stack([img, img, img], axis=2)
    return img


def get_image_gt(img_path):
    """
    reads an image and the corresponding ground truth. The ground truth should have the same name as the image,
    except that 'img' will be replaced with 'annot' and 'leftImg8bit' will be replaced with
    'gtFine_labelIds'(cityscapes). The image will be transformed to RGB if it is gray scale.

    :param img_path: path to the image file
    :return: ndarrays of the image and ground truth
    """
    gt_path = img_path.replace("img", "annot")
    gt_path = gt_path.replace("leftImg8bit", "gtFine_labelIds")

    img = read_image(img_path, transform_rgb=True)
    gt = read_image(gt_path)

    return img, gt


def write_image(img, path):
    """
    writes an image using openCV.

    :param img: image data
    :param path: write path
    """
    if len(img.shape) == 3:
        img = np.flip(img, 2)

    cv2.imwrite(path, img)


def write_overlaid_result(net_out, gt, img, path, class_labels, image_size):
    """
    converts the net out to a colored representation and writes the image overlaid with the net out. Wrong classified
    points will be colored grey.

    :param net_out: the output of the Network
    :param gt: the correct ground truth for the image. If this is not None every wrong classified pixel will be gray
    :param img: the input image corresponding to the net out
    :param path: write path
    :param class_labels: class labels containing the colors for every class
    :param image_size: the width and height of the image
    """
    class_mapping = [class_info[0] for class_info in class_labels]
    net_out = np.argmax(net_out, axis=-1)

    if gt is not None:
        class_mapping.append([100, 100, 100])
        net_out = np.where(np.logical_or(net_out == gt, gt < 0), net_out, np.full(net_out.shape, -1))
    rgb_net_out = np.take(class_mapping, net_out, axis=0)

    overlaid = 0.5 * rgb_net_out + 0.5 * img
    overlaid = np.reshape(overlaid, image_size + (3,))
    write_image(overlaid, path)


def one_hot_encoding(data, n_classes):
    """
    performs a one-hot-encoding with numpy (on CPU)

    :param data: input data in non-one-hot format
    :param n_classes: number of classes
    :return: data in one-hot format
    """
    data_flat = np.array(data).reshape(-1)
    res = np.eye(n_classes)[data_flat]
    res = res.reshape(list(data.shape) + [n_classes])
    return res


def iou(y_true, y_pred):
    """
    computes class IoU with tensorflow functions (on GPU)

    :param y_true: ground truth data of the class (One-Hot)
    :param y_pred: prediction of the data
    :return: Class IoU for the given data, -1 if the class doesn't appear in y_true and y_pred
    """
    with tf.variable_scope('class_Iou'):
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, 1), tf.equal(y_true, 1)), tf.float32), name="iou_tp")
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, 1), tf.equal(y_true, 0)), tf.float32), name="iou_fp")
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, 0), tf.equal(y_true, 1)), tf.float32), name="iou_fn")

        div = tp + fp + fn
        return tf.cond(tf.equal(div, 0), lambda: -1.0, lambda: tp / div)


def mean_iou(y_true_oh, y_true, y_pred):
    """
    computes mean IoU and class IoU's for all classes.

    :param y_true_oh: ground truth data in one-hot format
    :param y_true: ground truth data in not-one-hot format. Used to compute mask for ignore class
    :param y_pred: prediction of the network
    :return: mean-IoU of all classes and a list of all class-IoU's
    """
    with tf.variable_scope('mean_IoU'):
        n_classes = y_pred.get_shape().as_list()[-1]
        y_pred = tf.one_hot(tf.argmax(y_pred, -1), n_classes)
        result = 0
        class_count = 0
        class_iou_list = []

        ignore_class_mask = tf.not_equal(y_true, -1, name="ignore_class_mask")

        for i in range(n_classes):
            class_iou = iou(tf.boolean_mask(y_true_oh[:, :, :, i], ignore_class_mask),
                            tf.boolean_mask(y_pred[:, :, :, i], ignore_class_mask))
            class_count += tf.cond(tf.greater(class_iou, -1), lambda: 1, lambda: 0)
            result += tf.cond(tf.greater(class_iou, -1), lambda: class_iou, lambda: 0.0)
            class_iou_list.append(class_iou)

        class_iou_list = tf.stack(class_iou_list)
        return tf.divide(result, tf.cast(class_count, tf.float32), name="mean_iou"), class_iou_list


def compute_mean_class_iou(iou_list):
    """
    computes the mean class IoU for each class. Ignores entries with -1.

    :param iou_list: list of class IoU-lists.
    :return: list of mean-class-IoU's. One entry for each class
    """
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


def weighted_categorical_cross_entropy(y_true, y_pred, class_weights=None):
    """
    categorical cross entropy with possibility to specify weights for each class.

    :param y_true: not one hot encoded gt
    :param y_pred: output of network
    :param class_weights: Class weights to put more focus on some classes.
    :return:
    """
    with tf.variable_scope('weighted_categorical_cross_entropy'):
        result = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)
        if class_weights:
            result *= tf.reduce_max(class_weights * y_true, axis=-1)

        return result


def normalize_image(image, mean, std):
    """
    normalizes a image given the mean and std of the dataset

    :param image: input image
    :param mean: mean of the dataset
    :param std: std of the dataset
    :return: normalized image
    """
    return (image - mean) / std


def de_normalize_image(image, mean, std):
    """
    inverse function to normalize_image

    :param image: normalized image
    :param mean: mean of the dataset
    :param std: std of the dataset
    :return: not normalized image
    """
    if mean and std:
        return image * std + mean
    else:
        return image


def class_remapping(gt, class_mapping):
    """
    maps classes in ground truth data to other classes

    :param gt: ground truth data
    :param class_mapping: new mapping for each class. This should be a list with the same number of elements as
                          classes in the original ground truth
    :return: ground truth with classes mapped according to the class mapping
    """
    new_gt = np.take(class_mapping, gt)
    return new_gt


def save_histogram(data_array, x_label, y_label, filename, text=None):
    """
    creates a histogram and saves it in a file.

    :param data_array: histogram data
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param filename: filename for the histogram file
    :param text: if not None, text that will be added in the top left corner of the histogram
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.hist(data_array)

    if text:
        ax.text(0.3, 0.92, text, horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.grid(True)
    plt.savefig(filename)

    plt.close()
