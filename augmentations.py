import cv2
import numpy as np


def flip_h(image, gt):
    """
    performs a horizontal flip on the image and gt.

    :param image: image data
    :param gt: gt data
    :return: flipped image and gt data
    """
    result_im = cv2.flip(image, 1)
    result_gt = cv2.flip(gt, 1)

    return result_im, result_gt


def random_rotation(image, gt):
    """
    performs a random rotation between -10 and 10 degrees on the image and gt.

    :param image: image data
    :param gt: gt data
    :return: rotated image and gt data
    """
    degree = np.random.randint(-10, 10)
    rows, cols = image.shape[:2]

    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    result_im = cv2.warpAffine(image, mat, (cols, rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    result_gt = cv2.warpAffine(gt, mat, (cols, rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    return result_im, result_gt


def perform_augmentations(image, gt_image, augmentations, probabilities):
    """
    performs the given augmentations with the given probabilities on the image and gt data

    :param image: image data
    :param gt_image: gt data
    :param augmentations: list of augmentation functions. All functions must take the image and gt as arguments
    :param probabilities: list of probabilities for each augmentation. Has to be the same length as the augmentation list
    :return: augmented image and gt data
    """
    for i in range(len(augmentations)):
        if np.random.rand(1) < probabilities[i]:
            image, gt_image = augmentations[i](image, gt_image)

    return image, gt_image
