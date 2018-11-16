import cv2
import numpy as np


def flip_h(image, gt):
    result_im = cv2.flip(image, 1)
    result_gt = cv2.flip(gt, 1)

    return result_im, result_gt


def random_rotation(image, gt):
    degree = np.random.randint(-10, 10)
    rows, cols = image.shape[:2]

    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    result_im = cv2.warpAffine(image, mat, (cols, rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    result_gt = cv2.warpAffine(gt, mat, (cols, rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    return result_im, result_gt


def perform_augmentations(image, gt_image, augmentations, probs):
    for i in range(len(augmentations)):
        if np.random.rand(1) < probs[i]:
            image, gt_image = augmentations[i](image, gt_image)

    return image, gt_image
