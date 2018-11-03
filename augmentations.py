import cv2
import numpy as np


def flip_h(image):
    result = cv2.flip(image, 1)

    return result


def random_rotation(image):
    degree = np.random.randint(-10, 10)
    rows, cols = image.shape[:2]

    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    result = cv2.warpAffine(image, mat, (cols, rows))

    return result


def perform_augmentations(image, gt_image, augmentations, probs):
    for i in range(len(augmentations)):
        if np.random.rand(1) < probs[i]:
            image = augmentations[i](image)
            gt_image = augmentations[i](gt_image)

    return image, gt_image
