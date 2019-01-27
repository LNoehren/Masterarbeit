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


def random_rotation(image, gt, normal=False, random_state=None):
    """
    performs a random rotation between -10 and 10 degrees on the image and gt.

    :param image: image data
    :param gt: gt data
    :param normal: whether to use a normal or a uniform distribution
    :param random_state: numpy random state. If None a new random state will be used
    :return: rotated image and gt data
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    degree = int(random_state.normal(0, 10)) if normal else random_state.randint(-10, 10)
    rows, cols = image.shape[:2]

    mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
    result_im = cv2.warpAffine(image, mat, (cols, rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    result_gt = cv2.warpAffine(gt, mat, (cols, rows), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    return result_im, result_gt


def elastic_deformation(image, gt, alpha, sigma, random_state=None):
    """
    Elastic deformation of images as described in http://cognitivemedium.com/assets/rmnist/Simard.pdf

    :param image: image data
    :param gt: ground truth data
    :param alpha: scale factor for distortions
    :param sigma: variance for distortions
    :param random_state: numpy random state. If None a new random state will be used
    :return: elastic distorted image and gt data
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]
    blur_size = int(sigma*4) | 1
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32),
                          ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32),
                          ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    new_x = (x + dx).astype(np.float32)
    new_y = (y + dy).astype(np.float32)

    distorted_image = cv2.remap(image, new_x, new_y, borderMode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_NEAREST)
    distorted_gt = cv2.remap(gt, new_x, new_y, borderMode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_NEAREST)

    return distorted_image, distorted_gt


def random_crop(image, gt, crop_height, crop_width, random_state=None):
    """
    randomly crops an area of size crop_height x crop_width from the image and ground truth and resizes them back
    to their original size

    :param image: image data
    :param gt: ground truth data
    :param crop_height: height of the cropped area
    :param crop_width: width of the cropped area
    :param random_state: numpy random state. If None a new random state will be used
    :return: cropped and resized image and ground truth
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    height, width = image.shape[:2]

    y = random_state.randint(0, height - crop_height)
    x = random_state.randint(0, width - crop_width)

    cropped_image = image[y:y + crop_height, x:x + crop_width, :]
    cropped_gt = gt[y:y + crop_height, x:x + crop_height]

    cropped_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_NEAREST)
    cropped_gt = cv2.resize(cropped_gt, (width, height), interpolation=cv2.INTER_NEAREST)

    return cropped_image, cropped_gt


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
