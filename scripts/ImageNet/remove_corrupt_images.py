import imghdr
import os
from tqdm import tqdm
from random import shuffle
import cv2
from augmentations import resize_image
from utils import read_image, write_image
from multiprocessing import Pool


def read_resize_image(path):
    try:
        image = read_image(path)
    except Exception:
        return False
    image = resize_image(image, 64)
    new_path = "/home/lennard/Datasets/ImageNet/train/resized/" + path.split("/")[-1]
    write_image(image, new_path)
    print("resized image at {}".format(path))
    return True


def remove_corrupt_images(path):
    try:
        image = read_image(path)
    except Exception:
        os.remove(path)
        print("removing image {}".format(path))
        return False
    if image.shape is not (256, 256, 3):
        os.remove(path)
        print("removing image {}".format(path))
        return False
    return True


for root, dirs, files in os.walk("/home/lennard/Datasets/ImageNet/train"):

    filepaths = [root + "/" + file for file in files]
    filepaths = filepaths[:10000]
    pool = Pool(8)
    results = pool.map(read_resize_image, filepaths)

    successful = 0
    error = 0

    for result in results:
        if result:
            successful += 1
        else:
            error += 1

    print("kept: {} removed: {}".format(successful, error))

    pool.close()
    pool.join()
