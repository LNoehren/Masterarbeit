from multiprocessing import Pool, Manager
from threading import Thread
from utils import read_image, normalize_image
import numpy as np
import time
from augmentations import flip_h, random_rotation, perform_augmentations, resize_image

running = True


"""
This Class is very similar to the normal data_generation class. It was changed to perform data generation for a Training
on the ImageNet dataset. The main differences are that the ground truth for ImageNet is different then for semantic 
segmentation so it has to be handled differently.
"""


def worker_task(path, queue, use_augs, mean, std, class_mapping):
    """
    function that is performed on each data generation worker. Reads images and gt, normalizes the image if mean and
    std is given, performs a class re-mapping if a new class mapping is given performs augmentations if use_augs is
    true and puts the data in a queue. Currently the augmentations that are performed are fixed
    (flip_h, random_rotation each with 0.25 probability). If the given queue has > 10 elements the worker sleeps
    to avoid memory problems.

    :param path: path to the image file
    :param queue: multiprocessing queue in which the results should be stored
    :param use_augs: whether or not augmentations should be performed
    :param mean: mean of the dataset or None if images should not be normalized
    :param std: std of the dataset or None if images should not be normalized
    :param class_mapping: new class mapping for gt or None if it should not be changed
    """
    global running
    if not running:
        return
    # don't continue if queue is already filled
    while queue.qsize() > 300:
        if not running:
            return
        time.sleep(0.1)

    image = read_image(path)
    image = resize_image(image, 64)

    if len(image.shape) < 3:
        return

    if mean and std:
        image = normalize_image(image, mean, std)

    class_id = path.split("/")[-1].split("_")[0]
    gt = class_mapping[class_id]["id"]

    if use_augs:
        augs = [flip_h, random_rotation]
        probs = [0.25, 0.25]
        image, gt = perform_augmentations(image, gt, augs, probs)

    queue.put((image, gt))


class DataGenerator:
    """
    Data Generator that reads data and performs pre-processing of the data in parallel for best performance.
    """
    def __init__(self, path_list, batch_size, n_processes, normalization_params, use_augs=False, class_mapping=None, steps=None):
        """
        initializes varibles for Data Generation and starts the master thread, which supervises the data generation.

        :param path_list: list of paths to the images that should be read
        :param batch_size: batch size which i used by the network
        :param n_processes: number of processes that should be used for data generation
        :param normalization_params: mean and std of the dataset if image normalization should be performed
        :param use_augs: whether or not the data should be augmented
        :param class_mapping: new class mapping for the ground truth if it should be changed
        """
        self.queue = Manager().Queue()
        self.n_processes = n_processes
        self.batch_size = batch_size
        self.path_list = path_list
        assert self.path_list, "No images found at given Path!"
        self.use_augs = use_augs
        self.master_thread = Thread(target=self._generation_loop, args=[self.queue])
        self.master_thread.start()
        self.mean, self.std = normalization_params
        self.class_mapping = class_mapping
        self.steps = steps * self.batch_size if steps else len(path_list)
        global running
        running = True

    def stop(self):
        """
        stops the master thread and by that the data generation
        """
        global running
        running = False
        self.master_thread.join()

    def __next__(self):
        """
        returns the next image batch and ground truth batch form the queue

        :return: image batch and gt batch
        """
        image_batch = []
        gt_batch = []
        for b in range(self.batch_size):
            image_data, gt_data = self.queue.get()
            image_batch.append(image_data)
            gt_batch.append(gt_data)

        image_batch = np.array(image_batch)
        gt_batch = np.array(gt_batch)

        return image_batch, gt_batch

    def _generation_loop(self, queue):
        """
        runs the workers for the data generation. This is the function that is run by the master thread.

        :param queue: multiprocessing queue in which the data should be stored
        """
        with Pool(processes=self.n_processes) as pool:
            res = []
            for i in range(self.steps):
                path = self.path_list[i % len(self.path_list)]
                res.append(pool.apply_async(worker_task,
                                            args=(path, queue, self.use_augs, self.mean, self.std, self.class_mapping)))
            for r in res:
                if not running:
                    return
                r.get()

        if running:
            self._generation_loop(queue)
