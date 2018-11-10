from multiprocessing import Pool, Manager
from threading import Thread
from utils import get_image_gt
import numpy as np
import time
from augmentations import flip_h, random_rotation, perform_augmentations


def worker_task(path, queue, use_augs):
    # don't continue if queue is already filled
    while queue.qsize() > 10:
        time.sleep(0.1)

    image, gt = get_image_gt(path)

    if use_augs:
        augs = [flip_h, random_rotation]
        probs = [0.25, 0.25]
        image, gt = perform_augmentations(image, gt, augs, probs)

    queue.put((image, gt))


class DataGenerator:
    def __init__(self, path_list, batch_size, n_processes, use_augs=False):
        self.queue = Manager().Queue()
        self.n_processes = n_processes
        self.batch_size = batch_size
        self.path_list = path_list
        assert self.path_list, "No images found at given Path!"
        self.use_augs = use_augs
        self.master_thread = Thread(target=self.generation_loop, args=[self.queue])
        self.master_thread.start()

    def stop(self):
        self.master_thread.join()

    def __next__(self):
        image_batch = []
        gt_batch = []
        for b in range(self.batch_size):
            image_data, gt_data = self.queue.get()
            image_batch.append(image_data)
            gt_batch.append(gt_data)

        image_batch = np.array(image_batch)
        gt_batch = np.array(gt_batch)

        return image_batch, gt_batch

    def generation_loop(self, queue):
        with Pool(processes=self.n_processes) as pool:
            res = []
            for path in self.path_list:
                res.append(pool.apply_async(worker_task, args=(path, queue, self.use_augs)))

            for r in res:
                r.get()
