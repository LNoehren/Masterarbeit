import os
import cv2
import numpy as np


dataset_path = "/home/lennard/Datasets/vocalfolds-master/img"


all_data = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        data = cv2.imread(root + "/" + file, -1)
        all_data.append(np.reshape(data, (-1, 3)))

all_data = np.reshape(np.stack(all_data), (-1, 3))
mean = np.mean(all_data, axis=0)
std = np.std(all_data, axis=0)

print("mean: {} std: {}".format(mean, std))

# vocalfolds: mean: [ 72.23536678  92.29408995 160.99123017] std: [43.67935975 46.16065498 59.28851215]
