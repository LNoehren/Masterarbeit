import cv2
import numpy as np
import os

class_counts = [0] * 7
for root, dirs, files in os.walk("/home/lennard/Datasets/vocalfolds-master/annot/"):
    for file in files:
        data = cv2.imread(root + "/" + file, -1)
        for pixel in np.reshape(data, (-1)):
            class_counts[pixel] += 1

print(np.array(class_counts) / 2)