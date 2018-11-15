import os
import numpy as np
from utils import read_image


dataset_path = "/home/lennard/Datasets/vocalfolds-master/img"


sum = 0
sumsquare = 0
count = 0
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        data = read_image(root + "/" + file)
        sum += np.sum(data, axis=(0, 1))
        sumsquare += np.sum(data.astype(int)**2, axis=(0, 1))
        count += np.prod(data.shape[:2])

mean = sum / count
std = np.sqrt(sumsquare / count - (mean**2))

print("mean: {} std: {}".format(mean, std))

# vocalfolds: mean: [160.99123017  92.29408995  72.23536678] std: [59.28851215 46.16065498 43.67935978]
# cityscapes: mean: [72.45996064 82.30920661 71.8585933 ] std: [46.99697658 47.7633243  46.96281819]
