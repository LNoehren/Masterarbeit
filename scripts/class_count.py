import cv2
import numpy as np
import os
from utils import class_remapping

"""
this script was used to count the class occurences in datasets. This was necessary to use the weighted categorical 
cross entropy
"""


class_counts = [0] * 7
for root, dirs, files in os.walk("/home/lennard/Datasets/electron_microscopy/annot/"):
    for file in files:
        data = cv2.imread(root + "/" + file, -1)

        #data = class_remapping(data, [0, 0, 0, 0, 0, 0, 1])
        for pixel in np.reshape(data, (-1)):

            class_counts[pixel] += 1

print(np.array(class_counts))

# vocalfolds
# all classes:      [  1.799.515   33.842.753  30.357.751  8.538.275  336.769  23.619.232  6.363.305]
# bg, void:         [103058085    1799515]
# bg, vocalfolds:   [ 71014847   33842753]
# bg, other tissue: [ 74499849   30357751]
# bg, glottal space:[ 96319325    8538275]
# bg, pathology:    [104520831     336769]
# bg, surgical tool:[ 81238368   23619232]
# bg, intubation:   [ 98494295    6363305]

# electron_microscopy
# [368200930  21082910]
