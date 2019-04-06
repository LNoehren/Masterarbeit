import cv2
import os

"""
This script was used to resize the Cityscapes dataset, to speed up the training on the dataset
"""

for root, dirs, files in os.walk("/home/lennard/Datasets/Cityscapes/leftImg8bit"):
    for file in files:
        data = cv2.imread(root + "/" + file, -1)
        downsampled = cv2.resize(data, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)

        new_path = "/home/lennard/Datasets/Cityscapes/img/" + root.split("/")[-2] + "/" + file

        if cv2.imwrite(new_path, downsampled):
            print(new_path)
        else:
            print("failed")


for root, dirs, files in os.walk("/home/lennard/Datasets/Cityscapes/gtFine"):
    for file in files:
        if file.endswith("labelIds.png"):
            data = cv2.imread(root + "/" + file, -1)
            downsampled = cv2.resize(data, dsize=(1024, 512), interpolation=cv2.INTER_NEAREST)

            new_path = "/home/lennard/Datasets/Cityscapes/annot/" + root.split("/")[-2] + "/" + file

            if cv2.imwrite(new_path, downsampled):
                print(new_path)
            else:
                print("failed")
