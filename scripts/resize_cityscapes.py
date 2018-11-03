import cv2
import os


for root, dirs, files in os.walk("/home/lennard/Datasets/Cityscapes/img"):
    for file in files:
        data = cv2.imread(root + "/" + file, -1)
        downsampled = cv2.pyrDown(data, dstsize=(1024, 512))
        if(cv2.imwrite(root + "/" + file, downsampled)):
            print(root + "/" + file)
        else:
            print("failed")


for root, dirs, files in os.walk("/home/lennard/Datasets/Cityscapes/annot"):
    for file in files:
        data = cv2.imread(root + "/" + file, -1)
        downsampled = cv2.pyrDown(data, dstsize=(1024, 512))
        if (cv2.imwrite(root + "/" + file, downsampled)):
            print(root + "/" + file)
        else:
            print("failed")
