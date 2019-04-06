import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import csv
import os

"""
This script was used to create matplotlib diagrams of the experiment results.
"""


fig = plt.figure()

# architectures

#"deeplab/resnet101_original/", "c-", deeplab_orig_patch,
# dirpath = "/home/lennard/PycharmProjects/tensorflow_vocalfolds/results/final_experiments/vocalfolds/architekturen/"
# subdirs = ["u_net/", "segnet/", "e_net/", "erfnet/", "deeplab/xception/", "deeplab/resnet_non_bt/"]
# line_colors = ["r-", "g-", "y-", "b-", "k-", "m-"]
# unet_patch = mlines.Line2D([], [], color='red', label='U-Net')
# segnet_patch = mlines.Line2D([], [], color='green', label='SegNet')
# enet_patch = mlines.Line2D([], [], color='yellow', label='E-Net')
# erfnet_patch = mlines.Line2D([], [], color='blue', label='ERFNet')
# deeplab_nb_patch = mlines.Line2D([], [], color='magenta', label='DeepLabV3+ Non-bt-1D')
# deeplab_orig_patch = mlines.Line2D([], [], color='cyan', label='DeepLabV3+ ResNet101')
# deeplab_xc_patch = mlines.Line2D([], [], color='black', label='DeepLabV3+ Xception')
# plt.legend(handles=[unet_patch, segnet_patch, enet_patch, erfnet_patch, deeplab_nb_patch, deeplab_xc_patch])


# pre-training

dirpath = "/home/lennard/PycharmProjects/tensorflow_vocalfolds/results/final_experiments/electron_microscopy/"
subdirs = ["architekturen/erfnet", "architekturen/deeplab/resnet_non_bt", "pre-training/deeplab_cs", "pre-training/deeplab_imagenet", "pre-training/erfnet_cs"]
line_colors = ["b-", "m-", "y-", "g-", "r-"]
erfnet_patch = mlines.Line2D([], [], color='blue', label='ERFNet from scratch')
erfnet_cs_patch = mlines.Line2D([], [], color='red', label='ERFNet Cityscapes')
deeplab_patch = mlines.Line2D([], [], color='magenta', label='DeepLabV3+ Nbt from scratch')
deeplab_cs_patch = mlines.Line2D([], [], color='yellow', label='DeepLabV3+ Nbt Cityscapes')
deeplab_in_patch = mlines.Line2D([], [], color='green', label='DeepLabV3+ Nbt ImageNet')
plt.legend(handles=[erfnet_patch, erfnet_cs_patch, deeplab_patch, deeplab_cs_patch, deeplab_in_patch])


for i in range(len(subdirs)):
    mean_val_iou_list = []
    for root, dirs, files in os.walk(dirpath + subdirs[i]):
        for file in files:
            if "train_log.csv" in file:
                with open(root + "/" + file, newline='') as csvfile:
                    result_reader = csv.DictReader(csvfile)

                    val_iou_list = []
                    for row in result_reader:
                        if "Test" in row["Epoch"]:
                            break

                        val_iou_list.append(float(row["Val_iou"]))
                    mean_val_iou_list.append(val_iou_list)
    if len(mean_val_iou_list) > 0:
        mean_val_iou_list = np.mean(mean_val_iou_list, axis=0)
        plt.plot(range(len(mean_val_iou_list)), mean_val_iou_list, line_colors[i])

ax = fig.gca()
ax.xaxis.grid(True, linestyle="--")
ax.yaxis.grid(True, linestyle="--")
plt.axis([0, 149, 0, 0.85])
plt.ylabel("Mean-IOU")
plt.xlabel("Epochen")
plt.show()
