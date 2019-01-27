from utils import get_file_list, get_image_gt, write_image
from augmentations import flip_h, random_rotation, elastic_deformation, random_crop
from tqdm import tqdm
import numpy as np

data_path = "/home/lennard/Datasets/vocalfolds_augmented_big/img/train"

n = 0

# flip all images
train_paths = get_file_list(data_path)
for img_path in tqdm(train_paths):
    gt_path = img_path.replace("img", "annot")
    image, gt = get_image_gt(img_path)
    image, gt = flip_h(image, gt)

    new_img_path = img_path.replace(".png", "_{num:04d}.png".format(num=n))
    new_gt_path = gt_path.replace(".png", "_{num:04d}.png".format(num=n))

    write_image(image, new_img_path)
    write_image(gt, new_gt_path)
    n += 1

# rotate
train_paths = get_file_list(data_path)
for i in tqdm(range(400, 4000)):
    img_path = train_paths[i % len(train_paths)]
    gt_path = img_path.replace("img", "annot")
    image, gt = get_image_gt(img_path)

    rand0 = np.random.rand()
    if rand0 > 0.3:
        image, gt = random_rotation(image, gt, normal=True)

    rand1 = np.random.rand()
    if rand1 > 0.3:
        image, gt = elastic_deformation(image, gt, 5000, 100)

    rand2 = np.random.rand()
    if rand2 > 0.5:
        rand = np.random.rand()
        crop_size = rand + 1
        image, gt = random_crop(image, gt, int(image.shape[0]/crop_size), int(image.shape[1]/crop_size))

    if rand0 <= 0.3 and rand1 <= 0.3 and rand2 <= 0.5:
        image, gt = elastic_deformation(image, gt, 5000, 80)

    new_img_path = img_path.replace(".png", "_{num:04d}.png".format(num=n))
    new_gt_path = gt_path.replace(".png", "_{num:04d}.png".format(num=n))

    write_image(image, new_img_path)
    write_image(gt, new_gt_path)
    n += 1

# # elastic deformation
# train_paths = get_file_list(data_path)
# for i in tqdm(range(2000, 5000)):
#     img_path = train_paths[i % len(train_paths)]
#     gt_path = img_path.replace("img", "annot")
#     image, gt = get_image_gt(img_path)
#     alpha = 100 if i < len(train_paths) else 200
#     sigma = 10 if i < len(train_paths) else 15
#     image, gt = elastic_deformation(image, gt, alpha, sigma)
#
#     new_img_path = img_path.replace(".png", "_{num:04d}.png".format(num=n))
#     new_gt_path = gt_path.replace(".png", "_{num:04d}.png".format(num=n))
#
#     write_image(image, new_img_path)
#     write_image(gt, new_gt_path)
#     n += 1

# bisher genutzte konfigurationen:
# vocalfolds_augmented = wie im vocalfolds paper (geflipt + rotation +-10, 2000 bilder)
# vocalfolds_augmented_ed = wie vorher + alle rotierten auch elastisch deformiert (alpha=5000, sigma=100), 2000 Bilder
# vocalfolds_augmented_big = alle geflipt, 70% rotiert(normal sigma=10), 70% elastisch deformiert(alpha=5000, sigma=100)
# 50% random crop (crop size zwischen image/1 und image/2), rest elastisch deformiert(alpha=5000, sigma=80), 4000 Bilder
