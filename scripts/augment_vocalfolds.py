from utils import get_file_list, get_image_gt, write_image
from augmentations import flip_h, random_rotation

n = 0
data_path = "/home/lennard/Datasets/vocalfolds_augmented/img/train"

train_paths = get_file_list(data_path)
for img_path in train_paths:
    gt_path = img_path.replace("img", "annot")
    image, gt = get_image_gt(img_path)
    image = flip_h(image)
    gt = flip_h(gt)
    write_image(image, img_path + "_{num:04d}".format(num=n))
    write_image(gt, gt_path + "_{num:04d}".format(num=n))
    n += 1

train_paths = get_file_list(data_path)
for i in range(400, 2000):
    img_path = train_paths[i % len(train_paths)]
    gt_path = img_path.replace("img", "annot")
    image, gt = get_image_gt(img_path)
    image = random_rotation(image)
    gt = random_rotation(gt)
    write_image(image, img_path + "_{num:04d}".format(num=n))
    write_image(gt, gt_path + "_{num:04d}".format(num=n))
    n += 1
