from utils import get_file_list, get_image_gt, write_image
from augmentations import flip_h, random_rotation

n = 0
data_path = "/home/lennard/Datasets/vocalfolds_augmented/img/train"

train_paths = get_file_list(data_path)
for img_path in train_paths:
    gt_path = img_path.replace("img", "annot")
    image, gt = get_image_gt(img_path)
    image, gt = flip_h(image, gt)

    new_img_path = img_path.replace(".png", "_{num:04d}.png".format(num=n))
    new_gt_path = gt_path.replace(".png", "_{num:04d}.png".format(num=n))

    write_image(image, new_img_path)
    write_image(gt, new_gt_path)
    n += 1

train_paths = get_file_list(data_path)
for i in range(400, 2000):
    img_path = train_paths[i % len(train_paths)]
    gt_path = img_path.replace("img", "annot")
    image, gt = get_image_gt(img_path)
    image, gt = random_rotation(image, gt)

    new_img_path = img_path.replace(".png", "_{num:04d}.png".format(num=n))
    new_gt_path = gt_path.replace(".png", "_{num:04d}.png".format(num=n))

    write_image(image, new_img_path)
    write_image(gt, new_gt_path)
    n += 1
