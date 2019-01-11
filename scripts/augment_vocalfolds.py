from utils import get_file_list, get_image_gt, write_image
from augmentations import flip_h, random_rotation, elastic_deformation
from tqdm import tqdm

data_path = "/home/lennard/Datasets/vocalfolds_augmented_ed/img/train"

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
for i in tqdm(range(400, 2000)):
    img_path = train_paths[i % len(train_paths)]
    gt_path = img_path.replace("img", "annot")
    image, gt = get_image_gt(img_path)
    image, gt = random_rotation(image, gt)
    image, gt = elastic_deformation(image, gt, 5000, 100)

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
