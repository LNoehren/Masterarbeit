from utils import write_overlaid_result, get_image_gt, class_remapping

#image, gt = get_image_gt("/home/lennard/Datasets/vocalfolds_augmented/img/train/0001_0000_1401.png")
image, gt = get_image_gt("/home/lennard/Datasets/Cityscapes/img/train/aachen_000000_000019_leftImg8bit.png")


class_labels = [[[105,105,105], "void"],
                [[255,  0,  0], "vocal folds"],
                [[  0,  0,255], "other tissue"],
                [[  0,255,  0], "glottal space"],
                [[128,  0,128], "pathology"],
                [[255, 69,  0], "surgical tool"],
                [[255,255,  0], "intubation"]]

class_labels = [[[128, 64,128], "road"],
                [[244, 35,232], "sidewalk"],
                [[ 70, 70, 70], "building"],
                [[102,102,156], "wall"],
                [[190,153,153], "fence"],
                [[153,153,153], "pole"],
                [[250,170, 30], "traffic light"],
                [[220,220,  0], "traffic sign"],
                [[107,142, 35], "vegetation"],
                [[152,251,152], "terrain"],
                [[ 70,130,180], "sky"],
                [[220, 20, 60], "person"],
                [[255,  0,  0], "rider"],
                [[  0,  0,142], "car"],
                [[  0,  0, 70], "truck"],
                [[  0, 60,100], "bus"],
                [[  0, 80,100], "train"],
                [[  0,  0,230], "motorcycle"],
                [[119, 11, 32], "bicycle"],
                [[  0,  0,  0], "void"]]

class_mapping = [-1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, 2, 3, 4, -1, -1, -1, 5, -1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18]

gt = class_remapping(gt, class_mapping)

write_overlaid_result(gt, image, "test.png", class_labels, (512, 1024))
