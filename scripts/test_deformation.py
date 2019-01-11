from utils import get_image_gt, write_image, write_overlaid_result
from augmentations import elastic_deformation


image, gt = get_image_gt("/home/lennard/Datasets/vocalfolds-master/img/train/0001.png")

def_im, def_gt = elastic_deformation(image, gt, 5000, 100)

class_labels = [[[105,105,105], "void"],
                [[255,  0,  0], "vocal folds"],
                [[  0,  0,255], "other tissue"],
                [[  0,255,  0], "glottal space"],
                [[128,  0,128], "pathology"],
                [[255, 69,  0], "surgical tool"],
                [[255,255,  0], "intubation"]]


write_image(def_im, "test.png")
write_overlaid_result(def_gt, def_im, "test_gt.png", class_labels, (512, 512))
write_overlaid_result(gt, image, "orig_gt.png", class_labels, (512, 512))
