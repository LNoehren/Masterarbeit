from models.model import Model
import tensorflow as tf
from utils import get_file_list, read_image
from tqdm import tqdm
from models.u_net import u_net
from models.segnet import segnet
from models.e_net import e_net
from models.erfnet import erfnet
from models.deeplab_v3_plus import deeplab_v3_plus

width = 512
height = 512
n_classes = 6
model_structure = u_net
class_weights = [0.35289383, 0.39340525, 1.39874843, 35.46317718, 5.05643017, 1.87683896]
load_path = "/home/lennard/PycharmProjects/tensorflow_vocalfolds/results/final_experiments/vocalfolds/architekturen/u_net/u_net_19-03-10_145824/saved_model/u_net.ckpt"
path_list = get_file_list("/home/lennard/Datasets/vocalfolds-master/img/test")


model = Model(width, height, n_classes, model_structure, class_weights, is_rgb=True)
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.__name__)
saver = tf.train.Saver(variables)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver.restore(sess, load_path)

    for path in tqdm(path_list):
        image = read_image(path).reshape([1, width, height, 3])
        prediction = model.inference(sess, image)
