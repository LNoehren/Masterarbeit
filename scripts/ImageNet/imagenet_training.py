import tensorflow as tf
from models.deeplab_v3_plus import resnet101
from utils import weighted_categorical_cross_entropy, get_file_list
from data_generation_ImageNet import DataGenerator
import numpy as np
import datetime
import os
import pickle
from tqdm import tqdm
from random import shuffle


# script parameters
dataset_path = "/home/lennard/Datasets/ImageNet/"
class_dict_path = dataset_path + "ImageNet_classes.pkl"
image_width = 64
image_height = 64
n_classes = 1000
model_structure = resnet101
epochs = 100
batch_size = 16
load_path = "results/resnet101_19-02-14_235152/saved_model/resnet101.ckpt"

train_paths = get_file_list(dataset_path + "train/")
val_paths = get_file_list(dataset_path + "val/")
test_paths = get_file_list(dataset_path + "test/")
class_dict = pickle.load(open(class_dict_path, "rb"))

time = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
result_dir = "results/{}_{}/".format(model_structure.__name__, time)
os.makedirs(result_dir + "saved_model")
log_file = result_dir + "train_log.csv"

train_steps = 10000 #len(train_paths) // batch_size
val_steps = 1000 #len(val_paths) // batch_size


# initializing tf tensors
input_image = tf.placeholder(dtype=tf.float32, shape=(None, image_width, image_height, 3), name="image")
y_true = tf.placeholder(dtype=tf.int32, shape=(None,), name="gt")
y_true_oh = tf.one_hot(y_true, n_classes)

model_out, _ = model_structure(input_image)
model_out = tf.reduce_mean(model_out, axis=[1, 2], keepdims=False, name="avg_pool")
y_pred = tf.layers.Dense(n_classes, activation="softmax", name="classes")(model_out)

loss = weighted_categorical_cross_entropy(y_true_oh, y_pred)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, name="optimizer")
train = optimizer.minimize(loss, name="train_op")

# saver
variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model_structure.__name__)
saver = tf.train.Saver(variables)

with open(log_file, "w") as log:
    log.write("Epoch,Train_loss,Train_iou,Val_loss,Val_iou, Val_acc")
    log.write("\n")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if load_path:
        saver.restore(sess, load_path)

    best_val_loss = np.inf
    for epoch in range(epochs):
        shuffle(train_paths)
        shuffle(val_paths)

        # training
        print("Epoch {}/{}:".format(epoch + 1, epochs))
        print("Starting training")

        train_loss_list = []
        train_data_gen = DataGenerator(train_paths, batch_size, 8, [None, None], class_mapping=class_dict, steps=train_steps)
        for step in tqdm(range(train_steps)):

            image_batch, gt_batch = train_data_gen.__next__()
            train_loss, _ = sess.run((loss, train), feed_dict={input_image: image_batch, y_true: gt_batch})
            train_loss_list.append(train_loss)

        train_data_gen.stop()

        # validation
        print("Starting validation")

        val_loss_list = []
        accuracy_list = []
        val_data_gen = DataGenerator(train_paths, batch_size, 8, [None, None], class_mapping=class_dict, steps=val_steps)
        for step in tqdm(range(val_steps)):
            image_batch, gt_batch = val_data_gen.__next__()
            val_loss, pred = sess.run((loss, y_pred), feed_dict={input_image: image_batch, y_true: gt_batch})
            val_loss_list.append(val_loss)
            accuracy_list.append(np.argmax(pred, axis=-1) == gt_batch)

        val_data_gen.stop()

        mean_train_loss = np.mean(train_loss_list)
        mean_val_loss = np.mean(val_loss_list)
        mean_accuracy = np.sum(accuracy_list) / (val_steps * batch_size)
        print("train loss: {} - val loss {} - val accuracy {}".format(mean_train_loss, mean_val_loss, mean_accuracy))

        with open(log_file, "a") as log:
            log.write("{},{},{}, {}".format(epoch + 1, mean_train_loss, mean_val_loss, mean_accuracy))
            log.write("\n")

        if mean_val_loss < best_val_loss:
            save_path = saver.save(sess, result_dir + "saved_model/{}.ckpt".format(model_structure.__name__))
            print("Model saved in {}".format(save_path))
            best_val_loss = mean_val_loss
