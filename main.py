import tensorflow as tf
from utils import get_file_list, write_overlayed_result, compute_mean_class_iou
from models.u_net import u_net
from models.segnet import segnet
from models.enet import e_net
from models.erfnet import erfnet
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import datetime
from models.model import Model
import argparse
from tensorflow.python import debug as tf_debug
from data_generation import DataGenerator


def main(model_structure, dataset_root, number_epochs, batch_sizes, learning_rate, width, height, n_classes, load_path=None, debug=False, use_class_weights=True):
    model = Model(width, height, n_classes, model_structure, use_class_weights)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.__name__))

    train_paths = get_file_list(dataset_root + "train")
    val_paths = get_file_list(dataset_root + "val")
    test_paths = get_file_list(dataset_root + "test")

    train_steps = int(len(train_paths) / batch_sizes[0])
    val_steps = int(len(val_paths) / batch_sizes[1])
    test_steps = int(len(test_paths) / batch_sizes[2])

    best_val_iou = -1
    no_improve_count = 0

    time = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    result_dir = "results/{}_{}/".format(model.__name__, time)
    os.makedirs(result_dir + "saved_model")
    log_file = result_dir + "train_log.csv"

    with open(result_dir + "config", "w") as config_writer:
        config_writer.write("Model: {}\n".format(model.__name__))
        config_writer.write("Epochs: {}\n".format(number_epochs))
        config_writer.write("Dataset: {}\n".format(dataset_root))
        config_writer.write("Batch sizes: {}\n".format(batch_sizes))
        config_writer.write("Starting lr: {}\n".format(learning_rate))
        config_writer.write("Loaded weights: {}".format(load_path))

    with tf.Session() as sess:
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if load_path is not None:
            saver.restore(sess, load_path)
            print("Model restored.")

        with open(log_file, "w") as log:
            log.write("Epoch,Train_loss,Train_iou,Val_loss,Val_iou\n")

        for epoch in range(number_epochs):
            shuffle(train_paths)
            train_loss_list = []
            train_iou_list = []
            val_loss_list = []
            val_iou_list = []
            class_iou_list = []

            print("Epoch {}/{}:".format(epoch+1, number_epochs))
            print("Starting training")
            train_data_gen = DataGenerator(train_paths, batch_sizes[0], 8, use_augs=False)

            for step in tqdm(range(train_steps)):
                image_batch, gt_batch = train_data_gen.__next__()

                train_loss, train_iou = model.training(sess, image_batch, gt_batch, learning_rate)
                train_loss_list.append(np.mean(train_loss))
                train_iou_list.append(train_iou)

            train_data_gen.stop()

            print("Starting validation")
            val_data_gen = DataGenerator(val_paths, batch_sizes[1], 8)

            for step in tqdm(range(val_steps)):
                image_batch, gt_batch = val_data_gen.__next__()

                _, val_loss, val_iou, class_ious = model.validation(sess, image_batch, gt_batch)
                val_loss_list.append(np.mean(val_loss))
                val_iou_list.append(val_iou)
                class_iou_list.append(class_ious)

            val_data_gen.stop()

            mean_train_loss = np.mean(train_loss_list)
            mean_train_iou = np.mean(train_iou_list)
            mean_val_loss = np.mean(val_loss_list)
            mean_val_iou = np.mean(val_iou_list)
            mean_class_iou = compute_mean_class_iou(np.stack(class_iou_list))
            print("train loss: {} - train iou: {} - val loss: {} - val iou: {}".format(
                mean_train_loss, mean_train_iou, mean_val_loss, mean_val_iou))
            print("class ious: {}".format(mean_class_iou))

            with open(log_file, "a") as log:
                log.write("{},{},{},{},{}\n".format(epoch+1, mean_train_loss, mean_train_iou, mean_val_loss, mean_val_iou))

            if mean_val_iou > best_val_iou:
                save_path = saver.save(sess, result_dir + "saved_model/model.ckpt")
                print("Model saved in {}".format(save_path))
                best_val_iou = mean_val_iou
                no_improve_count = 0
            else:
                # lower learning rate if no improvements in 5 epochs
                no_improve_count += 1
                if no_improve_count > 5:
                    if learning_rate > 1e-7:
                        learning_rate *= 0.8
                        print("lowered learning rate to {}".format(learning_rate))
                    no_improve_count = 0

        os.makedirs(result_dir + "test_images/")

        test_loss_list = []
        test_iou_list = []
        class_iou_list = []
        print("Starting test")
        test_data_gen = DataGenerator(test_paths, batch_sizes[2], 8)

        for step in tqdm(range(test_steps)):
            image_batch, gt_batch = test_data_gen.__next__()

            result, test_loss, test_iou, class_ious = model.validation(sess, image_batch, gt_batch)
            test_loss_list.append(np.mean(test_loss))
            test_iou_list.append(test_iou)
            class_iou_list.append(class_ious)

            for b in range(batch_sizes[2]):
                result_path = result_dir + "test_images/" + test_paths[step * batch_sizes[2] + b].split('/')[-1]
                write_overlayed_result(result[b, :, :, :], image_batch[b, :, :, :], result_path)

        test_data_gen.stop()

        mean_class_iou = compute_mean_class_iou(np.stack(class_iou_list))
        print("test loss: {} - test iou: {}".format(np.mean(test_loss_list), np.mean(test_iou_list)))
        print("class ious: {}".format(mean_class_iou))

        with open(log_file, "a") as log:
            log.write(",Test_loss,Test_iou,,\n")
            log.write(",{},{},,\n".format(np.mean(test_loss_list), np.mean(test_iou_list)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a semantic Segmentation Training and Evaluation')
    parser.add_argument("model_structure", type=str,
                        help="The model structure that should be used. "
                             "Supported structures are 'unet', 'segnet', 'enet' and 'erfnet'.")
    parser.add_argument("epochs", type=int,
                        help="The number of training epochs. Chose 0 to only perform tests.")
    parser.add_argument("dataset_path", type=str,
                        help="The path to the dataset image root directory.")
    parser.add_argument("train_batch_size", type=int,
                        help='The train batch size.')
    parser.add_argument("val_batch_size", type=int,
                        help='The validation batch size.')
    parser.add_argument("test_batch_size", type=int,
                        help='The test batch size.')
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, default=1e-3,
                        help='The starting learning rate for the training. Set to 1e-3 by default.')
    parser.add_argument("-l", "--load_path", dest="load_path", type=str, default=None,
                        help="Path to a model checkpoint that should be loaded at the beginning.")
    parser.add_argument("--height", dest="height", type=int, default=512,
                        help="Height of the images of the Dataset.")
    parser.add_argument("--width", dest="width", type=int, default=512,
                        help="Width of the images of the Dataset.")
    parser.add_argument("--n_classes", dest="n_classes", type=int, default=7,
                        help="Number of classes in the Dataset.")
    parser.add_argument("--debug", action="store_true",
                        help="Activate tensorflow debugger.")
    args = parser.parse_args()

    if args.model_structure == "unet":
        model_structure = u_net
    elif args.model_structure == "segnet":
        model_structure = segnet
    elif args.model_structure == "enet":
        model_structure = e_net
    elif args.model_structure == "erfnet":
        model_structure = erfnet
    else:
        raise AttributeError("Unknown model structure: {}".format(args.model_structure))

    bs = (args.train_batch_size, args.val_batch_size, args.test_batch_size)

    assert os.path.isdir(args.dataset_path), "Invalid Dataset path: {}".format(args.dataset_path)

    main(model_structure, args.dataset_path, args.epochs, bs, args.lr, args.height, args.width, args.n_classes, args.load_path, args.debug, use_class_weights=True)
