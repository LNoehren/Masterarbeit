import tensorflow as tf
from utils import get_file_list, write_overlaid_result, compute_mean_class_iou
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import datetime
from models.model import Model
import argparse
from tensorflow.python import debug as tf_debug
from data_generation import DataGenerator
from configuration import Configuration


def main(config):
    # create Tensorflow model
    model = Model(config.image_size[0], config.image_size[1], config.n_classes,
                  config.model_structure, config.class_weights)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.__name__))

    train_paths = get_file_list(config.dataset_path + "img/train")
    val_paths = get_file_list(config.dataset_path + "img/val")
    test_paths = get_file_list(config.dataset_path + "img/test")

    train_steps = int(len(train_paths) / config.batch_sizes["train"])
    val_steps = int(len(val_paths) / config.batch_sizes["validation"])
    test_steps = int(len(test_paths) / config.batch_sizes["test"])

    best_val_iou = -1
    no_improve_count = 0

    time = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    result_dir = "results/{}_{}/".format(model.__name__, time)
    os.makedirs(result_dir + "saved_model")
    log_file = result_dir + "train_log.csv"

    config.save_config(result_dir + "config.yml")

    with tf.Session() as sess:
        if config.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load weights from previous training
        if config.load_path is not None:
            saver.restore(sess, config.load_path)
            print("Model restored.")

        # create log file
        with open(log_file, "w") as log:
            log.write("Epoch,Train_loss,Train_iou,Val_loss,Val_iou")
            for class_info in config.class_labels:
                log.write(",{}".format(class_info[1]))
            log.write("\n")

        # begin training
        for epoch in range(config.epochs):
            shuffle(train_paths)
            train_loss_list = []
            train_iou_list = []
            val_loss_list = []
            val_iou_list = []
            class_iou_list = []

            print("Epoch {}/{}:".format(epoch+1, config.epochs))
            print("Starting training")
            # data generator for training
            train_data_gen = DataGenerator(train_paths, config.batch_sizes["train"], config.n_processes,
                                           config.normalization_params, use_augs=config.use_augs,
                                           class_mapping=config.class_mapping)

            # train loop
            for _ in tqdm(range(train_steps)):
                image_batch, gt_batch = train_data_gen.__next__()

                train_loss, train_iou = model.training(sess, image_batch, gt_batch, config.learning_rate)
                train_loss_list.append(np.mean(train_loss))
                train_iou_list.append(train_iou)

            train_data_gen.stop()

            print("Starting validation")
            # data generator for validation
            val_data_gen = DataGenerator(val_paths, config.batch_sizes["validation"], config.n_processes,
                                         config.normalization_params, class_mapping=config.class_mapping)

            # validation loop
            for _ in tqdm(range(val_steps)):
                image_batch, gt_batch = val_data_gen.__next__()

                _, val_loss, val_iou, class_ious = model.validation(sess, image_batch, gt_batch)
                val_loss_list.append(np.mean(val_loss))
                val_iou_list.append(val_iou)
                class_iou_list.append(class_ious)

            val_data_gen.stop()

            # compute train and val accuracy
            mean_train_loss = np.mean(train_loss_list)
            mean_train_iou = np.mean(train_iou_list)
            mean_val_loss = np.mean(val_loss_list)
            mean_val_iou = np.mean(val_iou_list)
            mean_class_iou = compute_mean_class_iou(np.stack(class_iou_list))
            print("train loss: {} - train iou: {} - val loss: {} - val iou: {}".format(
                mean_train_loss, mean_train_iou, mean_val_loss, mean_val_iou))
            print("class ious: {}".format(mean_class_iou))

            # write to log
            with open(log_file, "a") as log:
                log.write("{},{},{},{},{}".format(epoch+1, mean_train_loss, mean_train_iou,
                                                    mean_val_loss, mean_val_iou))
                for class_iou in mean_class_iou:
                    log.write(",{}".format(class_iou))
                log.write("\n")

            if mean_val_iou > best_val_iou:
                # save model in case of improvement
                save_path = saver.save(sess, result_dir + "saved_model/model.ckpt")
                print("Model saved in {}".format(save_path))
                best_val_iou = mean_val_iou
                no_improve_count = 0
            else:
                # lower learning rate if no improvements in 5 epochs
                no_improve_count += 1
                if no_improve_count > 5:
                    if config.learning_rate > 1e-7:
                        config.learning_rate *= 0.8
                        print("lowered learning rate to {}".format(config.learning_rate))
                    no_improve_count = 0

        # begin tests
        os.makedirs(result_dir + "test_images/")

        test_loss_list = []
        test_iou_list = []
        class_iou_list = []
        print("Starting test")
        # data generator for tests
        test_data_gen = DataGenerator(test_paths, config.batch_sizes["test"], config.n_processes,
                                      config.normalization_params)

        # test loop
        for step in tqdm(range(test_steps)):
            image_batch, gt_batch = test_data_gen.__next__()

            result, test_loss, test_iou, class_ious = model.validation(sess, image_batch, gt_batch)
            test_loss_list.append(np.mean(test_loss))
            test_iou_list.append(test_iou)
            class_iou_list.append(class_ious)

            # write images with overlaid test results
            for b in range(config.batch_sizes["test"]):
                result_path = result_dir + "test_images/" + \
                              test_paths[step * config.batch_sizes["test"] + b].split('/')[-1]
                write_overlaid_result(result[b, :, :, :], image_batch[b, :, :, :], result_path,
                                      config.class_labels, tuple(config.image_size))

        test_data_gen.stop()

        mean_class_iou = compute_mean_class_iou(np.stack(class_iou_list))
        print("test loss: {} - test iou: {}".format(np.mean(test_loss_list), np.mean(test_iou_list)))
        print("class ious: {}".format(mean_class_iou))

        # add test results to log
        with open(log_file, "a") as log:
            log.write("Test_loss,Test_iou")
            for class_info in config.class_labels:
                log.write(",{}".format(class_info[1]))
            log.write("\n{},{}".format(np.mean(test_loss_list), np.mean(test_iou_list)))
            for class_iou in mean_class_iou:
                log.write(",{}".format(class_iou))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a semantic Segmentation Training and Evaluation')
    parser.add_argument("config_path", type=str,
                        help="The path to the config file for the Experiment. Check the Configuration class "
                             "for format information.")
    args = parser.parse_args()
    main(Configuration(args.config_path))
