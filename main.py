import tensorflow as tf
from utils import get_file_list, write_overlaid_result, compute_mean_class_iou, de_normalize_image, save_histogram
from models.model import Model
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import datetime
import argparse
from tensorflow.python import debug as tf_debug
from data_generation import DataGenerator
from configuration import Configuration


def main(config):
    """
    Main function. Initializes the tensorflow model, performs training, validation and tests.

    :param config: Configuration Object containing all parameters.
    """
    # create Tensorflow model
    model = Model(config.image_size[0], config.image_size[1], config.n_classes,
                  config.model_structure, config.class_weights)

    # number of trainable parameters
    number_of_params = np.sum([np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()])
    print("Number of parameters in Model: {}".format(number_of_params))

    ensemble = isinstance(config.model_structure, list)

    if ensemble:
        if config.pre_training:
            raise NotImplementedError("Pre-Training with different Datasets is not supported for Ensemble Models")

        # load all sub-models for ensemble models
        saver = [tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=sub_model.__name__))
                 for sub_model in config.model_structure]
    else:
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.__name__)
        if config.pre_training == "cityscapes":
            # remove softmax layer from variable set
            variables = [var for var in variables if model.__name__ + "/classes/" not in var.name]
        elif config.pre_training == "imagenet":
            # only load encoder variables
            if model.__name__ is not "deeplab_v3_plus":
                raise NotImplementedError("ImageNet Pre-Training is currently only supported for DeepLabV3+")
            variables = {var.name.split("deeplab_v3_plus/")[-1].split(":")[0]: var for var in
                         tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deeplab_v3_plus/resnet101")}

        saver = tf.train.Saver(variables)

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
    os.makedirs(result_dir + "diagrams")
    log_file = result_dir + "train_log.csv"

    config.save_config(result_dir + "config.yml")

    with tf.Session() as sess:
        if config.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        # tensorboard
        train_writer = tf.summary.FileWriter(result_dir + '/tensorboard/train', sess.graph)
        val_writer = tf.summary.FileWriter(result_dir + '/tensorboard/val')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # load weights from previous training
        if config.load_path is not None:
            if ensemble:
                for i in range(len(saver)):
                    saver[i].restore(sess, config.load_path[i])
                    print("Model '{}' restored.".format(config.model_structure[i]))
            else:
                saver.restore(sess, config.load_path)

                if config.pre_training:
                    # after the pre-trained model is restored all variables can be added to the saver
                    print("Model '{}' restored after Pre-Training with {}.".format(model.__name__, config.pre_training))
                    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=model.__name__))
                else:
                    print("Model '{}' restored.".format(model.__name__))

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
            for step in tqdm(range(train_steps)):
                image_batch, gt_batch = train_data_gen.__next__()

                do_summary = step % 100
                train_loss, train_iou, summary = model.training(sess, image_batch, gt_batch,
                                                                config.learning_rate, do_summary)
                train_loss_list.append(np.mean(train_loss))
                train_iou_list.append(train_iou)

                if do_summary:
                    train_writer.add_summary(summary, epoch * train_steps + step)

            train_data_gen.stop()

            print("Starting validation")
            # data generator for validation
            val_data_gen = DataGenerator(val_paths, config.batch_sizes["validation"], config.n_processes,
                                         config.normalization_params, class_mapping=config.class_mapping)

            # validation loop
            for step in tqdm(range(val_steps)):
                image_batch, gt_batch = val_data_gen.__next__()

                do_summary = step == val_steps-1
                _, val_loss, val_iou, class_ious, summary = model.validation(sess, image_batch, gt_batch, do_summary)
                val_loss_list.append(np.mean(val_loss))
                val_iou_list.append(val_iou)
                class_iou_list.append(class_ious)

                if do_summary:
                    val_writer.add_summary(summary, epoch)

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

            save_histogram(train_iou_list, "Training IoU", "Number of Samples",
                           result_dir + "diagrams/train_histogram.png",
                           "Mean-IoU={}".format(np.mean(train_iou_list).round(decimals=2)))
            save_histogram(val_iou_list, "Validation IoU", "Number of Samples",
                           result_dir + "diagrams/val_histogram.png",
                           "Mean-IoU={}".format(np.mean(val_iou_list).round(decimals=2)))

            # write to log
            with open(log_file, "a") as log:
                log.write("{},{},{},{},{}".format(epoch+1, mean_train_loss, mean_train_iou,
                                                  mean_val_loss, mean_val_iou))
                for class_iou in mean_class_iou:
                    log.write(",{}".format(class_iou))
                log.write("\n")

            if mean_val_iou > best_val_iou:
                # save model in case of improvement only for non ensemble models
                if not ensemble:
                    save_path = saver.save(sess, result_dir + "saved_model/{}.ckpt".format(model.__name__))
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
                                      config.normalization_params, class_mapping=config.class_mapping)

        # test loop
        for step in tqdm(range(test_steps)):
            image_batch, gt_batch = test_data_gen.__next__()

            result, test_loss, test_iou, class_ious, _ = model.validation(sess, image_batch, gt_batch)
            test_loss_list.append(np.mean(test_loss))
            test_iou_list.append(test_iou)
            class_iou_list.append(class_ious)

            # write images with overlaid test results
            for b in range(config.batch_sizes["test"]):
                result_path = result_dir + "test_images/" + \
                              test_paths[step * config.batch_sizes["test"] + b].split('/')[-1]
                img = de_normalize_image(image_batch[b, :, :, :],
                                         config.normalization_params[0], config.normalization_params[1])
                write_overlaid_result(result[b, :, :, :], gt_batch[b, :, :], img, result_path,
                                      config.class_labels, tuple(config.image_size))

        test_data_gen.stop()

        save_histogram(test_iou_list, "Test IoU", "Number of Samples", result_dir + "diagrams/test_histogram.png",
                       "Mean-IoU={}".format(np.mean(test_iou_list).round(decimals=2)))

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

    tf.reset_default_graph()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a semantic Segmentation Training and Evaluation')
    parser.add_argument("config_path", type=str, nargs='+',
                        help="The path to the config file for the Experiment. Check the Configuration class "
                             "for format information.")
    args = parser.parse_args()
    if isinstance(args.config_path, list):
        for configuration in args.config_path:
            main(Configuration(configuration))
    else:
        main(Configuration(args.config_path))
