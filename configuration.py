import yaml
from models.u_net import u_net
from models.segnet import segnet
from models.e_net import e_net
from models.erfnet import erfnet
import os
import warnings


class Configuration:
    """
    This Class loads configuration parameters for the experiments from yaml files. The following fields are supported:
        - model_structure: default: undefined
            This needs to be specified to select the model that should be used. Currently supported models are
            u_net, segnet, e_net and erfnet
        - dataset_path: default: None
            The path to the root directory of the dataset. The directory structure of the dataset has to be like this:
                img/
                    train/
                        train-data
                    val/
                        validation-data
                    test/
                        test-data
                annot/
                    train/
                        train-gt
                    val/
                        validation-gt
                    test/
                        test-gt
        - epochs: default: 0
            The number of training epochs. If 0 only the Tests are performed.
        - learning_rate: default: 1e-4
            The starting learning rate of the optimizer. This is reduced during the training if the validation
            mean IoU stagnates.
        - batch_sizes: default: {train: 1, validation: 1, test: 1}
            The train, validation and test batch sizes. Needs to be a dictionary with the same keys as in default.
        - image_size: default: [512, 512]
            The image sizes of the Dataset. Needs to be a list or tuple. First element is height, second is width,
        - n_classes: default: 7
            The number of classes in the Dataset.
        - load_path: default: None
            Path to a trained model that should be loaded. If it is None no model will be loaded.
        - use_augs: default: False
            Whether or not to use augmentations during training
        - class_weights: default: True
            Whether to use class weights in the loss function. Only available for the vocalfolds Dataset.
        - n_processes: default: 8
            The number of processes that should be used for Pre-Processing (reading images, augmentations)
        - debug: default: False
            Activates the Tensorflow debugger.
        - normalization_params: default [None, None]
            Mean and std of the dataset for image normalization. If this is None no normalization will be performed.
        - class_labels: default None
            List of classes in the dataset. Each element should be a list where the first element is the color
            that should be used for the class and the second element is the name of the class.
        - class_mapping: default None
            List that is used to map one class to another. For example cityscapes has in total 34 classes, but usually
            only 19 are used for training, so the mapping should be a list with 34 elements that map each class to one
            of the 19 main classes or -1 to ignore it.

    Check the configs directory for example config files for the vocalfolds and cityscapes datasets.
    """
    def __init__(self, config_path):
        with open(config_path, "r") as config:
            data = yaml.load(config)

        supported_fields = ["model_structure", "dataset_path", "epochs", "learning_rate", "batch_sizes", "image_size",
                            "n_classes", "load_path", "use_augs", "class_weights", "n_processes", "debug",
                            "normalization_params", "class_labels", "class_mapping"]
        for key in data.keys():
            if key not in supported_fields:
                warnings.warn("Unknown Field in config file: {}: {}".format(key, data[key]), SyntaxWarning)

        model_name = data.get("model_structure", "undefined")
        if model_name == "u_net":
            self.model_structure = u_net
        elif model_name == "segnet":
            self.model_structure = segnet
        elif model_name == "e_net":
            self.model_structure = e_net
        elif model_name == "erfnet":
            self.model_structure = erfnet
        else:
            raise AttributeError("Unknown model structure: {}".format(model_name))

        self.dataset_path = data.get("dataset_path", None)
        self.dataset_path = self.dataset_path + "/" if not self.dataset_path.endswith("/") else self.dataset_path
        assert os.path.isdir(self.dataset_path), "Invalid Dataset path: {}".format(self.dataset_path)

        self.epochs = data.get("epochs", 0)
        self.learning_rate = float(data.get("learning_rate", 1e-4))
        self.batch_sizes = data.get("batch_sizes", {"train": 1, "validation": 1, "test": 1})
        self.image_size = data.get("image_size", [512, 512])
        self.n_classes = data.get("n_classes", 7)
        self.load_path = data.get("load_path", None)
        self.use_augs = data.get("use_augs", False)
        self.class_weights = data.get("class_weights", None)
        self.n_processes = data.get("n_processes", 8)
        self.debug = data.get("debug", False)
        self.mean, self.std = data.get("normalization_params", [None, None])
        self.class_labels = data.get("class_labels", None)
        self.class_mapping = data.get("class_mapping", None)

    def save_config(self, save_path):
        class_dict = self.__dict__
        class_dict["model_structure"] = class_dict["model_structure"].__name__

        with open(save_path, 'w') as outfile:
            yaml.dump(class_dict, outfile, default_flow_style=False)
