import yaml
from models.u_net import u_net
from models.segnet import segnet
from models.e_net import e_net
from models.erfnet import erfnet
import os


class Configuration:
    def __init__(self, config_path):
        with open(config_path, "r") as config:
            try:
                data = yaml.load(config)
            except yaml.YAMLError as exc:
                print(exc)

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
        assert os.path.isdir(self.dataset_path), "Invalid Dataset path: {}".format(self.dataset_path)

        self.epochs = data.get("epochs", 0)
        self.learning_rate = float(data.get("learning_rate", 1e-4))
        self.batch_sizes = data.get("batch_sizes", {"train": 1, "validation": 1, "test": 1})
        self.image_size = data.get("image_size", [512, 512])
        self.n_classes = data.get("n_classes", 7)
        self.load_path = data.get("load_path", None)
        self.use_augs = data.get("use_augs", False)
        self.debug = data.get("debug", False)
        self.use_class_weights = data.get("class_weights", True)
        self.n_processes = data.get("n_processes", 8)

    def save_config(self, save_path):
        class_dict = self.__dict__
        class_dict["model_structure"] = class_dict["model_structure"].__name__

        with open(save_path, 'w') as outfile:
            yaml.dump(class_dict, outfile, default_flow_style=False)
