import pickle
import scipy.io


"""
This script turns the imagenet metadata file into a python pickle file. The pickle file contains a dict with the fields
'id' and 'name'.
"""

mat = scipy.io.loadmat('/home/lennard/Datasets/meta.mat')["synsets"]

class_dict = {}

for element in mat:
    ilsvrc_id = element[0][0][0][0]
    wnid = element[0][1][0]
    name = element[0][2][0]
    class_dict[wnid] = {"id": ilsvrc_id, "name": name}

print(class_dict)
pickle.dump(class_dict, open("/home/lennard/Datasets/ImageNet_classes.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
