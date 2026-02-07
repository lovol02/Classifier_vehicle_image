import scipy.io
import random
import shutil
import os

# Percorsi
anno_file = "dataset/car_devkit/devkit/cars_train_annos.mat"
images_dir = "dataset/cars_train/cars_train"
reduced_dir = "dataset/cars_train/images_small"
os.makedirs(reduced_dir, exist_ok=True)

# Carica le annotazioni
#scipy is a Python library for scientific and numerical computing,
#built on top of NumPy. It provides advanced functions for 
#mathematics, signal processing, optimization, statistics, and more.
annotations = scipy.io.loadmat(anno_file)['annotations'][0]

# Gruppo immagini per classe
class_to_images = {}
for ann in annotations:
    fname = ann['fname'][0]       # nome file
    cls_id = int(ann['class'][0][0]) # id classe
    class_to_images.setdefault(cls_id, []).append(fname)

# Campiona N immagini per classe, es. 5
for cls_id, files in class_to_images.items():
    sampled_files = random.sample(files, min(5, len(files)))
    for f in sampled_files:
        #Python standard library module that provides high-level operations 
        #on files and directories, like copying, moving, deleting, and archiving.
        shutil.copy(os.path.join(images_dir, f), os.path.join(reduced_dir, f))

print("Dataset ridotto creato con varietà di classi mantenuta.")
