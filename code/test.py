import scipy.io
import os

# Percorsi
meta_file = "dataset/car_devkit/devkit/cars_meta.mat"
anno_file = "dataset/car_devkit/devkit/cars_train_annos.mat"
images_dir = "dataset/cars_train/cars_train"
reduced_dir = "dataset/cars_train/images_small"
os.makedirs(reduced_dir, exist_ok=True)

# Carica le annotazioni
#scipy is a Python library for scientific and numerical computing,
#built on top of NumPy. It provides advanced functions for 
#mathematics, signal processing, optimization, statistics, and more.
annotations = scipy.io.loadmat(anno_file)['annotations'][0]
meta = scipy.io.loadmat(meta_file)['class_names'][0]
# Gruppo immagini per classe

for ann in annotations:
    fname = ann['fname'][0]       # nome file
    cls_id = int(ann['class'][0][0]) # id classe
    #print(fname+" "+str(cls_id))
    
print(ann)
print("\n"+ann['fname'][0]+"\n")
print(ann['class'][0][0])
print(ann['bbox_x1'][0][0])
print(ann['bbox_y1'][0][0])
print(ann['bbox_x2'][0][0])
print(ann['bbox_y2'][0][0])


class_name=[c[0] for c in meta]
for c in class_name:
    pass
print(c)