import os
import numpy as np

import keras 
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

def load_img_data(img_path):
image = load_img(img_path,target_size=(224,224))
x = img_to_array(image)
x = np.expand_dims(x,axis=0)
return x;

def load_data(dir,X,Y):
for filename in os.listdir(dir):
for files in os.listdir(dir+"/"+filename):
img = load_img_data(dir+"/"+filename+"/"+files)
X.append(img)
Y.append(filename[3:])



train_dir1 = "/home/ritwik/Desktop/minor/car-damage-dataset/data1a/training"
test_dir1 = "/home/ritwik/Desktop/minor/car-damage-dataset/data1a/validation"
train_dir2 = "/home/ritwik/Desktop/minor/car-damage-dataset/data2a/training"
test_dir2 = "/home/ritwik/Desktop/minor/car-damage-dataset/data2a/validation"
train_dir3 = "/home/ritwik/Desktop/minor/car-damage-dataset/data3a/training"
test_dir3 = "/home/ritwik/Desktop/minor/car-damage-dataset/data3a/validation"

X_data1 = []
Y_data1 = []
X_data2 = []
Y_data2 = []
X_data3 = []
Y_data3 = []

load_data(train_dir1,X_data1,Y_data1)
load_data(train_dir2,X_data2,Y_data2)
load_data(train_dir3,X_data3,Y_data3)
