import numpy
import glob
import cv2
import random
from skimage import transform
import skimage
import sklearn
from sklearn.model_selection import train_test_split   ### import sklearn tool

import os
import numpy as np

import keras 
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

def load_data(path,X,i):
	for filename in os.listdir(path):
		x = []
		for files in os.listdir(path+"/"+filename):
			frames = []
			img_path = path+"/"+filename+"/"+files
			img = load_set(img_path)
			x.append(img)
		X[i] = x
		i = i+1	



def load_set(path):
	img = load_img(img_path)
	tmp = skimage.color.rgb2gray(numpy.array(img))
	tmp = transform.resize(tmp, (144, 256))
	return tmp

def load_set(path):
	img = load_img(img_path)
	tmp = skimage.color.rgb2gray(numpy.array(img))
	tmp = skimage.transform.resize(tmp, (144, 256))
	tmp = numpy.array(tmp)
	tmp = numpy.flip(tmp, axis = 1)
	return tmp
i=0
X = []
load_set("/home/ritwik/Desktop/MINI PROJECT/99frames",X,i)
