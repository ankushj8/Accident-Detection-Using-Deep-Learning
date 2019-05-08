import glob
import numpy as np
from sklearn.model_selection import train_test_split 

'''
	Loading all the Positive and negative according to the desired 
	file path and the load_set preprocess the data in which each frame
	is extracted to a paricular size of (144,256)

'''

def load_set(img_path):
	img = load_img(img_path)
	tmp = skimage.color.rgb2gray(numpy.array(img))
	tmp = transform.resize(tmp, (144, 256))
	return tmp

'''
	Loading all the Positive and negative according to the desired 
	file path and the load_set preprocess the data in which each frame
	is extracted to a paricular size of (144,256) and is horizontally filpped

'''

def horizontal_flip(img_path):
	img = load_img(img_path)
	tmp = skimage.color.rgb2gray(numpy.array(img))
	tmp = skimage.transform.resize(tmp, (144, 256))
	tmp = numpy.array(tmp)
	tmp = numpy.flip(tmp, axis = 1)
	return tmp


'''
	Loading all the Positive and negative files assigned to varaiable
	neg and pos respectively
	All files contains both the files paths

'''
img_filepath = "/home/ritwik/Desktop/MINI PROJECT/"
pos = glob.glob(img_filepath + '99frames/*.mp4')
neg = glob.glob(img_filepath + 'negative/*.mp4')
all_files =  np.concatenate((pos, neg[0:len(pos)]))

#print(len(neg),len(pos))
#print(all_files)       


'''
	label matrix is used to make one hot encoding ie [0 1] for
	positve data and [1 0] for negative data

'''


def label_matrix(values):
    
    n_values = np.max(values) + 1    
    return np.eye(n_values)[values] 

labels = np.concatenate(([1]*len(pos), [0]*len(neg[0:len(pos)])))  
labels = label_matrix(labels)    
#print(len(labels))      

x_train, x_t1, y_train, y_t1 = train_test_split(all_files, labels, test_size=0.40, random_state=0) 
print(x_train)
print(y_train)


'''

import the required libraries for deep learning
use autoencoders for classification

'''

import keras
from keras.models import Model 
from keras.layers import Input,Dense,TimeDistributed
from keras.layers import LSTM

batch_size = 15
num_classes = 2
epochs = 30

row_hidden = 128
col_hidden = 128


frame , row, col =(99,144,256)

x =Input(shape=(frame, row, col))
encoded_rows = TimeDistributed(LSTM(row_hidden))(x) 
encoded_columns =LSTM(col_hidden)(encoded_rows)

prediction = Dense(num_classes, activation='softmax')(encoded_columns)

model = Model(x, prediction)

model.compile(loss='categorical_crossentropy', 
				optimizer='NAdam',               
				metrics=['accuracy']) 

'''				
