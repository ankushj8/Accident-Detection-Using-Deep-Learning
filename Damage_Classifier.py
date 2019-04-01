import os
import cv2
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img


num_classes = 2
batch_size = 64
epochs = 5

def load_img_data(img_path):
	image = cv2.imread(img_path)
	x = cv2.resize(image, (128, 128))
	x = np.expand_dims(x,axis=0)
	return np.array(x);

def load_data(dir):
	X = []
	Y = []
	count = 0
	for filename in os.listdir(dir):
		
		for files in os.listdir(dir+"/"+filename):

			img = load_img_data(dir+"/"+filename+"/"+files)
			X.append(img)
			Y.append(count)
		count += 1 
	return np.array(X),np.array(Y)
train_dir1 = "/home/ritwik/Desktop/MINOR PROJECT/car-damage-dataset/data1a/training"
test_dir1 = "/home/ritwik/Desktop/MINOR PROJECT/car-damage-dataset/data1a/validation"


x_train,y_train = load_data(train_dir1)
x_test,y_test = load_data(test_dir1)

x_train = x_train.reshape(x_train.shape[0],128,128,3)
x_test = x_test.reshape(x_test.shape[0],128,128,3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /=255



y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)



model= Sequential()


model.add(Conv2D(16, (3, 3),input_shape=(128,128,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(128,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(2))

model.add(Activation('softmax'))

gen = ImageDataGenerator()
train_generator = gen.flow(x_train, y_train, batch_size=batch_size)
model.compile(loss='categorical_crossentropy'
  , optimizer=keras.optimizers.Adam()
 , metrics=['accuracy']
)

model.fit_generator(train_generator, steps_per_epoch=batch_size, epochs=epochs,
	validation_data=(x_test, y_test)
)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', 100*score[1])
predictions = model.predict(x_test)

print(model.summary())

image_path = '/home/ritwik/Desktop/MINOR PROJECT/0007.jpg'
x = load_img_data(image_path)
x = x.reshape(1,128,128,3)
x = x.astype('float32')
x /= 255
ans = model.predict(x).argmax()
print(str(ans))

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, predictions[:,0])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(cm , title='Confusion Matrix')