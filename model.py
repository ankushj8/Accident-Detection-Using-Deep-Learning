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
