# Importing all required libraries at once
from keras.datasets import mnist

import tensorflow as tf
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Using a dataset as an input
(x_train,y_train), (x_test,y_test)= tf.keras.datasets.fashion_mnist.load_data()

x_train[0].shape

x_train =x_train.reshape((x_train.shape[0],28*28)).astype('float')

x_test =x_test.reshape((x_test.shape[0],28*28)).astype('float')
x_train /=255
x_test/=255

y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

num_classes=y_test.shape[1]

model=Sequential()

# 1st hidden layer 
model.add(Dense(45,input_dim=28*28,activation='relu'))
# 2nd hidden layer 
model.add(Dense(30,activation='relu'))
# 3rd hidden layer 
model.add(Dense(60,activation='relu'))
# 4th hidden layer 
model.add(Dense(50,activation='relu'))
# 5th hidden layer 
model.add(Dense(50,activation='relu'))
# 6th hidden layer 
model.add(Dense(30,activation='relu'))
# 7th hidden layer 
model.add(Dense(10,activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, validation_split=0.7 , epochs=20, batch_size=10000)


result_score = model.evaluate(x_test, y_test)
print(result_score)
