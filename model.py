import csv 
import cv2
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = []
with open('driving_data.p', mode='rb') as f:
	driving_data = pickle.load(f)

with open('driving_data_1.p', mode='rb') as f:
        driving_data_1 = pickle.load(f)

with open('driving_data_recover.p', mode='rb') as f:
	driving_data_recover = pickle.load(f)

with open('driving_data_recover2.p', mode='rb') as f:
        driving_data_recover2 = pickle.load(f)

data = np.concatenate((driving_data['images'], driving_data_1['images'], driving_data_recover['images'], driving_data_recover2['images']))
labels = np.concatenate((driving_data['labels'], driving_data_1['labels'], driving_data_recover['labels'], driving_data_recover2['labels']))
#print(data[:-10])
#print(type(data))
#print(data.shape)
#exit()

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=0)
#X_train = X_train[:1000]
#y_train = y_train[:1000]
#X_val = X_val[:1000]
#y_val = y_val[:1000]

def preprocess_old(image):
    resized = cv2.resize(image, (80, 60))
    normalized = resized/255.0 - 0.5
    return normalized

def preprocess(image):
    from keras.backend import tf
    resized = tf.image.resize_images(image, (80, 160))
    normalized = resized/255.0 - 0.5
    return normalized

def generator(data, labels, batch_size):
	start = 0
	end = start + batch_size
	n = data.shape[0]

	while True:

		image_files = data[start:end]
		images = []
		for image_file in image_files:
			image = cv2.imread(image_file)
			#image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
			images.append(image)

		images = np.array(images)

		X_batch = images
		y_batch = labels[start:end]
		start += batch_size
		end += batch_size
		if start >= n:
			start = 0
			end = batch_size

		yield (X_batch, y_batch)

batch_size = 32
train_generator = generator(X_train, y_train, batch_size)
validation_generator = generator(X_val, y_val, batch_size)
train_steps = X_train.shape[0]//batch_size + 1
val_steps = X_val.shape[0]//batch_size + 1

model = Sequential()
#model.add(Reshape((80, 160, 3), input_shape=(160,320,3)))
#model.add(Lambda(lambda x:  x / 255.0 - 0.5, input_shape=(80,160,3), output_shape=(80,160,3)))
model.add(Lambda(lambda x: preprocess(x), input_shape=(160, 320, 3), output_shape=(80, 160, 3)))
#model.add(Lambda(lambda x:  x / 255.0 - 0.5, input_shape=(160,320,3), output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((30,0), (0,0))))
#print(model.layers[-1].output_shape)
#exit()
#model.add(Reshape((50, 160, 3)))
model.add(Convolution2D(8, (3, 3), activation="relu"))
model.add(Convolution2D(16, (3, 3), activation="relu"))
model.add(Convolution2D(32, (3, 3), activation="relu"))
#model.add(Convolution2D(64, (5, 5), activation="relu"))
#model.add(Convolution2D(48, (5, 5), activation="relu"))
#model.add(Convolution2D(64, (3, 3), activation="relu"))
#model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Flatten())
#model.add(Dense(512))
#model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.fit_generator(generator=train_generator, steps_per_epoch=train_steps, 
					validation_data=validation_generator, 
					validation_steps=val_steps, epochs=5)
model.save('model.h5')
