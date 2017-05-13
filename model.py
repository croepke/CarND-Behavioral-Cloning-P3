import csv 
import cv2
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = []
with open('driving_data.p', mode='rb') as f:
	driving_data = pickle.load(f)

data, labels = driving_data['images'], driving_data['labels']
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=0)

def generator(data, labels, batch_size):
	start = 0
	end = start + batch_size
	n = data.shape[0]

	while True:

		image_files = data[start:end]
		images = []
		for image_file in image_files:
			image = cv2.imread(image_file)
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

batch_size = 16
train_generator = generator(X_train, y_train, batch_size)
validation_generator = generator(X_val, y_val, batch_size)
train_steps = X_train.shape[0]//batch_size + 1
val_steps = X_val.shape[0]//batch_size + 1

model = Sequential()
model.add(Lambda(lambda x:  x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(24, (5, 5), activation="relu"))
model.add(Convolution2D(36, (5, 5), activation="relu"))
model.add(Convolution2D(48, (5, 5), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(generator=train_generator, steps_per_epoch=train_steps, 
					validation_data=validation_generator, 
					validation_steps=val_steps, epochs=5)
model.save('model.h5')
