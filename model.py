import csv 
import cv2
import numpy as np

lines = []

with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines:
	source_path_center = line[0]
	source_path_left = line[1] 
	source_path_right = line[2]
	filename_center = source_path_center.split('/')[-1]
	filename_left = source_path_left.split('/')[-1]
	filename_right = source_path_right.split('/')[-1]

	current_path_center = './data/IMG/' + filename_center
	current_path_left = './data/IMG/' + filename_left
	current_path_right = './data/IMG/' + filename_right

	correction = 0.2
	measurement_center = float(line[3])
	measurement_left = measurement_center + correction
	measurement_right = measurement_center - correction

	image_center = cv2.imread(current_path_center)
	image_left = cv2.imread(current_path_left)
	image_right = cv2.imread(current_path_right)

	images.append(image_center)
	images.append(image_left)
	images.append(image_right)

	measurements.append(measurement_center)
	measurements.append(measurement_left)
	measurements.append(measurement_right)

	images.append(cv2.flip(image_center,1))
	images.append(cv2.flip(image_left,1))
	images.append(cv2.flip(image_right,1))

	measurements.append(measurement_center * -1)
	measurements.append(measurement_left * -1)
	measurements.append(measurement_right * -1)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:  x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(24, 5, 5, activation="relu"))
model.add(Convolution2D(36, 5, 5, activation="relu"))
model.add(Convolution2D(48, 5, 5, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
model.save('model.h5')
