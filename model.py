import csv 
import cv2
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Activation, Cropping2D, Reshape, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = []
with open('driving_data.p', mode='rb') as f:
	driving_data = pickle.load(f)

with open('driving_data_1.p', mode='rb') as f:
        driving_data_1 = pickle.load(f)

with open('driving_data_3.p', mode='rb') as f:
        driving_data_3 = pickle.load(f)

with open('driving_data_recover.p', mode='rb') as f:
	driving_data_recover = pickle.load(f)

with open('driving_data_recover2.p', mode='rb') as f:
        driving_data_recover2 = pickle.load(f)

with open('driving_data_4.p', mode='rb') as f:
        driving_data_4 = pickle.load(f)

with open('driving_data_5.p', mode='rb') as f:
        driving_data_5 = pickle.load(f)

#data = driving_data['images']
#labels = driving_data['labels']
data = np.concatenate((driving_data['images'], driving_data_1['images'], driving_data_3['images'],driving_data_4['images'],driving_data_5['images'], driving_data_recover['images'], driving_data_recover2['images']))
labels = np.concatenate((driving_data['labels'], driving_data_1['labels'],driving_data_3['labels'],driving_data_4['labels'],driving_data_5['labels'],driving_data_recover['labels'], driving_data_recover2['labels']))
#print(data[:-10])
#print(type(data))
#print(data.shape)
#exit()

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=0)
print(X_train.shape[0])
#X_train = X_train[:1000]
#y_train = y_train[:1000]
#X_val = X_val[:1000]
#y_val = y_val[:1000]

def preprocess_old(image):
    resized = cv2.resize(image, (80, 60))
    normalized = resized/255.0 - 0.5
    return normalized

def preprocess(image):
    #from keras.backend import tf
    #resized = tf.image.resize_images(image, (80, 160))
    normalized = image/255.0 - 0.5
    #normalized = image/255.0 - 0.5
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
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

batch_size = 16
train_generator = generator(X_train, y_train, batch_size)
validation_generator = generator(X_val, y_val, batch_size)
train_steps = X_train.shape[0]//batch_size + 1
val_steps = X_val.shape[0]//batch_size + 1
L2_REG_SCALE=0.
LR=1e-4
model = Sequential()
model.add(Lambda(preprocess, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
#model.add(Convolution2D(16, (3, 3), activation="relu"))
#model.add(Convolution2D(32, (3, 3), activation="relu"))
#model.add(Convolution2D(64, (3, 3), activation="relu"))
#model.add(Convolution2D(64, (5, 5), activation="relu"))
#model.add(Convolution2D(48, (5, 5), activation="relu"))
#model.add(Convolution2D(64, (3, 3), activation="relu"))
#model.add(Convolution2D(64, (3, 3), activation="relu"))
model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding='same', kernel_regularizer=l2(L2_REG_SCALE)))
model.add(ELU())
model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(L2_REG_SCALE)))
model.add(ELU())
model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(L2_REG_SCALE)))
model.add(Flatten())
model.add(Dense(512))
#model.add(Dense(32))
#model.add(Dense(32))
#model.add(Dense(16))
model.add(Dense(1))
#model = Sequential()
#model.add(Lambda(preprocess, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
#model.add(Cropping2D(cropping=((30,0), (0,0))))
#model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
#model.add(Convolution2D(24, (5, 5), activation='elu', strides=(2, 2)))
#model.add(Convolution2D(36, (5, 5), activation='elu', strides=(2, 2)))
#model.add(Convolution2D(48, (5, 5), activation='elu', strides=(2, 2)))
#model.add(Convolution2D(64, (3, 3), activation='elu'))
#model.add(Convolution2D(64, (3, 3), activation='elu'))
#model.add(Dropout(0.2))
#model.add(Flatten())
#model.add(Dense(100, activation='elu'))
#model.add(Dense(50, activation='elu'))
#model.add(Dense(10, activation='elu'))
#model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=LR))

model.fit_generator(generator=train_generator, steps_per_epoch=train_steps, 
					validation_data=validation_generator, 
					validation_steps=val_steps, epochs=5)
model.save('model.h5')
