import cv2
import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, Lambda, Activation, Cropping2D, ELU
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

samples = []
with open('driving_data.p', mode='rb') as f:
	driving_data = pickle.load(f)

data = driving_data['images']
labels = driving_data['labels']

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=0)

def preprocess(image):
    normalized = image/255.0 - 0.5
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

# hyperparameters and variables
batch_size = 16
train_generator = generator(X_train, y_train, batch_size)
validation_generator = generator(X_val, y_val, batch_size)
train_steps = X_train.shape[0]//batch_size + 1
val_steps = X_val.shape[0]//batch_size + 1
L2_REG_SCALE=0.
LR=1e-4

# model definition
model = Sequential()
model.add(Lambda(preprocess, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding='same', kernel_regularizer=l2(L2_REG_SCALE)))
model.add(ELU())
model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(L2_REG_SCALE)))
model.add(ELU())
model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=l2(L2_REG_SCALE)))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(lr=LR))

# fitting and storing the model
model.fit_generator(generator=train_generator, steps_per_epoch=train_steps, 
					validation_data=validation_generator, 
					validation_steps=val_steps, epochs=10)
model.save('model.h5')
