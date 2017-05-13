import pickle
import csv
import cv2
import numpy as np

samples = []

with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)

images = []
measurements = []

for sample in samples:
	source_path_center = sample[0]
	source_path_left = sample[1] 
	source_path_right = sample[2]
	
	filename_center = source_path_center.split('/')[-1]
	filename_left = source_path_left.split('/')[-1]
	filename_right = source_path_right.split('/')[-1]

	current_path_center = './data/IMG/' + filename_center
	current_path_left = './data/IMG/' + filename_left
	current_path_right = './data/IMG/' + filename_right

	correction = 0.2
	measurement_center = float(sample[3])
	measurement_left = measurement_center + correction
	measurement_right = measurement_center - correction

	image_center = cv2.imread(current_path_center)
	image_left = cv2.imread(current_path_left)
	image_right = cv2.imread(current_path_right)

	images.append(current_path_center)
	images.append(current_path_left)
	images.append(current_path_right)

	measurements.append(measurement_center)
	measurements.append(measurement_left)
	measurements.append(measurement_right)

	image_center_f = cv2.flip(image_center,1)
	image_left_f = cv2.flip(image_left,1)
	image_right_f = cv2.flip(image_right,1)	

	image_center_f_path = current_path_center.replace('.jpg', '_f.jpg')
	image_left_f_path = current_path_left.replace('.jpg', '_f.jpg')
	image_right_f_path = current_path_right.replace('.jpg', '_f.jpg')

	cv2.imwrite(image_center_f_path,image_center_f)
	cv2.imwrite(image_left_f_path,image_left_f)
	cv2.imwrite(image_right_f_path,image_right_f)

	images.append(image_center_f_path)
	images.append(image_left_f_path)
	images.append(image_right_f_path)

	measurements.append(measurement_center * -1)
	measurements.append(measurement_left * -1)
	measurements.append(measurement_right * -1)

X = np.array(images)
y = np.array(measurements)

data = {'images': np.array(X), 'labels': np.array(y)}

# Save to pickle file
with open('driving_data.p', mode='wb') as f:
	pickle.dump(data, f)