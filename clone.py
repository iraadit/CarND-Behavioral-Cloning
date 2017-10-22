import csv
import cv2
import numpy as np

lines = []
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		
images = []
measurements = []
for k, line in enumerate(lines):
	if k:
		'''
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = '../data/IMG/' + filename
		image = cv2.imread(current_path)
		images.append(image)
		measurement = float(line[3])
		measurements.append(measurement)
		'''
		steering_center = float(line[3])
		
		# create adjusted steering measurements for the side camera images
		correction = 0.2 # this is a parameter to tune
		steering_left = steering_center + correction
		steering_right = steering_center - correction

		# read in images from center, left and right cameras
		path = '../data/IMG/'
		img_center = cv2.imread(path + line[0].split('/')[-1]) #process_image(np.asarray(Image.open(path + line[0])))
		img_left = cv2.imread(path + line[1].split('/')[-1]) #process_image(np.asarray(Image.open(path + line[1])))
		img_right = cv2.imread(path + line[2].split('/')[-1]) #process_image(np.asarray(Image.open(path + line[2])))

		# add images and angles to data set
		images += [img_center, img_left, img_right]
		measurements += [steering_center, steering_left, steering_right]

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image, 1)) # or np.fliplr(image)
	augmented_measurements.append(-measurement)
	
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
	
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
import matplotlib.pyplot as plt

# Basic
'''
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))
'''

# LeNet
'''
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(120))
#model.add(Dropout(0.5))
model.add(Dense(84))
model.add(Dense(1))
'''

# Nvidia
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, verbose=1)

model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
exit()
