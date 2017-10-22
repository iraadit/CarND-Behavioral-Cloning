import csv
import cv2
import numpy as np

EPOCHS = 6
STEERING_CORRECTION = 0.25
BATCH_SIZE = 64
TOP_CROP = 70
BOT_CROP = 25

def get_lines(folder_path = '../data', skip=True):
	lines = []
	with open(folder_path + '/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		if skip:
			next(reader, None)
		for line in reader:
			lines.append(line)
	return lines
	
def get_samples(lines, folder_path = '../data/IMG/', correction = STEERING_CORRECTION):
	images_paths = []
	measurements = []
	for line in lines:
		steering_center = float(line[3])
		
		# create adjusted steering measurements for the side camera images
		steering_left = steering_center + correction
		steering_right = steering_center - correction

		# read in images from center, left and right cameras
		img_center = folder_path + line[0].split('/')[-1] #cv2.imread(path + sample[0].split('/')[-1]) #process_image(np.asarray(Image.open(path + sample[0])))
		img_left = folder_path + line[1].split('/')[-1] #cv2.imread(path + sample[1].split('/')[-1]) #process_image(np.asarray(Image.open(path + sample[1])))
		img_right = folder_path + line[2].split('/')[-1] #cv2.imread(path + sample[2].split('/')[-1]) #process_image(np.asarray(Image.open(path + sample[2])))

		# add images paths and angles to data set
		images_paths += [img_center, img_left, img_right]
		measurements += [steering_center, steering_left, steering_right]
		
	return list(zip(images_paths, measurements))

# Get samples
lines = get_lines()
samples = get_samples(lines)
print('Total samples:', len(samples))

# Split samples between train and val sets
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)




import sklearn
import random

def horizontal_flip(image, steering):
	random_value = random.random() # random number in range [0.0,1.0)
	if random_value < 0.5:
		image = cv2.flip(image, 1) # or np.fliplr(image)
		steering = -steering	
	return image, steering

def preprocessing(image, steering):
	image, steering = horizontal_flip(image, steering)
	return image, steering

def generator(samples, training = False, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		samples = sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				img_path = batch_sample[0]
				image = cv2.imread(img_path)
				steering = batch_sample[1]
				if training:
					image, steering = preprocessing(image, steering)
				images.append(image)
				angles.append(steering)

			X = np.array(images)
			y = np.array(angles)
			yield sklearn.utils.shuffle(X, y)

# Create the generators
train_generator = generator(train_samples, training = True, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)





from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
import matplotlib.pyplot as plt

def model_preprocessing():
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((TOP_CROP, BOT_CROP), (0,0))))
	return model

def basic_model():
	model = model_preprocessing()
	model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Flatten())
	model.add(Dense(1))
	return model

def LeNet_model():
	model = model_preprocessing()
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
	return model

def Nvidia_model():
	model = model_preprocessing()
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
	return model
	
def Nvidia_model_dropout():
	model = model_preprocessing()
	model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dropout(0.5))
	model.add(Dense(50))
	model.add(Dropout(0.5))
	model.add(Dense(10))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	return model

# Compile the model
model = Nvidia_model_dropout()
model.compile(loss='mse', optimizer='adam')

# Save visualization of the model
from keras.utils.visualize_util import plot
plot(model, to_file='model.jpg', show_shapes=True, show_layer_names=False)

# Train the model using the generators
history_object = model.fit_generator(train_generator, samples_per_epoch = len(train_samples), \
								validation_data = validation_generator, nb_val_samples = len(validation_samples), \
								nb_epoch = EPOCHS, verbose=1)

# Save the model
model.save('model.h5')


def show_history(history_object):
	### print the keys contained in the history object, as well as training loss and validation loss
	print('History keys')
	print(history_object.history.keys())
	print('Training Loss')
	print(history_object.history['loss'])
	print('Validation Loss')
	print(history_object.history['val_loss'])

	### plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.savefig('losses')
	plt.show()
	
show_history(history_object)

#~ model = load_model('model.h5')

def model_summary(model):
	for layer in model.layers:
		print(layer.get_weights())
	print("model summary: \n{}\n".format(model.summary()))
	print("model parameters: \n{}\n".format(model.count_params()))
	
model_summary(model)

exit()
