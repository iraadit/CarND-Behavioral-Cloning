import csv
import cv2
import numpy as np
import os
import random
import sklearn
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
import pandas as pd
from keras.callbacks import ModelCheckpoint

EPOCHS = 6
STEERING_CORRECTION = 0.25
BATCH_SIZE = 64
TOP_CROP = 70
BOT_CROP = 25
DATA_PATH = '/Users/iraadit/Datasets/Udacity/Behavorial Cloning/data' #'../data/data'
CSV_PATH = os.path.join(DATA_PATH, "driving_log.csv")
STRAIGHT_DROP = 0.7 # 70%

# data_folders = []
# if one:
# 	data_folders.append(folder_path)
# else:
# 	folders = [x[0] for x in os.walk(folder_path)]
# 	data_folders = list(filter(lambda folder: os.path.isfile(folder + '/driving_log.csv'), folders))

def load_data(csv_path):
	"""
	Load data
	"""
	samples_df = pd.read_csv(csv_path)

	X = samples_df[['center', 'left', 'right']].values
	y = samples_df['steering'].values

	return X, y


def calculate_number_straight_samples_to_drop(y_train):
	"""
	Calculate the number of "straight line" samples to drop at each epoch
	"""
	index = np.where(abs(y_train) < .05)[0]
	drop_num = int(len(index) * STRAIGHT_DROP)
	return drop_num

def random_drop_low_steering(X_train, y_train):
	"""
	Randomly drop low steering ("straight line") samples
	"""
	index = np.where(abs(y_train) < .05)[0]
	#drop_rows = [i for i in index if np.random.randint(10) < STRAIGHT_DROP] # NOT OK because will not get same number of samples every time called
	#drop_rows = random.sample(list(index), int(len(list(index))*STRAIGHT_DROP)) # NOT OK because can choose the same sample several times
	np.random.shuffle(index)
	drop_rows = index[:int(len(index) * STRAIGHT_DROP)]
	X_train = np.delete(X_train, drop_rows, axis=0)
	y_train = np.delete(y_train, drop_rows, axis=0)
	print("Dropped %s samples with low steering"%(len(drop_rows)))
	return X_train, y_train



def random_brightness(image):
	"""
	Randomly modify brightness of the image
	"""
	# Convert to HSV colorspace from RGB colorspace
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	# Generate new random brightness
	rand = 0.3 + np.random.uniform() #random.uniform(0.5,1.0)
	#hsv[:,:,2] = rand*hsv[:,:,2] # problem of overflow when new_value > 255

	# TAKING A LOT OF TIME !!!
	# for i in range(hsv.shape[0]):
	# 	for j in range(hsv.shape[1]):
	# 		new_value = rand*hsv[i,j,2]
	# 		hsv[i,j,2] = 255 if new_value > 255 else new_value

	#hsv[:,:,2] = np.where(rand*hsv[:,:,2] > 255 , 255, rand*hsv[:,:,2]) # NOT WORKING

	hsv[:,:,2] = np.where(rand*hsv[:,:,2].astype(float) > 255. , 255, rand*hsv[:,:,2])

	# Convert back to RGB colorspace
	new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
	return new_img

def random_horizontal_flip(image, steering):
	"""
	Randomly flip the image (one chance on two), and accordingly inverse the steering angle (or not)
	"""
	random_value = random.randint(0,1) # random value : 0 or 1
	if random_value:
		image = cv2.flip(image, 1) # or np.fliplr(image)
		steering = -steering
	return image, steering

def random_shadow(image):
	"""
	Randomly add a shadow on the image
	"""
	top_y = 320*np.random.uniform()
	top_x = 0
	bot_x = 160
	bot_y = 320*np.random.uniform()
	hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	shadow_mask = 0*hsv[:,:,1]
	X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
	Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
	shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
	if np.random.randint(2)==1:
		random_bright = .25+.5*np.random.uniform()
		cond1 = shadow_mask==1
		cond0 = shadow_mask==0
		if np.random.randint(2)==1:
			hsv[:,:,2][cond1] = hsv[:,:,2][cond1]*random_bright
		else:
			hsv[:,:,2][cond0] = hsv[:,:,2][cond0]*random_bright
	image = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
	return image

def convert_to_yuv(image):
	"""
	Convert the image from RGB to YUV
	"""
	return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocessing(image, steering):
	"""
	Preprocess the image, and modify the steering angle accordingly
	Only for training images
	"""
	image, steering = random_horizontal_flip(image, steering)
	image = random_brightness(image)
	image = random_shadow(image)
	return image, steering

def get_random_image_and_steering_angle(center, left, right, steering_angle, data_path):
	"""
	Select one image at random between center, left and right images; and modify the steering angle accordingly
	Also convert the image from BGR to RGB
	Only for training images
	"""
	random = np.random.randint(4)
	if (random == 0):
		img_path = left.strip()
		shift_ang = .25
	if (random == 1 or random == 2):
		img_path = center.strip()
		shift_ang = 0.
	if (random == 3):
		img_path = right.strip()
		shift_ang = -.25
	abs_img_path = os.path.join(data_path, img_path)
	image = cv2.imread(abs_img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	steering = steering_angle + shift_ang
	return image, steering

def get_center_image_and_steering_angle(center, steering_angle, data_path):
	"""
	Select the center
	Also convert the image from BGR to RGB
	Only for validation images
	"""
	img_path = center
	abs_img_path = os.path.join(data_path, img_path)
	image = cv2.imread(abs_img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	steering = steering_angle
	return image, steering

def generator(images_paths, steering_angles, training = False, batch_size=32):
	"""
	Generate the batches of images
	"""
	while 1: # Loop forever so the generator never terminates
		images_paths_in, steering_angles_in = sklearn.utils.shuffle(images_paths, steering_angles)

		if training:
			images_paths_in, steering_angles_in = random_drop_low_steering(images_paths, steering_angles)

		num_samples = len(images_paths_in)

		for offset in range(0, num_samples, batch_size):
			batch_images_paths = images_paths_in[offset:offset+batch_size]
			batch_steering_angles = steering_angles_in[offset:offset+batch_size]

			images = []
			angles = []

			for image_path_3, steering_angle in zip(batch_images_paths, batch_steering_angles):
				center, left, right = image_path_3

				if training:
					image, steering = get_random_image_and_steering_angle(center, left, right, steering_angle, DATA_PATH)
					image, steering = preprocessing(image, steering)
				else:
					image, steering = get_center_image_and_steering_angle(center, steering_angle, DATA_PATH)

				image = convert_to_yuv(image)

				images.append(image)
				angles.append(steering)

			X = np.array(images)
			y = np.array(angles)
			yield X, y


def model_preprocessing():
	"""
	Preprossing step common to all the Keras models
	Normalization of the images
	Cropping of the images
	"""
	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	# crop to take out sky and hood
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
	#model.add(Dropout(0.5))
	model.add(Dense(10))
	#model.add(Dropout(0.5))
	model.add(Dense(1))
	return model

def Commaai_model():
	model = model_preprocessing()
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))
	return model


def show_history(history_object):
	"""
	Print history of the training
	Save losses plot
	"""
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

def model_summary(model):
	"""
	Model summary
	"""
	# for layer in model.layers:
	# 	print(layer.get_weights())
	print("model summary: \n{}\n".format(model.summary()))
	print("model parameters: \n{}\n".format(model.count_params()))


def main():
	"""
	Load train/validation data set and train the model
	"""
	np.random.seed(0)

	# Get samples
	X, y = load_data(CSV_PATH)
	print('Total samples:', len(X))

	# Split samples between train and val sets
	from sklearn.model_selection import train_test_split
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

	print('Train samples:', len(X_train))
	print('Valid samples:', len(X_valid))

	# Now done in the generator for each epoch
	#X_train, y_train = random_drop_low_steering(X_train, y_train)
	#print('Train samples without low steering:', len(X_train))

	# Create the generators
	train_generator = generator(X_train, y_train, training = True, batch_size=BATCH_SIZE)
	validation_generator = generator(X_valid, y_valid, training = False, batch_size=BATCH_SIZE)

	# Compile the model
	model = Nvidia_model()
	model.compile(loss='mse', optimizer='adam')

	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
									 monitor='val_loss',
									 verbose=0,
									 save_best_only=False,
									 mode='auto')

	# Save visualization of the model
	plot(model, to_file='model.jpg', show_shapes=True, show_layer_names=False)

	training_length_with_dropped_straight_samples = len(X_train) - calculate_number_straight_samples_to_drop(y_train)
	print(training_length_with_dropped_straight_samples)
	# Train the model using the generators
	history_object = model.fit_generator(train_generator, samples_per_epoch = training_length_with_dropped_straight_samples, \
									validation_data = validation_generator, nb_val_samples = len(X_valid), \
									nb_epoch = EPOCHS, verbose=1, callbacks=[checkpoint])

	# Save the model
	model.save('model.h5')

	show_history(history_object)

	#~ model = load_model('model.h5')

	model_summary(model)

if __name__ == '__main__':
	main()
