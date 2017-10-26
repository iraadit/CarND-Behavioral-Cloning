def get_lines(folder_path = '../data', skip=False):
	lines = []
	with open(folder_path + '/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		if skip:
			next(reader, None)
		for line in reader:
			lines.append(line)
	return lines

def get_samples(folder_path = '../data/IMG', correction = STEERING_CORRECTION, one = False):
	data_folders = []
	if one:
		data_folders.append(folder_path)
	else:
		folders = [x[0] for x in os.walk(folder_path)]
		data_folders = list(filter(lambda folder: os.path.isfile(folder + '/driving_log.csv'), folders))

	images_paths = []
	measurements = []
	for data_folder in data_folders:
		lines = get_lines(folder_path = data_folder)
		for line in lines:
			steering_center = float(line[3])

			# create adjusted steering measurements for the side camera images
			steering_left = steering_center + correction
			steering_right = steering_center - correction

			img_folder_path = data_folder + '/IMG/'
			# read in images from center, left and right cameras
			img_center = img_folder_path + line[0].split('/')[-1] #cv2.imread(path + sample[0].split('/')[-1]) #process_image(np.asarray(Image.open(path + sample[0])))
			img_left = img_folder_path + line[1].split('/')[-1] #cv2.imread(path + sample[1].split('/')[-1]) #process_image(np.asarray(Image.open(path + sample[1])))
			img_right = img_folder_path + line[2].split('/')[-1] #cv2.imread(path + sample[2].split('/')[-1]) #process_image(np.asarray(Image.open(path + sample[2])))

			# add images paths and angles to data set
			images_paths += [img_center, img_left, img_right]
			measurements += [steering_center, steering_left, steering_right]

	return list(zip(images_paths, measurements))

# Get samples
samples = get_samples(folder_path = '../data')
