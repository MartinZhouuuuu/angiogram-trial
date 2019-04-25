from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from imgaug import augmenters as iaa
model = Sequential()
#convolution
model.add(Conv2D(32,(5,5), activation = 'relu', 
	input_shape = (360,360,1)))
#model.add(BatchNormalization())
model.add(Conv2D(32,(5,5), activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(5,5),activation = 'relu'))
#model.add(BatchNormalization())
model.add(Conv2D(64,(5,5),activation = 'relu'))
#model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
#convert 3D tensor to 1D
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
	rescale =  1./255,
	zoom_range = 0.05,
	fill_mode = 'nearest',
	featurewise_center = True,
	featurewise_std_normalization = True,
	width_shift_range = 0.05,
	height_shift_range = 0.05,
	rotation_range = 0.2
	)
train_generator = train_datagen.flow_from_directory(
	'good-bad',
	target_size = (360,360),
	batch_size = 1,
	class_mode = 'binary',
	color_mode = 'grayscale',
	)
'''test_datagen = ImageDataGenerator(
	rescale = 1./255,
	preprocessing_function = augmentation)

test_generator = test_datagen.flow_from_directory(
	'testSet',
	target_size = (28,28),
	batch_size = 1,
	class_mode = 'categorical',
	color_mode = 'grayscale')'''

model.fit_generator(
	train_generator,
	steps_per_epoch = 600,
	epochs = 30,
	)
'''validation_data = test_generator,
validation_steps = 1000'''
