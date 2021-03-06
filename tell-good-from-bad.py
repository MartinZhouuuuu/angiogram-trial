from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import *
model = Sequential()
#convolution
model.add(Conv2D(32,(3,3), activation = 'relu', 
	input_shape = (360,360,1)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(32,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Flatten())
'''model.add(Dense(128,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(32,activation = 'relu'))
model.add(Dropout(0.5))'''
model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(
	rescale =  1./255,
	zoom_range = 0.05,
	fill_mode = 'nearest',
	width_shift_range = 0.05,
	height_shift_range = 0.05,
	rotation_range = 0.2,
	horizontal_flip = True,
	vertical_flip = True
	)
train_generator = train_datagen.flow_from_directory(
	'good-bad/train',
	target_size = (360,360),
	batch_size = 16,
	class_mode = 'binary',
	color_mode = 'grayscale',
	)
test_datagen = ImageDataGenerator(
	rescale = 1./255,
	)

test_generator = test_datagen.flow_from_directory(
	'good-bad/test',
	target_size = (360,360),
	batch_size = 16,
	class_mode = 'binary',
	color_mode = 'grayscale')

model.fit_generator(
	train_generator,
	steps_per_epoch = 6000,
	epochs = 30,
	validation_data = test_generator,
	validation_steps = 100
	)
'''validation_data = test_generator,
validation_steps = 1000'''
