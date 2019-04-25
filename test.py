from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from imgaug import augmenters as iaa
def augment(image):
	aug_1 = iaa.Crop(
		px = (),
		keep_size = False)
	image = aug_1.augment_image(image)
	return image

augment('/Users/apple/Google Drive/angiogram-trial/good-bad/good/angio1-frame20.jpg' )
print(image.shape)