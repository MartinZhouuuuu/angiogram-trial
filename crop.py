import cv2
import glob

def crop_images(path):
	#images = [cv2.imread(file) for file in glob.glob(str(path + '*.jpg'))]
	#for file in glob.glob(path):
		#print(file)
	count = 0
	#for image in images:
	image = cv2.imread(path)
	crop_image = image[:,(image.int(width-360)/2):int(360(width-360)/2)]
	cv2.imwrite('/Users/apple/Google Drive/angiogram-trial/good-bad/good-sliced/%d.jpg'%count,crop_image)
	count +=1
#crop_images('Users/apple/Google Drive/angiogram-trial/good-bad/good/angio1-frame20.jpg')

image = cv2.imread('/Users/apple/Google Drive/angiogram-trial/good-bad/good/angio1-frame20.jpg')
crop_image = image[:,140:500]
cv2.imwrite('/Users/apple/Google Drive/angiogram-trial/good-bad/good-sliced/1.jpg',crop_image)