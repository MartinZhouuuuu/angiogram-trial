import cv2
import glob

def crop_images(path):
	images = [cv2.imread(file) for file in glob.glob(path)]
	count = 0
	for image in images:
		crop_image = image[:,int((image.shape[1]-360)/2):int((image.shape[1]-360)/2+360)]
		cv2.imwrite('/Users/apple/Google Drive/angiogram-trial/good-bad/good-sliced/good-%d.jpg'%count,crop_image)
		count +=1
crop_images('/Users/apple/Google Drive/angiogram-trial/good-bad/good/*.jpg')

'''image = cv2.imread('/Users/apple/Google Drive/angiogram-trial/good-bad/good/angio1-frame20.jpg')
crop_image = image[:,140:500]
cv2.imwrite('/Users/apple/Google Drive/angiogram-trial/good-bad/good-sliced/1.jpg',crop_image)'''