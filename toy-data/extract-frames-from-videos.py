import cv2 
import os
def extract_frames(path):
	video_object = cv2.VideoCapture(path)

	count = 0
	success = 1
	while success:
		success, image = video_object.read()
		count += 1
		if not os.path.exists(
			'/Users/apple/Google Drive/angiogram-trial/extracted-frames/%s'%path.replace('.mp4','')
			):
			os.makedirs('/Users/apple/Google Drive/angiogram-trial/extracted-frames/%s'%path.replace('.mp4',''))
		cv2.imwrite(
			'/Users/apple/Google Drive/angiogram-trial/extracted-frames/%s/frame%d.jpg'%(path.replace('.mp4',''),count), image)
	
	os.remove(
		'/Users/apple/Google Drive/angiogram-trial/extracted-frames/%s/frame%d.jpg'%(path.replace('.mp4',''),count)
		)
extract_frames('nejm.mp4')
