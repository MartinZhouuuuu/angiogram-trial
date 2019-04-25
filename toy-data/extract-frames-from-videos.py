import cv2 
def extract_frames(path):
	video_object = cv2.VideoCapture(path)

	count = 0
	success = 1

	while success:
		success, image = video_object.read()

		cv2.imwrite('frame%d'%count, image)
		count += 1

