import cv2 
import os
  
def FrameCapture(path,folder): 
	cap = cv2.VideoCapture(path)
	property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
	length = int(cv2.VideoCapture.get(cap, property_id))
	count = 0
	success = 1
	cut = length-99

	_dir =  "/home/ritwik/Desktop/Accident-Frames/5sec/"+folder	
	os.mkdir(_dir)
	while success:
		success, image = cap.read()
		if count >= cut :
			n = count-cut
			cv2.imwrite(_dir+"/frame%d.jpg" % n, image) 
		count += 1
		


temp_dir = "/home/ritwik/Desktop/accident-dataset/5sec"
for video_file in os.listdir(temp_dir):
	path = temp_dir+"/"+video_file
	FrameCapture(path,video_file)
	
