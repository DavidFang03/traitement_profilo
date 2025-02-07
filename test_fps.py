import cv2
from profilo_film import get_min_frame_and_max_frame

vidcap = cv2.VideoCapture('./vids/0702/1_testscale.mp4')
fps = vidcap.get(cv2.CAP_PROP_FPS)
print(vidcap.isOpened())

print(f"{fps} frames per second")
