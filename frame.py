import numpy as np
import cv2
import pafy
url = 'https://www.youtube.com/watch?v=WmP_ncZJCi4'
vPafy = pafy.new(url)
play = vPafy.getbest(preftype="webm")
cap = cv2.VideoCapture(play.url)
count = 0
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.imwrite('frame%d.jpg'%count, frame);
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break
