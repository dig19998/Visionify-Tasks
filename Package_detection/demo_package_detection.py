import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import imutils
from imutils.video import VideoStream
from imutils.video import FPS



# cap = cv2.VideoCapture(0)
model = cv2.dnn.readNetFromTensorflow('/home/digvijayyadav48/package_detector/training-temp/digvijayyadav48/confident_leavitt/frozen_inference_graph.pb',
     '/home/digvijayyadav48/package_detector/training-temp/digvijayyadav48/confident_leavitt/config.pbtxt')

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# vs = VideoStream(src='/home/digvijayyadav48/Downloads/pexels-kampus-production-7835184.mp4').start()
# time.sleep(2.0)
fps = FPS().start()

while True:
    frames = vs.read()
    frames = imutils.resize(frames, width=400)
    rows, cols, chanels = frames.shape
     # Use the given image as input, which needs to be blob(s).
    model.setInput(cv2.dnn.blobFromImage(frames,size = (300,300), swapRB=True, crop=False))
    # model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
 
# Runs a forward pass to compute the net output
    networkOutput = model.forward()
# obj = []
 
# Loop on the outputs
    objs = 0
    for detection in networkOutput[0,0]:
    
        score = float(detection[2])
        if score > 0.8:
    
     
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
 
        #draw a blue rectangle around detected objects
            cv2.rectangle(frames, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
            if score != 0:
                objs+=1
    
        text = f'Total Packages:{objs}'
        org = (50,50)
        cv2.putText(frames, text, org,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) 
 
# Show the image with a rectagle surrounding the detected objects 
# cv2.displayOverlay('blob', text)
    cv2.imshow("frame",frames)

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()