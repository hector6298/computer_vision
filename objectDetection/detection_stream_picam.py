import argparse
from os import path
import time
import logging
import sys
import numpy as np
import cv2
from detector import detector
from picamera.array import PiRGBArray
from picamera import PiCamera

predictor = detector()
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))


time.sleep(0.1)

frame_rate_calc = 1
freq = cv2.getTickFrequency()

for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t1 = cv2.getTickCount()
    image = frame.array

    print("FPS: {0:.2f}".format(frame_rate_calc))
    cv2.putText(img, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

    result = predictor.detect(img)

    for obj in result:
        print('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                    format(obj[0], obj[1], obj[3], obj[2]))

        cv2.rectangle(img, obj[0], obj[1], (0, 255, 0), 2)
        cv2.putText(img, '{}: {:.2f}'.format(obj[3], obj[2]),
                    (obj[0][0], obj[0][1] - 5),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)


    # show the frame
    cv2.imshow("Stream", img)
    key = cv2.waitKey(1) & 0xFF

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1
    rawCapture.truncate(0)
  

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
