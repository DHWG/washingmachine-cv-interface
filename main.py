#!/usr/bin/env python

import numpy as np
from cv2 import *
import tempfile
import subprocess
import os

THRESHOLD_VALUE = 170
OUTPUT_SIZE = (400, 150)
# https://www.unix-ag.uni-kl.de/~auerswal/ssocr/
SSOCR_EXECUTABLE = '/Users/jeff/Downloads/ssocr-2.16.4/ssocr'

def sampleFrame(video, sampleCount = 10):
    '''Averages a frame from a number of samples to reduce flickering.'''
    samples = []
    frameCount = 0
    while frameCount < sampleCount:
        frameCount += 1
        ret, frame = video.read()
   
        # the digits are orange, but the red channel is oversatured
        # green seems to yield the best results
        green = frame[:,:,1]
    
        # threshold
        _, thresh = threshold(green, THRESHOLD_VALUE, 255, 0)

        # find bounding box
        _, contours, _ = findContours(thresh, RETR_LIST, CHAIN_APPROX_SIMPLE) 
        combinedContour = np.vstack(contours)
        x, y, w, h = boundingRect(combinedContour)

        # extract and normalize area of interest
        aoi = thresh[y:y+h, x:x+w]
        aoi = resize(aoi, OUTPUT_SIZE)
        samples.append(aoi)

    # create average over sample
    multiFrame = np.dstack(samples)
    mean = np.mean(multiFrame, axis = 2)
    # invert
    mean = 255 - mean
    mean = convertScaleAbs(mean)

    return mean


def ocrFrame(frame):
    '''Recognizes the numbers on the captured frame.'''
    _, tempFile = tempfile.mkstemp(suffix = '.png')
    imwrite(tempFile, frame)

    ocr = None
    try:
        ocr = subprocess.check_output([SSOCR_EXECUTABLE, '-d', '4', tempFile])
    except subprocess.CalledProcessError as e:
        # maybe add more error handling here
        ocr = e.output

    os.remove(tempFile)

    hours = int(ocr[0])
    minutes = int(ocr[2:4])
    return (hours, minutes)

video = VideoCapture('/users/jeff/Downloads/IMG_1552.MOV')
frame = sampleFrame(video)
video.release()
time = ocrFrame(frame)

nFrame = cvtColor(frame, COLOR_GRAY2BGR)
putText(nFrame, str(time), (30, 30), FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
imshow('Test', nFrame)
waitKey(10000)
