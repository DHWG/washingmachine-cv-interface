#!/usr/bin/env python

import glob
import cv2
import matplotlib.pyplot as plt

labels = {}

with open('labels.txt', 'r') as f:
    for l in f:
        file, label = l.strip().split(',')
        labels[file] = label

print('Labels: {}'.format(len(labels)))

plt.ion()
with open('labels.txt', 'w') as of:
    try:
        for f in glob.glob('*.png'):
            if f not in labels:
                img = cv2.imread(f)
                plt.imshow(img)
                print('Img: ' + f)
                label = raw_input()
                labels[f] = label
    finally:
        for (file, label) in labels.iteritems():
            of.write(file + ',' + label + '\n')
