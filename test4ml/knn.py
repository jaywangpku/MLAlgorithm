#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
import time

time_start = time.time()
with open("train-labels.idx1-ubyte", 'rb') as lbpath:
    magic, n = struct.unpack('>II', lbpath.read(8))
    train_labels = np.fromfile(lbpath, dtype=np.uint8)
with open("train-images.idx3-ubyte", 'rb') as imgpath:
    magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
    train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(train_labels), 784)
with open("t10k-labels.idx1-ubyte", 'rb') as tlbpath:
    tmagic, tn = struct.unpack('>II', tlbpath.read(8))
    test_labels = np.fromfile(tlbpath, dtype=np.uint8)
with open("t10k-images.idx3-ubyte", 'rb') as timgpath:
    tmagic, tnum, trows, tcols = struct.unpack('>IIII', timgpath.read(16))
    test_images = np.fromfile(timgpath, dtype=np.uint8).reshape(len(test_labels), 784)

knn = neighbors.KNeighborsClassifier(algorithm = 'auto', leaf_size = 30, n_neighbors = 10, weights = 'uniform')
knn.fit(train_images, train_labels)
predictLabel = knn.predict(test_images)

match = 0
for i in range(len(test_labels)):
    if(predictLabel[i] == test_labels[i]):
        match += 1

print "error rate: " + str(1-match/1.0/len(test_labels))

time_end = time.time()
time_used = time_end - time_start
print time_used