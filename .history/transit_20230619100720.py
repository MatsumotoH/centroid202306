import os
import time

# 1412
import cv2

import custom618

while True:
    img = cv2.imread('informationpanel.jpg')

    # custom618のnew_idとnew_centroidをprint
    print(new_id, custom618.new_centroid)
    cv2.imshow('Information', img)

    key = cv2.waitKey(1)
    if key == (ord('q')):
        break
