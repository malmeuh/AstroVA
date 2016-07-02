# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:11:48 2016

@author: malvache
"""

import numpy as np
import cv2

# Load an color image in grayscale
img = cv2.imread('Images/20160501205032.jpg',0)

# Draw a circle
#img = cv2.circle(img,(447,63), 63, (0,0,255), 3)

# Add text
#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)

# Change pixel value
#img[100:120,100:120] = [255,255,255]

# Equalize histogram for each color
#b,g,r = cv2.split(img)
#b = cv2.equalizeHist(b)
#g = cv2.equalizeHist(g)
#r = cv2.equalizeHist(r)
#img1 = cv2.merge((b,g,r))

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img2 = clahe.apply(img)

# Save image
cv2.imwrite('test.png',img)

# Display image
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.imshow('image',img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()