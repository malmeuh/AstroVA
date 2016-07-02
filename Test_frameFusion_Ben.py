# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:51:37 2016

@author: malvache
"""

from frameFusion2 import *
from os import walk

dir_pics = 'Images/'

# Needed for equalizeHist, only works on single channel 8bits pictures
def toGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

f = []
for (dirpath, dirnames, filenames) in walk(dir_pics):
    f.extend(filenames)
    break

#Initialize
im = toGray(cv2.imread('Images/'+f[0]))
F = FrameFusion(im)
F.show()

for item in f:  # More pythonic
    im = toGray(cv2.imread('Images/'+item))
    a = np.asarray(im)
    F.pile_up(a)
    F.show()
