# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:51:37 2016

@author: malvache
"""

from frameFusion2 import *
from os import walk
from PIL import Image

dir_pics = 'Images/Orion/'

f = []
for (dirpath, dirnames, filenames) in walk(dir_pics):
    f.extend(filenames)

# Initialize
im = cv2.imread(dir_pics+f[0])
F = FrameFusion(im)
F.show()

for i in range(1, len(f)):
    filename = f[i]
    im = Image.open(dir_pics+filename)
    a = np.asarray(im)
    F.pile_up(a)
    F.show()


