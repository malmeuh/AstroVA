# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:51:37 2016

@author: malvache
"""

from frameFusion2 import *
from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('Images/Orion/'):
    f.extend(filenames)
    break

# Initialize
im = cv2.imread('Images/Orion/'+f[0])
F = FrameFusion(im)
F.show()

for i in range(1, len(f)):
    filename = f[i]
    im = Image.open('Images/'+filename)
    a = np.asarray(im)
    F.pile_up(a)
    F.show()
