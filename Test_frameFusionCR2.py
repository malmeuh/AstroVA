# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:51:37 2016

@author: malvache
"""

from frameFusion2 import *
from os import walk
from PIL import Image
import rawpy

dir_pics = 'ImagesCR2/'

f = []
for (dirpath, dirnames, filenames) in walk(dir_pics):
    f.extend(filenames)
    break

#Initialize
with rawpy.imread(dir_pics+f[0]) as raw:
    im = raw.postprocess()
#im = cv2.imread(dir_pics+f[0])
F = FrameFusion(im)
F.show()

for i in range(1,len(f)):
    filename = f[i]
    with rawpy.imread(dir_pics+filename) as raw:
        im = raw.postprocess()
    a = np.asarray(im)
    F.pile_up(a)
    F.show()