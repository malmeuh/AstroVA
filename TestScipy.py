# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:24:56 2016

@author: malvache

http://www.scipy-lectures.org/advanced/image_processing/
"""

from scipy import misc
from frameFusion2 import *
from os import walk
from PIL import Image
import matplotlib.pyplot

# Read images
dir_pics = 'Images/'

f = []
for (dirpath, dirnames, filenames) in walk(dir_pics):
    f.extend(filenames)

#Initialize
im = misc.imread(dir_pics+f[0])
F = FrameFusion(im)
F.show()


img =

# Save image
# misc.imsave('face.png', img)  # uses the Image module (PIL)

# Display image
plt.imshow(img)
plt.show()

# Img size, !!! s[0], s[1],...
# s = img.