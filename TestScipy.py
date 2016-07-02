# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:24:56 2016

@author: malvache

http://www.scipy-lectures.org/advanced/image_processing/
"""

from scipy import misc
import matplotlib.pyplot as plt

# Read image
img = misc.imread('Images/PICT_20150310_214645.jpg')

# Save image
misc.imsave('face.png', img)  # uses the Image module (PIL)

# Display image
plt.imshow(img)
plt.show()

# Img size, !!! s[0], s[1],...
# s = img.