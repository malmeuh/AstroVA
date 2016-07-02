# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:19:46 2016

@author: malvache

Class to combine different pictures on top of one another.
"""

import cv2
import numpy as np

class FrameFusion:

    # The constructor, on top of an initial frame
    def __init__(self, frame_first, gamma = 1.0, motion_compensation = False):
        # Define settings
        self.n_fused_frames = 0

        # Allocate buffer
        self.last_frame = np.float32(frame_first) #last frame
        self.eq_frame = np.float32(frame_first) #equalized frame
        self.acc_frame = np.float32(frame_first) #Sum of all frames
        self.display_frame = np.float32(frame_first) #Frame to display

    def show(self):

        # Show the current combined picture
        print "Showing frame {}".format(self.n_fused_frames)

        # Do all the resizing beforehand
        frame_fusion_resize = cv2.resize(self.display_frame, (800,600))

        cv2.namedWindow("FrameFusion")
        cv2.imshow("FrameFusion", np.uint8(frame_fusion_resize))
        cv2.waitKey(1000)
#        cv2.destroyAllWindows()
    
    def pile_up(self, new_frame):
        """
        Add a new frame to the current accumulation

        @param new_frame:
        @return: number of frames in the current pile
        """
#        cv2.equalizeHist(new_frame, self.eq_frame)        
        
        cv2.accumulate(new_frame, self.acc_frame)  #Second argument must be a float
        cv2.normalize(self.acc_frame, self.display_frame, 0., 255., cv2.NORM_MINMAX) #Normalize from 0 to 255
        
        # Update and return
        self.n_fused_frames += 1

        return self.n_fused_frames