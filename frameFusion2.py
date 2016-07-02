# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:19:46 2016

@author: malvache

Class to combine different pictures on top of one another.
"""

import cv2
import numpy as np
import threading

MAX_FRAME_NUMBER = 12


class FrameFusion:
    # The constructor, on top of an initial frame
    def __init__(self):
        # Define settings
        self.n_fused_frames = 0

        self.list_frame = None

    def _process_queue(self, pict_queue, show):
        import Queue

        while True:
            try:
                pic = pict_queue.get(block=True, timeout=5)
                print "Got a new frame"

                self.pile_up(pic)

                if show:
                    self.show()

            except Queue.Empty:
                print "No more frames"
                break

    def process_pict(self, pict_queue, show=False):
        # Spawn a thread, which will read a picture
        # Parallel threads (1)
        for i in range(1):
            t = threading.Thread(target=self._process_queue, args=(pict_queue, show))
            t.start()

        return t

    def show(self):
        # Show the current combined picture
        print "Showing frame {}".format(self.n_fused_frames)
        frame_size = self.list_frame.shape

        self.display_frame = np.ndarray([frame_size[0], frame_size[1], 3])
        for i in range(3):
            self.display_frame[:, :, i] = np.mean(self.list_frame[:, :, 1:self.n_fused_frames, i], 2)

        # Do all the resizing beforehand
        frame_fusion_resize = cv2.resize(self.display_frame, (800, 600))

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
        if self.n_fused_frames == 0:
            # Allocate buffer
            frame_size = new_frame.shape
            self.list_frame = np.ndarray([frame_size[0], frame_size[1], MAX_FRAME_NUMBER, 3])

#        cv2.equalizeHist(new_frame, self.eq_frame)        
        
        self.list_frame[:, :, self.n_fused_frames, :] = new_frame
        # cv2.normalize(self.acc_frame, self.display_frame, 0., 255., cv2.NORM_MINMAX)  # Normalize from 0 to 255
        
        # Update and return
        self.n_fused_frames += 1

        return self.n_fused_frames