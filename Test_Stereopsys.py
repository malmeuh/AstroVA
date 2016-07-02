# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:51:37 2016

@author: malvache
"""

from frameFusion2 import *
import frameGrabber  # Wrap the frame grabbing process
import Queue


def run():

    # Initialize frame buffer for VA
    f = FrameFusion()

    # Get the inputs
    frame_source = frameGrabber.PictsFile('Pics/')

    pict_queue = Queue.Queue()
    grabber = frame_source.populate(pict_queue, 1)

    processing = f.process_pict(pict_queue, show=True)  # Blocking call


    grabber.join()
    processing.join()

    frame_source.release()

    print "Bye bye.."

if __name__ == "__main__":
    run()