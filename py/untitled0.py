#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:31:48 2018

@author: tsakai
"""

if __name__ == '__main__':
    
    import numpy as np
    from os import listdir
    import glob
    from PIL import Image
    from scipy import ndimage
    import matplotlib.pyplot as plt
    from time import sleep

    resize_depthXXXX_tiff = glob.glob('crop_depth/resize_depth*.tiff')
    #resize_depthXXXX_tiff = resize_depthXXXX_tiff[600:700]
    plt.figure()
    ax = plt.axes()
    for f0, f1 in zip(resize_depthXXXX_tiff[1:], resize_depthXXXX_tiff):
        im0 = np.asarray(Image.open(f0), dtype=np.float32)
        im1 = np.asarray(Image.open(f1), dtype=np.float32)
        im0 = ndimage.gaussian_filter(im0, sigma=3.0)
        im1 = ndimage.gaussian_filter(im1, sigma=3.0)

        ax.clear()
        plt.imshow(im1-im0, cmap='seismic', interpolation='nearest', vmin=-100, vmax=100)
        #plt.colorbar()
        plt.pause(0.01)
        #sleep(0.5)