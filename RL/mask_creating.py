# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 22:27:48 2024

@author: meghislain
"""
import scipy
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import disk, square, rectangle, cube, octahedron, ball, octagon, star

def observeEnvAs2DImage(self, pos = None, dose_q = 1, form = "ball", targetSize = 2):
    """
    The noise thing is a work in progress
    """

    envImg1 = np.zeros((self.depth, self.num_row, self.num_col), np.float64)
    ref_position = [int(self.depth / 2), int(self.num_row / 2), int(self.num_col / 2)]
    
    if self.form == "cube":
        target = (dose_q/2)*cube(2*(targetSize+2) + 1) 
    if self.form == "ball":
        target = (dose_q/2)*ball(targetSize+2)
    if pos is not None:
        targetCenter = ref_position
    targetCenterInPixels = np.array(np.round(targetCenter), dtype=int)
    
    envImg2 = np.zeros((self.depth, self.num_row, self.num_col), np.float64)
    if self.form == "cube":
        target = (dose_q)*cube(2*(targetSize+1) + 1) 
    if self.form == "ball":
        target = (dose_q)*ball(targetSize+1)
    if pos is not None:
        targetCenter = ref_position
    envImg2[targetCenterInPixels[0] - int(targetSize): targetCenterInPixels[0] + int(targetSize + 1),
            targetCenterInPixels[1] - int(targetSize+1): targetCenterInPixels[1] + int(targetSize + 2),
            targetCenterInPixels[2] - int(targetSize+1): targetCenterInPixels[2] + int(targetSize + 2)] = target[1:-1,:,:]
    
    envImg = envImg1+envImg2 
    shift = np.zeros(3)
    shift[0] = pos[0] - targetCenterInPixels[0]
    shift[1] = pos[1] - targetCenterInPixels[1]
    shift[2] = pos[2] - targetCenterInPixels[2]
    envImg_shifted = scipy.ndimage.shift(envImg, shift, order=1)
    
    return envImg_shifted
        
def observeEnvAs2DImage_plain(self, pos = None, dose_q = 1, form = "ball", targetSize = 2):
    """
    The noise thing is a work in progress
    """
    envImg = np.zeros((self.depth, self.num_row, self.num_col), np.float64)
    ref_position = [int(self.depth / 2), int(self.num_row / 2), int(self.num_col / 2)]
    
    if self.form == "cube":
        target = (dose_q)*cube(2*targetSize + 1)
    if self.form == "ball":
        target = (dose_q)*ball(targetSize)
    if pos is not None:
        targetCenter = ref_position
    targetCenterInPixels = np.array(np.round(targetCenter), dtype=int)
    envImg[targetCenterInPixels[0] - int(targetSize): targetCenterInPixels[0] + int(targetSize + 1),
            targetCenterInPixels[1] - int(targetSize): targetCenterInPixels[1] + int(targetSize + 1),
            targetCenterInPixels[2] - int(targetSize): targetCenterInPixels[2] + int(targetSize + 1)] = target
    shift = np.zeros(3)
    shift[0] = pos[0] - targetCenterInPixels[0]
    shift[1] = pos[1] - targetCenterInPixels[1]
    shift[2] = pos[2] - targetCenterInPixels[2]
    envImg_shifted = scipy.ndimage.shift(envImg, shift, order=1)
    
    return envImg_shifted