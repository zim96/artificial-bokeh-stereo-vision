# -*- coding: utf-8 -*-

"""# Artificial *Bokeh* using Stereo Vision

## Load Images and Ground Truths
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys

# Convert to Grayscale
left_img_gray = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
right_img_gray = cv.cvtColor(right_img, cv.COLOR_BGR2GRAY)

"""## Mask Creation Function"""

def createMask(dmap, fRange):
    
    x, y = fRange
    
    if x >= 255:
        print('X value must be less than 255, It should be at most the maximum value in the Disparity Range.')
        return
    
    temp = np.copy(dmap)
    temp[(temp < x) | (temp > y)] = 255
    temp[(temp >= x) & (temp <= y)] = 0
    
    return temp

"""## Stereo Block Matching (Plane Sweeping)
### Sum of Squared Differences
"""

def stereoBM_ssd(im_l, im_r, b_size, disparityRange):
    """ Find disparity image using Sum of Squared Differences. """
    
    print('Starting Block Matching SSD')
    
    m, n = im_r.shape

    # Array to hold depth planes
    dmaps = np.zeros((m, n))
    
    # Block matching with nested for-loops
    for r in range(m):
        minr = max(0, r-b_size)
        maxr = min(m-1, r+b_size)
        for c in range(n):
            minc = max(0, c-b_size)
            maxc = min(n-1, c+b_size)
            
            # Calculate Disparity Range
            mind = disparityRange[0]
            maxd = min(disparityRange[1], n-maxc)
            
            # Construct template from the right image
            template = im_r[minr:maxr, minc:maxc]

            # Search template in the left image - Calculate SSD
            numBlocks = maxd - mind
            scores = cv.matchTemplate(im_l[minr:maxr, minc+mind:minc+maxd], template, method=cv.TM_SQDIFF_NORMED)
            
            # Find the index of the best matched region
            # It is the disparity
            matched_idx = np.argmin(scores)
            dmaps[r, c] = matched_idx + mind
        
        sys.stdout.write('Row %d/%d \r' % (r+1, m))
        sys.stdout.flush()
        
    return dmaps

"""### Normalised Cross Correlation"""

def stereoBM_ncc(im_l, im_r, b_size, disparityRange):
    """ Find disparity image using Normalised Cross Correlation. """
    
    print('Starting Block Matching NCC')
    
    m, n = im_r.shape

    # Array to hold depth planes
    dmaps = np.zeros((m, n))
    
    # Block matching with nested for-loops
    for r in range(m):
        minr = max(0, r-b_size)
        maxr = min(m-1, r+b_size)
        for c in range(n):
            minc = max(0, c-b_size)
            maxc = min(n-1, c+b_size)
            
            # Calculate Disparity Range
            mind = disparityRange[0]
            maxd = min(disparityRange[1], n-maxc)
            
            # Construct template from the right image
            template = im_r[minr:maxr, minc:maxc]

            # Search template in the left image - Calculate NCC
            numBlocks = maxd - mind
            scores = cv.matchTemplate(im_l[minr:maxr, minc+mind:minc+maxd], template, method=cv.TM_CCORR_NORMED)
                
            # Find the index of the best matched region
            # It is the disparity
            matched_idx = np.argmax(scores)
            dmaps[r, c] = matched_idx + mind
        
        sys.stdout.write('Row %d/%d \r' % (r+1, m))
        sys.stdout.flush()
        
    return dmaps

"""## Dynamic Programming (Unconstrained)"""

def stereoDP(im_l, im_r):
    """ Find disparity image using Dynamic Programming. """
    
    print('Starting Dynamic Programming')
    
    m, n = im_r.shape
    dmap_l = np.zeros((m, n))
    dmap_r = np.zeros((m, n))
    
    occlusionCost = 20
    
    # Build Disparity Space Image
    for r in range(m):
        
        dsi = np.zeros((n, n))
        dpath = np.zeros((n, n))
        
        for i in range(0, n):
            dsi[i][0] = i*occlusionCost
            dsi[0][i] = i*occlusionCost
        
        for c in range(n):
            left = im_l[r, c]
            for d in range(0, n):
                right = im_r[r, d]
                cost = np.abs(int(left)-int(right))
                
                # Find path
                min1 = dsi[c-1, d-1]+cost
                min2 = dsi[c-1][d]+occlusionCost
                min3 = dsi[c][d-1]+occlusionCost
                cmin = np.min((min1, min2, min3))
                dsi[c, d] = cmin
                
                if min1 == cmin:
                    dpath[c, d] = 1
                elif min2 == cmin:
                    dpath[c, d] = 2
                elif min3 == cmin:
                    dpath[c, d] = 3
        
        # Explore Path and Set disparity value
        i = n-1
        j = n-1
        while i!=0 and j!=0:
            if dpath[i, j] == 1:
                dmap_l[r, i] = np.abs(i-j)
                dmap_r[r, j] = np.abs(i-j)                
                i = i-1
                j = j-1
            elif dpath[i, j] == 2:
                i = i-1
            elif dpath[i, j] == 3:
                j = j-1
            else:
                print('i =', i, ', j =', j)
                return dsi, dpath
        
        sys.stdout.write('Row %d/%d \r' % (r+1, m))
        sys.stdout.flush()
        
    return dmap_l, dmap_r
