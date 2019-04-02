# Artificial *bokeh* using Binocular Stereo Vision

A Jupyter Notebook written to develop and demonstrate simple artificial bokeh using stereo block matching and dynamic programming techniques.

## Concept

1. Calculate the depth disparity between a pair of stereo images which correlates with the depth value.
2. Once this is done, a mask of black and white pixels can be created such that pixels where the depth disparity value is in the range specified will be 255 (White) and the rest will be 0 (Black).
3. The entire image is blurred/smoothed using a Gaussian kernel.
4. A hybridised image is formed with guidance from the mask. 'White' pixels will be taken from the smoothed image, 'Black' pixels are taken from the original image.

## Implementation

The entire process is written in a Jupter Notebook together with widgets to allow some interaction with users. Users will be able to specify the 'Mask' range and see the blur change in real-time.

There are 3 techniques to calculate the depth disparity which are Block Matching with Sum-of-Squared Differences, Block Matching with Normalised Cross-Correlation and Unconstrained Dynamic Programming.
