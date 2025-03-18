# CSC320 Spring 2024
# Assignment 3
# (c) Kyros Kutulakos
#
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
from inpainting import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
from inpainting import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    patch_center = (psiHatP.row(), psiHatP.col())
    patch_radius = psiHatP.radius()
    
    confidence_patch, _ = copyutils.getWindow(confidenceImage, patch_center, patch_radius)
    filled_patch, _ = copyutils.getWindow(filledImage, patch_center, patch_radius)
    filled_patch = filled_patch == 255
    
    #########################################
    # Replace this dummy value with your own code
    C = np.sum(confidence_patch[filled_patch]) / (2 * patch_radius + 1)**2
    #########################################
    
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).


def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    # Prepare patch
    patch_center = (psiHatP.row(), psiHatP.col())
    patch_radius = psiHatP.radius()
    
    color_patch, _ = copyutils.getWindow(inpaintedImage, patch_center, patch_radius)
    gray_patch = cv.cvtColor(color_patch, cv.COLOR_BGR2GRAY)
    
    filled_patch, _ = copyutils.getWindow(filledImage, patch_center, patch_radius)
    
    kernel = np.ones((3,3), np.uint8)
    valid_mask = cv.erode(filled_patch, kernel, iterations=1) == 255
    
    # create gradient patch that contains the gradient for each pixel in the patch
    grad_x = cv.Sobel(gray_patch, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(gray_patch, cv.CV_64F, 0, 1, ksize=3)
    
    # filter out invalid pixels in the gradient patch (pixels that was not filled)
    mask = (filled_patch == 255) & valid_mask
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag[~mask] = -1
    
    # find the max magnitude of the gradient for each pixel in the patch and its index

    max_grad_idx = np.argmax(grad_mag)
    max_grad_coords = np.unravel_index(max_grad_idx, grad_mag.shape)
    if grad_mag[max_grad_coords] == -1:
        return 0, 0
    
    #########################################
    
    # Replace these dummy values with your own code
    Dy = grad_y[max_grad_coords]
    Dx = grad_x[max_grad_coords]
    #########################################
    
    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#
def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    
    ff_size = np.count_nonzero(fillFront == 255)
    if ff_size <= 1:
        return None, None
    
    P = (psiHatP.row(), psiHatP.col())
    
    # Invert FillFront
    inverted_fillFront = 255 - fillFront
    
    # Get matrix of distance from the fill front
    distance = cv.distanceTransform(inverted_fillFront, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    
    # find the gradient of the distance matrix, which points at the direction that distance from the fill front increases the fastest
    grad_x = cv.Sobel(distance, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(distance, cv.CV_64F, 0, 1, ksize=3)

    
    # find the gradient at the patch center
    Ny = grad_y[P[0], P[1]]
    Nx = grad_x[P[0], P[1]]
    
    if Ny == 0 and Nx == 0:
        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # Up, Down, Left, Right
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonals
        ]
        
        for dy, dx in neighbors:
            newY, newX = P[0] + dy, P[1] + dx
            if 0 <= newY < fillFront.shape[0] and 0 <= newX < fillFront.shape[1]:
                if fillFront[newY, newX] == 255:  # Found a valid fill front neighbor
                    Ny, Nx =  dy, dx  # Approximate normal
                    break
            
    normalizer = np.sqrt(Ny**2 + Nx**2) + 1e-6
    
    #########################################
    
    # Replace these dummy values with your own code
    Ny = - Ny / normalizer
    Nx = - Nx / normalizer
    #########################################

    return Ny, Nx
