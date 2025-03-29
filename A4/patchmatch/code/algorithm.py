# CSC320 Fall 2024
# Assignment 4
# (c) Olga (Ge Ya) Xu, Robin Swanson, Kyros Kutulakos
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

# Import basic packages.
from typing import List, Union, Tuple, Dict
import numpy as np

#
# Basic numpy configuration
#

# Set random seed.
np.random.seed(seed=131)
# Ignore division-by-zero warning.
np.seterr(divide="ignore", invalid="ignore")


def propagation_and_random_search(
    source_patches: np.ndarray,
    target_patches: np.ndarray,
    f: np.ndarray,
    alpha: float,
    w: int,
    propagation_enabled: bool,
    random_enabled: bool,
    odd_iteration: bool,
    best_D: Union[np.ndarray, None] = None,
    global_vars: Union[Dict, None] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Basic PatchMatch loop.

    This function implements the basic loop of the PatchMatch algorithm, as
    explained in Section 3.2 of the paper. The function takes an NNF f as
    input, performs propagation and random search, and returns an updated NNF.

    Args:
        source_patches:
            A numpy matrix holding the patches of the color source image,
              as computed by the make_patch_matrix() function in this module.
              For an NxM source image and patches of width P, the matrix has
              dimensions NxMxCx(P^2) where C is the number of color channels
              and P^2 is the total number of pixels in the patch.  For
              your purposes, you may assume that source_patches[i,j,c,:]
              gives you the list of intensities for color channel c of
              all pixels in the patch centered at pixel [i,j]. Note that patches
              that go beyond the image border will contain NaN values for
              all patch pixels that fall outside the source image.
        target_patches:
            The matrix holding the patches of the target image, represented
              exactly like the source_patches argument.
        f:
            The current nearest-neighbour field.
        alpha:
            Algorithm parameter, as explained in Section 3 and Eq.(1).
        w:
            Algorithm parameter, as explained in Section 3 and Eq.(1).
        propagation_enabled:
            If True, propagation should be performed. Use this flag for
              debugging purposes, to see how your
              algorithm performs with (or without) this step.
        random_enabled:
            If True, random search should be performed. Use this flag for
              debugging purposes, to see how your
              algorithm performs with (or without) this step.
        odd_iteration:
            True if and only if this is an odd-numbered iteration.
              As explained in Section 3.2 of the paper, the algorithm
              behaves differently in odd and even iterations and this
              parameter controls this behavior.
        best_D:
            And NxM matrix whose element [i,j] is the similarity score between
              patch [i,j] in the source and its best-matching patch in the
              target. Use this matrix to check if you have found a better
              match to [i,j] in the current PatchMatch iteration.
        global_vars:
            (optional) if you want your function to use any global variables,
              return them in this argument and they will be stored in the
              PatchMatch data structure.

    Returns:
        A tuple containing (1) the updated NNF, (2) the updated similarity
          scores for the best-matching patches in the target, and (3)
          optionally, if you want your function to use any global variables,
          return them in this argument and they will be stored in the
          PatchMatch data structure.
    """
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    #############################################

    return new_f, best_D, global_vars


def reconstruct_source_from_target(target: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Reconstruct a source image using pixels from a target image.

    This function uses a computed NNF f(x,y) to reconstruct the source image
    using pixels from the target image.  To reconstruct the source, the
    function copies to pixel (x,y) of the source the color of
    pixel (x,y)+f(x,y) of the target.

    The goal of this routine is to demonstrate the quality of the
    computed NNF f. Specifically, if patch (x,y)+f(x,y) in the target image
    is indeed very similar to patch (x,y) in the source, then copying the
    color of target pixel (x,y)+f(x,y) to the source pixel (x,y) should not
    change the source image appreciably. If the NNF is not very high
    quality, however, the reconstruction of source image
    will not be very good.

    You should use matrix/vector operations to avoid looping over pixels,
    as this would be very inefficient.

    Args:
        target:
            The target image that was used as input to PatchMatch.
        f:
            A nearest-neighbor field the algorithm computed.
    Returns:
        An openCV image that has the same shape as the source image.
    """
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    #############################################

    return rec_source


def make_patch_matrix(im: np.ndarray, patch_size: int) -> np.ndarray:
    """
    PatchMatch helper function.

    This function is called by the initialized_algorithm() method of the
    PatchMatch class. It takes an NxM image with C color channels and a patch
    size P and returns a matrix of size NxMxCxP^2 that contains, for each
    pixel [i,j] in the image, the pixels in the patch centered at [i,j].

    You should study this function very carefully to understand precisely
    how pixel data are organized, and how patches that extend beyond
    the image border are handled.

    Args:
        im:
            A image of size NxM.
        patch_size:
            The patch size.

    Returns:
        A numpy matrix that holds all patches in the image in vectorized form.
    """
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = (
        im.shape[0] + patch_size - 1,
        im.shape[1] + patch_size - 1,
        im.shape[2],
    )
    padded_im = np.zeros(padded_shape) * np.nan
    padded_im[phalf : (im.shape[0] + phalf), phalf : (im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel.
    # If the original image had NxM pixels, this matrix will have
    # NxMx(patch_size*patch_size) pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size**2
    patch_matrix = np.zeros(patch_matrix_shape) * np.nan
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[
                i : (i + im.shape[0]), j : (j + im.shape[1]), :
            ]

    return patch_matrix


def make_coordinates_matrix(im_shape: Tuple, step: int = 1) -> np.ndarray:
    """
    PatchMatch helper function.

    This function returns a matrix g of size (im_shape[0] x im_shape[1] x 2)
    such that g(y,x) = [y,x].

    Pay attention to this function as it shows how to perform these types
    of operations in a vectorized manner, without resorting to loops.

    Args:
        im_shape:
            A tuple that specifies the size of the input images.
        step:
            (optional) If specified, the function returns a matrix that is
              step times smaller than the full image in each dimension.
    Returns:
        A numpy matrix holding the function g.
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
