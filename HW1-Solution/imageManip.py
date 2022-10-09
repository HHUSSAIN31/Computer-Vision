import math
import cv2
import skimage
import numpy as np
from PIL import Image
from skimage import color, io
import matplotlib.pyplot as plt # for display


def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    # Path for image1/image2 for reference only
    #skimage.io.imread('/home/haider/Documents/Computer_Vision/HW-1/notebook/HW_1/image1.jpg')
#skimage.io.imread('/home/haider/Documents/Computer_Vision/HW-1/notebook/HW_1/image2.jpg')

    #for any image
    out = skimage.io.imread(image_path)
   

    # Let's convert the image to be between the correct range.
    out = out.astype(np.float64) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None
    
    xN = 0
    xp = image
    xN = 0.5 * (xP*xP)
    xN = xN.astype(np.float64) / 255
    out = xN
 
    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: Look at `skimage.color` library to see if there is a function
    there you can use.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None
    
    grey_image = skimage.color.rgb2gray(image)
    
    out = grey_image
    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    image = Image.open(image)
    image_data = image.load()
    h,w = image.size

    if channel == 'R':
        for loop1 in range(h):
            for loop2 in range (w):
                r,g,b = image_data[loop1,loop2]
                image_data[loop1,loop2] = 0,g,b

    if channel == 'G':
        for loop1 in range(h):
            for loop2 in range (w):
                r,g,b = image_data[loop1,loop2]
                image_data[loop1,loop2] = r,0,b

    if channel == 'B':
        for loop1 in range(h):
            for loop2 in range (w):
                r,g,b = image_data[loop1,loop2]
                image_data[loop1,loop2] = r,g,0
    out = image
    
    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    hsv = color.rgb2hsv(image)
    
    
    out = None
    hsv = color.rgb2hsv(image)
    h,s,v = cv2.split(hsv)
    if channel == 'H':
        out = h

    if channel == 'S':
        out = s

    if channel ='V':
        out = v

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    
    out = None
    m1 = rgb_exclusion(image1, channel1)
    m2 = rgb_exclusion(image2, channel2)
    
    m1 = m1[50:300, 150:300,:]
    m2 = m2[50:300, 150:300,:]
  
    out = np.concatenate((m1,m2), axis = 1)
    
    return out


def mix_quadrants(image):
    """THIS IS AN EXTRA CREDIT FUNCTION.

    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
