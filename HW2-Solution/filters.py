import numpy as np
from scipy import signal

def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi,Wi))
    
    #flipping kernel with respect to the axis along which we're performing the convolution
    kernel = np.flip(np.flip(kernel,0),1)
    
    image_padded = np.zeros(shape=(Hi + Hk, Wi + Wk))    
    image_padded[Hk//2:-Hk//2, Wk//2:-Wk//2] = image

    for row in range(Hi):
        for col in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                    out[row, col] += image_padded[row + i, col + j]*kernel[i, j]

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """
    
    H, W = image.shape
    
    out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    out[pad_height:pad_height + H, pad_width:pad_width + W] = image
    
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    
    #Flip the kernel  
    kernel = np.flipud(np.fliplr(kernel))
    out = np.zeros_like(image) # convolution output
    
    image_padded = zero_pad(image,Hk//2,Wk//2)
    
    
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            out[y,x]=(kernel*image_padded[y:y + kernel.shape[0], x:x + kernel.shape[1]]).sum()
            
    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    
    out = None
    g = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f,g)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    m = np.mean(g)
    
    for i in range(len(g)):
        g[i] -= m
    
    out = cross_correlation(f,g)

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    out = None
    
    Hf,Wf = f.shape
    Hg,Wg = g.shape
    out = np.zeros((Hf,Wf))
    
    fpad = zero_pad(f,Hg//2,Wg//2)
    gs = np.std(g)
    gm = np.mean(g)
    
    gp = ((g - gm)/gs)
    
    for i in range(Hf):
        for j in range(Wf):
            fn = fpad[i:i+Hg,j:j+Wg]
            fs = np.std(fn)
            fm = np.mean(fn)
            out[i,j] = np.sum(gp * (fn - fm)/fs)
                    
    return out
