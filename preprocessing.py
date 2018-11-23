import numpy as np
import cv2

class GradientSetting:
    '''
    Settings for indiviual gradient setting
    '''
    def __init__(self, kernel=0, threshold=0):
        self.kernel = kernel
        self.threshold = threshold

class MaskSetting:
    '''
    Settings for combined gradient mask
    '''
    blur = None
    grad_x = GradientSetting()
    grad_y = GradientSetting()
    grad_mag = GradientSetting()

def gradient_mask(img, settings, mask_type='and'):
    '''
    Calculate gradient mask
    '''
    assert isinstance(settings, MaskSetting)
    
    r_channel = img[:,:,0]
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    
    # calculate masks
    _,_,_,r_mask = combined_thresh(r_channel, settings)
    _,_,_,s_mask = combined_thresh(s_channel, settings)
    
    # combine masks
    mask = np.zeros_like(r_channel)
    if mask_type == 'and':
        mask[(r_mask == 1)&(s_mask == 1)] = 1
    else: # 'or'
        mask[(r_mask == 1)|(s_mask == 1)] = 1
        
    return r_mask, s_mask, mask
    
def combined_thresh(img, settings):
    '''
    Combine gradient in x, gradient in y, and gradient abs magnitude 
    '''
    assert isinstance(settings, MaskSetting)
    
    # apply blur
    if settings.blur:
        img = cv2.GaussianBlur(img,(settings.blur,settings.blur),0)
        
    # gradients
    grad_x = abs_sobel_thresh(img, 'x', 
                              sobel_kernel=settings.grad_x.kernel, 
                              thresh=settings.grad_x.threshold)
    grad_y = abs_sobel_thresh(img, 'y', 
                              sobel_kernel=settings.grad_y.kernel, 
                              thresh=settings.grad_y.threshold)
    grad_mag = mag_thresh(img, 
                          sobel_kernel=settings.grad_mag.kernel, 
                          thresh=settings.grad_mag.threshold)
    
    # combine gradients (only x,y,mag)
    mask = np.zeros_like(img)
    mask[((grad_mag==1)&(grad_x==1)&(grad_y==1))] = 1
    
    return grad_x, grad_y, grad_mag, mask 
 
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    '''
    takes derivate in one axis and thresholds on it
    '''
    # single channel
    assert len(img.shape) == 2
    # take derivative
    derivative = None
    if orient == 'x':
        derivative = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        derivative = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # take abs value
    abs_derivative = np.absolute(derivative)
    # scale up to 8-bit
    scaled_derivative = np.int8(255*abs_derivative/np.max(abs_derivative))
    # create mask with threshold
    mask = np.zeros_like(scaled_derivative)
    mask[(scaled_derivative >= thresh[0]) & (scaled_derivative <= thresh[1])] = 1
    return mask

def mag_thresh(img, sobel_kernel=3, thresh=(0,255)):
    '''
    takes derivate in x and y and thresholds on combined magnitude
    '''
    # single channel
    assert len(img.shape) == 2
    # take derivative
    derivative_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    derivative_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # take magnitude value
    mag_derivative = np.sqrt(np.square(derivative_x) + np.square(derivative_y))
    # scale up to 8-bit
    scaled_derivative = np.int8(255*mag_derivative/np.max(mag_derivative))
    # create mask with threshold
    mask = np.zeros_like(scaled_derivative)
    mask[(scaled_derivative >= thresh[0]) & (scaled_derivative <= thresh[1])] = 1
    return mask

def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''
    takes derivative in x and y and thresholds on direction of gradient
    '''
    # single channel
    assert len(img.shape) == 2
    # take derivative
    derivative_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    derivative_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # take abs of x and y
    abs_x = np.absolute(derivative_x)
    abs_y = np.absolute(derivative_y)
    # find direction
    direction = np.arctan2(abs_y, abs_x)
    # create mask with threshold
    mask = np.zeros_like(direction)
    mask[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return mask

def roi(img, vertices):
    '''
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    '''
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def get_prespective_transform(src, dst):
    '''
    Calculate prespective transform matrices
    '''
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv

def warp_image(img, matrix):
    '''
    Warp image by applying prespective transform
    '''
    img_size = (img.shape[1], img.shape[0])
    # Warp the image using OpenCV warpPerspective()
    return cv2.warpPerspective(img, matrix, img_size)