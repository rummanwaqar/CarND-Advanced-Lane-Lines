import numpy as np
import cv2

# Settings
HLS_THRESHOLD = [[(0,255), (200,255), (0,255)], [(0,50),(50,255),(100,255)]]
GRADIENT_KERNEL_SIZE = 15
GRADIENT_THRESHOLD = [(30,100), (0.6, 1.30)]
ROI_SRC = np.array([[560,460], [190,690], [1150,690], [750,460]], dtype='float32')
ROI_DST = np.array([[200, 0], [200,720], [1080,720], [1080, 0]], dtype='float32')
    

def color_threshold_image3(img, thresholds):
    '''
    color thresholds a 3 channel image
    '''
    output = np.zeros([img.shape[0], img.shape[1]])   

    output[(((img[:,:,0] >= thresholds[0][0]) & (img[:,:,0] <= thresholds[0][1])) & 
            ((img[:,:,1] >= thresholds[1][0]) & (img[:,:,1] <= thresholds[1][1])) & 
            ((img[:,:,2] >= thresholds[2][0]) & (img[:,:,2] <= thresholds[2][1])))] = 1
    return output

def or_masks(mask1, mask2):
    '''
    OR operation for two masks
    '''
    mask_out = np.zeros_like(mask1)
    mask_out[((mask1==1) | (mask2==1))] = 1
    return mask_out

def and_masks(mask1, mask2):
    '''
    AND operation for two masks
    '''
    mask_out = np.zeros_like(mask1)
    mask_out[((mask1==1) & (mask2==1))] = 1
    return mask_out

def scale_8(img):
    '''
    scaled an image to 8-bit (0-255)
    '''
    return np.int8(255*img/np.max(img))

def gradient_threshold(img, kernel_size=3, thresholds=[[0,255], [0,np.pi/2]]):
    '''
    thresholds an image using gradient in x and gradient dir 
    '''
    g_x = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size))
    g_y = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size))
    
    # gradient x mask
    g_x_scaled = scale_8(g_x)
    mask_x = np.zeros_like(img)
    mask_x[((g_x_scaled >=thresholds[0][0]) & (g_x_scaled <= thresholds[0][1]))] = 1
    
    # direction mask
    g_dir = np.arctan2(g_y, g_x)
    mask_dir = np.zeros_like(img)
    mask_dir[((g_dir >= thresholds[1][0]) & (g_dir <= thresholds[1][1]))] = 1
    
    return and_masks(mask_x, mask_dir)

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
    masked_image = np.float32(cv2.bitwise_and(img, mask) != 0)
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