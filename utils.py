import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from preprocessing import warp_image

# output image folder
output_folder = 'output_images/'

def display_images(images, labels, fname='', path=output_folder, figsize=None, cmap=None):
    assert len(images) > 0
    assert len(images) == len(labels)
    
    if figsize is not None:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    
    for idx in range(len(images)):
        plt.subplot((len(images)//3)+1,3,idx+1)
        plt.title(labels[idx])
        if cmap is not None:
            plt.imshow(images[idx], cmap=cmap)
        else:
            plt.imshow(images[idx])
        
    if fname:
        plt.savefig(os.path.join(path, fname), bbox_inches='tight')
    plt.show()
    
def plot_ploy_lines(img, left_fit, right_fit):
    '''
    plot lines on image
    '''
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    plt.imshow(img, cmap='gray')
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='red')
    
def draw_lanes(img, left_fit, right_fit, M_inv):
    '''
    draw lane lines
    '''
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    color_warp = np.zeros_like(img)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = warp_image(color_warp, M_inv)
    return newwarp

def embed_image(img1, img2, size=(240,135), offset=(50,50)):
    '''
    embeds a scaled image into another image
    '''
    output = np.copy(img1)
    # scale second image
    scaled = cv2.resize(img2,(size[0],size[1])) * 255
    scaled = np.dstack((scaled, scaled, scaled))
    # embed image
    output[offset[1]:offset[1]+scaled.shape[0], offset[0]:offset[0]+scaled.shape[1]] = scaled
    return output