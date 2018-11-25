import warnings
import numpy as np

from calibration import Calibration
from utils import *
import preprocessing

class LaneDetection:
    def __init__(self, src_roi, dst_roi, settings):
        self.src_roi = src_roi
        self.dst_roi = dst_roi
        self.settings = settings

        self.M, self.M_inv = preprocessing.get_prespective_transform(src_roi, dst_roi)
        self.cal = Calibration(nx=9, ny=6)
        self.cal.compute_cal('camera_cal/calibration*.jpg')
        
        self.width = None
        self.height = None
        
        self.left = Line()
        self.right = Line()
        
        self.xm = None
        self.ym = None
        
    def process(self, img):
        warped = self.preprocess_frame(img)
        leftx, lefty, rightx, righty, leftx_base, rightx_base = LaneDetection.find_lane_pixels(warped, nwindows=9, margin=100, minpix=50)
        self.compute_lane(leftx, lefty, rightx, righty, leftx_base, rightx_base)
        result = self.draw_image(img)
        return result
        
    def preprocess_frame(self, img):
        if self.width is None or self.height is None:
            self.width = img.shape[1]
            self.height = img.shape[0]
            
        # undistort
        dst = self.cal.undistort(img)
        # create binary mask
        _,_,mask = preprocessing.gradient_mask(img, self.settings, 'and')
        # roi
        roi = preprocessing.roi(mask, self.src_roi)
        # warp image
        warped = preprocessing.warp_image(roi, self.M)
        return warped
    
    def compute_lane(self, leftx, lefty, rightx, righty, leftx_base, rightx_base):
        # meters per pixels (LPF)
        xm = 3.7/(rightx_base-leftx_base)
        ym = 30/720
        if self.xm is None or self.ym is None:
            self.xm = xm
            self.ym = ym
        else:
            self.xm = self.xm * 0.7 + xm * 0.3
            self.ym = self.ym * 0.7 + ym * 0.3
            
        self.left.add_line_points(leftx, lefty, leftx_base, self.xm, self.ym)
        self.right.add_line_points(rightx, righty, rightx_base, self.xm, self.ym)
    
    def draw_image(self, img):
        lane_image = draw_lanes(img, self.left.get_fit(), self.right.get_fit(), self.M_inv)
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, lane_image, 0.3, 0)
        # display text
        cv2.putText(result,
                    'Curvature: ({:.2f},{:.2f})m   Offset:{:.2f}m'.format(
                        self.left.get_curvature(), self.right.get_curvature(), self.get_offset()),
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3,cv2.LINE_AA)
        return result
    
    def get_offset(self):
        '''
        get offset of lanes from image midpoint
        right offset +ve
        left offset -ve
        '''
        return (((self.left.get_base()+self.right.get_base())/2)-(self.width/2))*self.xm

    @staticmethod
    def find_lane_pixels(img, nwindows, margin, minpix):
        '''
        Searches mask image for lanes
        '''

        # Create histogram of image binary activations
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(img.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            # Find the four below boundaries of the window
            win_xleft_low = leftx_current-margin
            win_xleft_high = leftx_current+margin
            win_xright_low = rightx_current-margin
            win_xright_high = rightx_current+margin

            #Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window
            # (`right` or `leftx_current`) on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty, leftx_base, rightx_base

    
class Line():
    '''
    characteristics of each line detection
    '''
    RADIUS_Y = 690
    def __init__(self, n=3):
        # number of iterations
        self.n = n
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent n fits
        self.current_fits = []  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
        
    def add_line_points(self, x, y, base, xm, ym):
        self.allx = x
        self.ally = y
        self.line_base_pos = base
        
        fit = Line.fit_polynomial(self.allx, self.ally)
        if fit is not None:
            self.detected = True
            self.current_fits.append(fit)
            if len(self.current_fits) > self.n:
                self.current_fits.pop(0)
            self.best_fit = np.sum(self.current_fits, axis=0) / len(self.current_fits)
            self.radius_of_curvature = Line.measure_curvature(self.RADIUS_Y*ym, 
                                                              Line.cvt_line_to_meters(self.best_fit, xm, ym))
        else:
            self.detected = False
    
    def get_fit(self):
        return self.best_fit
    
    def get_curvature(self):
        return self.radius_of_curvature
    
    def get_base(self):
        return self.line_base_pos
    
    @staticmethod
    def fit_polynomial(x, y):
        '''
        fit line to pixels
        '''
        # Fit a second order polynomial to each using `np.polyfit`
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                return np.polyfit(y, x, 2)
            except np.RankWarning:
                return None

    @staticmethod
    def measure_curvature(y, line_eq):
        '''
        Calculates the curvature of polynomial functions.
        '''
        # calculation of R_curve (radius of curvature)
        return ((1 + (2*y*line_eq[0]+line_eq[1])**2)**(3/2))/np.absolute(2*line_eq[0])

    @staticmethod
    def cvt_line_to_meters(line_eq, xm, ym):
        '''
        cvt line from pixels to meters
        '''
        new_line = np.copy(line_eq)
        new_line[0] *= xm/(ym**2)
        new_line[1] *= xm/ym
        return new_line