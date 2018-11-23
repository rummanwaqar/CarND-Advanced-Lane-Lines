import preprocessing
from calibration import Calibration
from lane_detection import *
from utils import *

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
        
    def process(self, img):
        warped = self.preprocess_frame(img)
        leftx, lefty, rightx, righty, leftx_base, rightx_base = find_lane_pixels(warped, nwindows=9, margin=100, minpix=50)
        midpoint, left_fit, right_fit, left_curvature, right_curvature = self.compute_lane(leftx, lefty, rightx, righty, leftx_base, rightx_base)
        result = self.draw_image(img, left_fit, right_fit, self.M_inv, left_curvature, right_curvature, midpoint)
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
        # meters per pixels
        xm = 3.7/(rightx_base-leftx_base)
        ym = 30/720

        midpoint = get_offset(leftx_base,rightx_base,self.width)*xm
        left_fit = fit_polynomial(leftx, lefty)
        right_fit = fit_polynomial(rightx, righty)
        left_curvature = measure_curvature(719*ym, cvt_line_to_meters(left_fit, xm, ym))
        right_curvature = measure_curvature(719*ym, cvt_line_to_meters(right_fit, xm, ym))
        
        return midpoint, left_fit, right_fit, left_curvature, right_curvature
    
    def draw_image(self, img, left_fit, right_fit, M_inv, left_curvature, right_curvature, midpoint):
        lane_image = draw_lanes(img, left_fit, right_fit, M_inv)
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, lane_image, 0.3, 0)
        # display text
        cv2.putText(result,
                    'Curvature: ({:.2f},{:.2f})m   Offset:{:.2f}m'.format(left_curvature, right_curvature, midpoint),
                    (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),3,cv2.LINE_AA)
        return result