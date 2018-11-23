import os
import glob
import pickle
import numpy as np
import cv2

class Calibration:
    '''
    compute camera calibration and undistorts images
    '''
    def __init__(self, nx=9, ny=6, cal_fname='calibration.pkl'):
        self.nx = nx
        self.ny = ny
        self.cal_fname = cal_fname
        self.camera_matrix = None
        self.distortion_coff = None
        
    def compute_cal(self, images):
        '''
        checks for calibration file. If DNE then it computes calibration using glob path to calibration images
        '''
        if os.path.isfile(self.cal_fname):
            # retrive calibration and undistort
            self.camera_matrix, self.distortion_coff = pickle.load(open(self.cal_fname, 'rb'))
            print('Loaded camera config from file: {}'.format(self.cal_fname))
        else:
            print('Calibrating camera ...')
            # get all camera calibration images
            fnames = glob.glob(images)

            objpoints = [] # 3d points in real world space
            imgpoints = [] # 2d points in image plane

            # initialize object points (these stay const)
            objp = np.zeros((self.nx*self.ny,3), np.float32)
            objp[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1, 2)

            # enumerate over each image and add image points to array
            for idx, fname in enumerate(fnames):
                img = cv2.imread(fname)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # find corners
                ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)

                if ret:
                    objpoints.append(objp)
                    imgpoints.append(corners)
                else:
                    print('No corners found for {}'.format(fname))
                    
            # calibrate camera
            ret, self.camera_matrix, self.distortion_coff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
            # save calibration
            with open(self.cal_fname, 'wb') as cal_file:
                pickle.dump((self.camera_matrix, self.distortion_coff), cal_file)
                print('Calibration complete. Saved to {}'.format(self.cal_fname))
    
    def undistort(self, img):
        '''
        returns undisorted image
        '''
        if (self.camera_matrix is not None) and (self.distortion_coff is not None):
            return cv2.undistort(img, self.camera_matrix, self.distortion_coff, None, self.camera_matrix)
        else:
            return None
    
    def __str__(self):
        return 'Camera matrix: \n{}\nDist coefficients: \n{}\n'.format(self.camera_matrix, self.distortion_coff)