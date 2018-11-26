## Advanced Lane Finding Project

---

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use colour transforms and gradients to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to centre.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---

[//]: # (Image References)

[calibration]: ./output_images/calibration.jpg "Calibration"
[preprocessing_pipeline]: ./output_images/preprocessing_pipeline.png "Preprocessing Pipeline"
[color_spaces]: ./output_images/color_spaces.png "Color Spaces"
[color_thresholding]: ./output_images/color_thresholding.png "Color Thresholding"
[gradient_thresholding]: ./output_images/gradient_thresholding.png "Gradient Thresholding"
[thresholding]: ./output_images/thresholding.png "Thresholding"
[roi_vector]: ./output_images/roi_vector.png "ROI Vector"
[roi]: ./output_images/roi.png "ROI"
[warped]: ./output_images/warped.png "Warped"

### Camera Calibration

We need to find the camera's intrinsic parameters, along with distortion coefficients for the camera. This is necessary because the lenses curves add distortions to the real image. Therefore lines that are straight in the real world may not be anymore on our photos.

To compute these matrices, we use multiple pictures of a chessboard on a flat surface taken by the same camera. OpenCV's `cv2.findChessboardCorners()` function identifies points where the black and white squares intersect. The mappings between these intersections in our 2D image and the real-world coordinates of the same points in 3D space enable us to obtain the distortion coefficients.

The code for this step is contained in [calibration.py](calibration.py).

`cv2.undistort()` is used to undistort images using the distortion coefficients as shown in the image below:

![calibration]

### Image Preprocessing Pipeline

The goal of the preprocessing pipeline is to take a raw camera image and convert it into a birds-eye view
mask of road lanes. This image is then used for lane detection and tracking. The code for this step can be found in [preprocessing.py](preprocessing.py). [Image Preprocessing Notebook](Image&#32;Preprocessing.ipynb) was used
as a playground to test different parameters.

The following image shows each step of the pipeline:

![preprocessing_pipeline]

#### 1. Undistort the image

#### 2. Thresholding

We started by exploring different colour spaces. RGB, HSV and HLS gave us the best results for both yellow and white lane lines.

![color_spaces]

##### Color Thresholding

We conducted varies tests for colour thresholding in RGB, HSV and HLS colour spaces. Extracting the yellow and white lines with two different thresholds gave us the best results. The following image shows colour thresholding in the various colour spaces:

![color_thresholding]

HLS gave us the most robust output over multiple images and is our method of choice.

[`color_threshold_image3()`](preprocessing.py#L12) was used to compute colour masks. [`or_masks()`](preprocessing.py#L23) was used to combine yellow and white lane masks.

##### Gradient Thresholding

We conducted various tests for gradient thresholding using gradient in x, gradient in y, absolute gradient magnitude and gradient direction. Given that most lane lines are vertical gradient in X and gradient direction thresholding gave us the best results. Gradient in Y and gradient magnitude add unnecessary noise to our thresholding by picking up horizontal artifacts like shadows.

Saturation channel from the HLS colour space was used for gradient calculation. Gradient in X and gradient direction masks were computed and then combined using element wise *and*. The following image shows the obtained gradient mask:

![gradient_thresholding]

[`gradient_threshold()`](preprocessing.py#L45) was used to compute the gradient masks.

##### Combined Mask

The colour and gradient masks were combined together using element-wise *or* to give us the best lane lines.

![thresholding]

#### 3. Image ROI and perspective transform (birds-eye view)

We now need to define a trapezoidal region in the 2D image that will go through a perspective transform to convert into a bird's eye view.

![roi_vector]

This trapezoidal region is first used to create a region of interest. Pixels outside this region are ignored.
[`roi()`](preprocessing.py#L64) function is used to compute this mask.

![roi]

The source and destination polygons are then used to create a perspective transform.
This transformation matrix is used to get the birds-eye view of the road lanes.
[`get_prespective_transform()`](preprocessing.py#L89) returns the transformation and inverse transformation matrices. [`warp_image()`](preprocessing.py#L98) takes and image and transformation matrix and returns the transformed image.

![warped]


![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
