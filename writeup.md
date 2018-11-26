## Advanced Lane Finding Project

---

The lane finding pipeline is set up as follows:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Image preprocessing
  * Apply a distortion correction to raw images.
  * Use colour transforms and gradients to create a thresholded binary image.
  * Apply a perspective transform to rectify binary image ("birds-eye view").
* Lane detection
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
[lane_curves]: ./output_images/lane_curves.png "Lane Curves"
[smoothing]: ./output_images/smoothing.png "Smoothing"
[final]: ./output_images/final.png "Final"
[gif]: ./output_videos/output.gif "Output GIF"

### Camera Calibration

We need to find the camera's intrinsic parameters, along with distortion coefficients for the camera. This is necessary because the lenses curves add distortions to the real image. Therefore lines that are straight in the real world may not be anymore on our photos.

To compute these matrices, we use multiple pictures of a chessboard on a flat surface taken by the same camera. OpenCV's `cv2.findChessboardCorners()` function identifies points where the black and white squares intersect. The mappings between these intersections in our 2D image and the real-world coordinates of the same points in 3D space enable us to obtain the distortion coefficients.

The code for this step is contained in [calibration.py](calibration.py).

`cv2.undistort()` is used to undistort images using the distortion coefficients as shown in the image below:

![calibration]

### Image Preprocessing Pipeline

The goal of the preprocessing pipeline is to take a raw camera image and convert it into a birds-eye view
binary image of road lanes. This image is then used for lane detection and tracking. The code for this step can be found in [preprocessing.py](preprocessing.py). [Image Preprocessing Notebook](Image&#32;Preprocessing.ipynb) was used
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

### Lane Detection

The birds-eye view binary image of road lanes is used for lane detection. A sliding window algorithm is used to identify lane pixels.
The code for this section can be found in the [Project Notebook](Advance&#32;Lane&#32;Finding.ipynb).

Initially, we compute a column-wise histogram on the bottom half of the binary image to detect the x positions where the pixel intensities are highest. With these starting positions, we run a sliding window search to identify the pixel coordinates belonging to each lane. *(`find_lane_pixels()` in Cell[6])*

We then compute a second degree polynomial using numpy's `polyfit()` function to fit the lane pixels. *(`fit_polynomial()` in Cell[6])*

Once we have the lane curves for the first frame, we can further improve the accuracy of our algorithm by performing a localized around this curve for the next frame. If we do not find enough lane pixels we revert back to the original sliding window approach. *(`search_around_poly()` in Cell[6])*

![lane_curves]

We smooth out the lane lines by averaging them over multiple frames. The Euclidean distance of each line is compared with the Euclidean distance of the average to remove outliers and bad estimates.
*(`class <LaneHistory>` in Cell[9])*

*Red line shows current lane estimate. Purple line shows average lane estimate.*
![smoothing]

##### Lane Curvature and Offset

We need to convert all our units from pixels to meters by defining appropriate pixel height to lane length and pixel width to lane width ratios

```
xm = lane_width_in_meters / lane_width_in_pixels     # 3.7/800
ym = lane_length_in_meters/ image_height_in_pixels   # 32./720
```

We compute our offset from the centre of the road by first calculating the starting x-positions
of both lanes *(`eval_point_at_line()` in Cell[9])* and then calculating their offset from the centre of the image. *(`offset()` in Cell[9])*.


We also compute the lane curvature for each lane by calculating the radius of the smallest circle that could be a tangent to our lane lines. The line coefficients are first converted to meters *(`cvt_line_to_meters()` in Cell[9])* and then we convert the curvature of the lane *(`measure_curvature()` in Cell[9])*.

##### Drawing the lanes on the image

Finally we draw the inside region of the lanes using `cv2.polyfill()` function and then unwarp the image using the inverse transformation matrix we previously calculated. We mix this with the original road image and then overlay text for curvature and offset on the image.

![Final]

*Please note that all drawing functions can be found in [utils.py](utils.py)*

---

### Video

The following gif shows the output of lane detection algorithm on a test video:

![gif]

The complete output video can be found [here](output_videos/output.mp4).

---

### Discussion

#### Problems encountered
* The pipeline is the colour dependent. Different lighting conditions (e.g. night time and shadows) or road conditions (e.g. snow or water) renders the algorithm ineffective.
* Finding the right set of colourspace + colour and gradient threshold parameters was very time consuming.
* The coordinates of the perspective transform is hard-coded. The numbers had to be tweaked to give us the proper transform. 

#### Possible improvements to the pipeline
* Investigating other colour spaces more like LAB.
* Dynamic thresholding colours
* Using the vanishing point along with lane lines can be used to automatically calculate the perspective transform.
* Implementing better sanity checks like comparing gradients of lane lines at various points.
* Averaging the curvature values to smooth them out.
* A deep neural network can use used for lane detection. This method removes the need to manually find features and tune them.
