import cv2
import depthai as dai

# Calibration configuration
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
checkerboard_size = [12, 8]  # number of squares on the checkerboard
square_length = 0.0215  # size of a square in meters
marker_length = 0.017  # size of a marker in meters

# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
# checkerboard_size = [8, 6]  # number of squares on the checkerboard
# square_length = 0.0285  # size of a square in meters
# marker_length = 0.0225  # size of a marker in meters

calibration_data_dir = "calibration_data"  # directory relative to app.py

# Camera sensor settings
# Options: THE_720_P, THE_800_P, THE_400_P, THE_480_P, THE_1080_P, THE_4_K
rgb_resolution = dai.ColorCameraProperties.SensorResolution.THE_1080_P
depth_resolution = dai.MonoCameraProperties.SensorResolution.THE_800_P
COLOR = True  # Use color camera of mono camera
isp_scale = 1  # downscale factor for the image signal processor

# Manual settings
manual_exposure = False
exposure = 5000  # 1-33000
iso = 400  # 100-1600

manual_focus = False
focus = 130  # 0-255

manual_whitebalance = False
whitebalance = 5000  # 1000-12000 (blue -> yellow)

# Camera IR settings
laser_dot = 800  # Laser dot projector brightness, in mA, 0..1200
flood_light = 300  # Flood light brightness, in mA, 0..1500

# Range configuration
min_range = 350  # mm (from 350)
max_range = 650  # mm

# Depth configuration
stereo_preset = (
    dai.node.StereoDepth.PresetMode.HIGH_ACCURACY
)  # Options: HIGH_DENSITY, HIGH_ACCURACY
lrcheck = True  # Better handling for occlusions
extended = (
    True  # Closer-in minimum depth (35cm vs 70cm in normal), disparity range is doubled
)
subpixel = True  # Better accuracy for longer distance, fractional disparity 32-levels
subpixel_bits = 5  # Fractional disparity bits (3, 4, 5)
disparity_shift = 10  # Disparity shift
confidence_threshold = 180  # 0-255, 255 = low confidence, 0 = high confidence
median = (
    dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
)  # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7

# Post-processing configuration
speckle = True  # Filter speckle noise
speckle_range = 50  # Max range of speckle

temporal = True  # Filter temporal noise
persistency_mode = (
    dai.RawStereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_2_IN_LAST_3
)  # Options: PERSISTENCY_OFF, VALID_8_OUT_OF_8, VALID_2_IN_LAST_3, VALID_2_IN_LAST_4, VALID_2_OUT_OF_8, VALID_1_IN_LAST_2, VALID_1_IN_LAST_5, VALID_1_IN_LAST_8, PERSISTENCY_INDEFINITELY

spatial = False  # Filter spatial noise
spatial_iterations = 1  # Number of iterations for spatial noise filter
spatial_hole_filling_radius = 30  # Fill gaps in disparity map

decimation = 1  # Downscale factor for disparity map

# Pointcloud generation configuration
downsample = False  # Voxel downsampling
downsample_size = 0.0005  # Voxel downsampling size
remove_noise = False  # Remove noise from point cloud

# Pointcloud color mask
remove_color_live = False  # Remove color from point cloud
color_to_remove = "red"  # Color to remove from point cloud
color_treshold = 0.9  # Treshold for color removal

# Pointcloud fine alignment configuration
voxel_radius = [0.01]  # Voxel downsampling size
max_iter = [5]  # Maximum number of iterations

# voxel_radius = [0.04, 0.02, 0.01]
# max_iter = [50, 30, 14]

# Pointcloud show configuration
show = True  # Show point cloud in a GUI
point_size = 2  # size of the points in the point cloud
save = False  # Save point cloud to file

# Postprocessing configuration
remove_color_post = True  # Remove color from point cloud
remove_statistical_outlier = True  # Remove statistical outliers from point cloud
