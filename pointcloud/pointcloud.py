import os

import cv2
import depthai as dai
import numpy as np
import open3d as o3d
from loguru import logger

import main.config as config
from main.camera import Camera
from pointcloud.host_sync import HostSync


class Pointcloud(Camera):
    """
    Class for the point cloud camera.
    """

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int):
        super().__init__(device_info, friendly_id)

        self.pipeline = self.create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.device.setIrLaserDotProjectorBrightness(config.laser_dot)
        self.device.setIrFloodLightBrightness(config.flood_light)

        self.image_queue = self.device.getOutputQueue(
            name="image", maxSize=10, blocking=False
        )
        self.depth_queue = self.device.getOutputQueue(
            name="depth", maxSize=10, blocking=False
        )
        self.host_sync = HostSync(["image", "depth"])

        self.point_cloud = o3d.geometry.PointCloud()

        self.load_default_calibration()
        self.load_calibration()

        logger.info(f"{self.friendly_id}: Pointcloud initialized")

    def create_pipeline(self):
        """
        Creates the pipeline for the cameras.
        """

        pipeline = dai.Pipeline()

        # Depth cam -> 'depth'
        mono_left = pipeline.createMonoCamera()
        mono_right = pipeline.createMonoCamera()
        mono_left.setResolution(config.depth_resolution)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(config.depth_resolution)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_stereo = pipeline.createStereoDepth()
        cam_stereo.setDefaultProfilePreset(config.stereo_preset)

        cam_stereo.initialConfig.setMedianFilter(config.median)
        cam_stereo.initialConfig.setConfidenceThreshold(config.confidence_threshold)
        cam_stereo.initialConfig.setDisparityShift(config.disparity_shift)
        cam_stereo.setLeftRightCheck(config.lrcheck)
        cam_stereo.setExtendedDisparity(config.extended)
        cam_stereo.setSubpixel(config.subpixel)
        cam_stereo.setSubpixelFractionalBits(config.subpixel_bits)

        mono_left.out.link(cam_stereo.left)
        mono_right.out.link(cam_stereo.right)

        init_config = cam_stereo.initialConfig.get()
        init_config.postProcessing.speckleFilter.enable = config.speckle
        init_config.postProcessing.speckleFilter.speckleRange = config.speckle_range
        init_config.postProcessing.temporalFilter.enable = config.temporal
        init_config.postProcessing.temporalFilter.persistencyMode = (
            config.persistency_mode
        )
        init_config.postProcessing.spatialFilter.enable = config.spatial
        init_config.postProcessing.spatialFilter.holeFillingRadius = (
            config.spatial_hole_filling_radius
        )
        init_config.postProcessing.spatialFilter.numIterations = (
            config.spatial_iterations
        )
        init_config.postProcessing.thresholdFilter.minRange = config.min_range
        init_config.postProcessing.thresholdFilter.maxRange = config.max_range
        init_config.postProcessing.decimationFilter.decimationFactor = config.decimation
        cam_stereo.initialConfig.set(init_config)

        xout_depth = pipeline.createXLinkOut()
        xout_depth.setStreamName("depth")
        cam_stereo.depth.link(xout_depth.input)

        # RGB cam or mono right -> 'image'
        xout_image = pipeline.createXLinkOut()
        xout_image.setStreamName("image")

        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setResolution(config.rgb_resolution)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.setIspScale(1, config.isp_scale)

        if config.manual_exposure:
            cam_rgb.initialControl.setManualExposure(config.exposure, config.iso)
        if config.manual_focus:
            cam_rgb.initialControl.setManualFocus(config.focus)
        if config.manual_whitebalance:
            cam_rgb.initialControl.setManualWhiteBalance(config.whitebalance)

        cam_stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        cam_rgb.isp.link(xout_image.input)
        self.image_size = cam_rgb.getIspSize()

        return pipeline

    def update(self):
        """
        Updates the frames.
        """

        for queue in [self.depth_queue, self.image_queue]:
            new_msgs = queue.tryGetAll()
            if new_msgs is not None:
                for new_msg in new_msgs:
                    self.host_sync.add(queue.getName(), new_msg)

        msg_sync = self.host_sync.get()
        if msg_sync is None:
            return

        self.depth_frame = msg_sync["depth"].getFrame()
        self.image_frame = msg_sync["image"].getCvFrame()

        rgb = cv2.cvtColor(self.image_frame, cv2.COLOR_BGR2RGB)
        self.rgbd_to_point_cloud(self.depth_frame, rgb)

    def save_point_cloud_alignment(self):
        """
        Saves the point cloud alignment to file.
        """

        np.save(
            f"{config.calibration_data_dir}/point_cloud_alignment_{self.mxid}.npy",
            self.point_cloud_alignment,
        )

        np.save(
            os.path.join(
                os.path.expanduser("~/ros2_ws"),
                "calibration_data",
                f"point_cloud_alignment_{self.mxid}.npy",
            ),
            self.point_cloud_alignment,
        )

        logger.info(f"{self.friendly_id}: Point cloud alignment saved")

    def rgbd_to_point_cloud(
        self,
        depth_frame,
        image_frame,
        # downsample=config.downsample,
        remove_noise=config.remove_noise,
    ):
        """
        Converts the RGBD frames to a point cloud.
        """

        rgb_o3d = o3d.geometry.Image(image_frame)
        df = np.copy(depth_frame).astype(np.float32)
        # df -= 20
        depth_o3d = o3d.geometry.Image(df)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=(len(image_frame.shape) != 3)
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, self.pinhole_camera_intrinsic, self.world_to_cam
        )

        # if downsample:
        #     point_cloud = point_cloud.voxel_down_sample(
        #         voxel_size=config.downsample_size
        #     )

        if remove_noise:
            point_cloud = point_cloud.remove_statistical_outlier(
                nb_neighbors=30, std_ratio=0.1
            )[0]

        self.point_cloud.points = point_cloud.points
        self.point_cloud.colors = point_cloud.colors

        # correct upside down z axis
        T = np.eye(4)
        T[2, 2] = -1
        self.point_cloud.transform(T)

        # apply point cloud alignment transform
        self.point_cloud.transform(self.point_cloud_alignment)

        return self.point_cloud
