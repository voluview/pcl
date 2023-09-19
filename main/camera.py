import depthai as dai
import numpy as np
import open3d as o3d
from loguru import logger

import main.config as config


class Camera:
    """
    Base class for the cameras.
    """

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int):
        self.device_info = device_info
        self.friendly_id = friendly_id
        self.mxid = device_info.getMxId()

        self.show_video = False
        self.show_point_cloud = True

        logger.info(f"{self.friendly_id}: Connected to " + self.device_info.getMxId())

    def load_default_calibration(self):
        """
        Loads the default camera calibration data.
        """

        # Camera intrinsic parameters
        calibration = self.device.readCalibration()
        self.intrinsics = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.RGB if config.COLOR else dai.CameraBoardSocket.RIGHT,
            dai.Size2f(*self.image_size),
        )

        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            *self.image_size,
            self.intrinsics[0][0],
            self.intrinsics[1][1],
            self.intrinsics[0][2],
            self.intrinsics[1][2],
        )

        self.intrinsic_mat = np.array(self.intrinsics)
        self.distortion_coef = np.zeros((1, 5))

        # Camera extrinsic parameters
        self.rot_vec = None
        self.trans_vec = None
        self.world_to_cam = None
        self.cam_to_world = None

        logger.info(
            f"{self.friendly_id}: Loaded default calibration for "
            + self.device_info.getMxId()
        )

    def load_calibration(self):
        """
        Loads the camera calibration data from a file.
        """

        path = f"{config.calibration_data_dir}/camera_{self.mxid}.npz"
        try:
            extrinsics = np.load(path)
            self.cam_to_world = extrinsics["cam_to_world"]
            self.world_to_cam = extrinsics["world_to_cam"]
            logger.info(
                "{}: Loaded calibration data for camera {} from {}".format(
                    self.friendly_id, self.mxid, path
                )
            )
        except FileNotFoundError:
            logger.warning(
                "{}: Could not load calibration data for camera {} from {}!".format(
                    self.friendly_id,
                    self.mxid,
                    path,
                )
            )
            self.cam_to_world = None
            self.world_to_cam = None

        try:
            self.point_cloud_alignment = np.load(
                f"{config.calibration_data_dir}/point_cloud_alignment_{self.mxid}.npy"
            )
            logger.info(
                "{}: Loaded point cloud alignment for camera {} from {}".format(
                    self.friendly_id, self.mxid, path
                )
            )
        except FileNotFoundError:
            logger.warning(
                "{}: Could not load point cloud alignment for cam {} from {}!".format(
                    self.friendly_id,
                    self.mxid,
                    path,
                )
            )
            self.point_cloud_alignment = np.eye(4)

        logger.info(self.pinhole_camera_intrinsic)

    def close(self):
        """
        Closes the camera.
        """

        self.device.close()
        logger.warning(f"{self.friendly_id}: Closed " + self.device_info.getMxId())
