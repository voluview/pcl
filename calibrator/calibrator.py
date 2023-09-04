import os
import time

import cv2
import depthai as dai
import numpy as np

# from cv2 import aruco
from loguru import logger

from main import config
from main.camera import Camera


class Calibrator(Camera):
    """
    Class for the camera calibration.
    """

    def __init__(self, device_info: dai.DeviceInfo, friendly_id: int):
        super().__init__(device_info, friendly_id)

        self.pipeline = self.create_pipeline()
        self.device = dai.Device(self.pipeline, self.device_info)

        self.rgb_queue = self.device.getOutputQueue(
            name="rgb", maxSize=1, blocking=False
        )
        self.still_queue = self.device.getOutputQueue(
            name="still", maxSize=1, blocking=False
        )
        self.control_queue = self.device.getInputQueue(name="control")

        self.show_windows()
        self.load_default_calibration()

        logger.info(f"{self.friendly_id}: Calibrator initialized")

    def create_pipeline(self):
        """
        Creates the pipeline for the cameras.
        """

        pipeline = dai.Pipeline()

        # RGB cam -> 'rgb'
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam_rgb.setPreviewSize(640, 360)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setPreviewKeepAspectRatio(False)
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        # Still encoder -> 'still'
        still_encoder = pipeline.create(dai.node.VideoEncoder)
        still_encoder.setDefaultProfilePreset(
            1, dai.VideoEncoderProperties.Profile.MJPEG
        )
        cam_rgb.still.link(still_encoder.input)
        xout_still = pipeline.createXLinkOut()
        xout_still.setStreamName("still")
        still_encoder.bitstream.link(xout_still.input)

        # Camera control -> 'control'
        control = pipeline.create(dai.node.XLinkIn)
        control.setStreamName("control")
        control.out.link(cam_rgb.inputControl)

        self.image_size = cam_rgb.getIspSize()

        return pipeline

    def show_windows(self):
        """
        Shows the windows for the cameras.
        """

        self.window_name = f"[{self.friendly_id}] Camera - mxid: {self.mxid}"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 360)

    def update(self):
        """
        Updates the camera frames.
        """

        in_rgb = self.rgb_queue.tryGet()

        if in_rgb is None:
            return

        self.frame_rgb = in_rgb.getCvFrame()

        cv2.imshow(self.window_name, self.frame_rgb)

    def capture_still(self, timeout_ms: int = 1000):
        """
        Captures a still high-resolution image from the camera.
        """

        logger.info(f"{self.friendly_id}: Capturing still image...")

        # Empty the queue
        self.still_queue.tryGetAll()

        # Send a capture command
        logger.info(f"{self.friendly_id}: Sending 'still' event to the camera...")
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        self.control_queue.send(ctrl)

        # Wait for the still to be captured
        in_still = None
        start_time = time.time() * 1000
        while in_still is None:
            time.sleep(0.1)
            in_still = self.still_queue.tryGet()
            if time.time() * 1000 - start_time > timeout_ms:
                logger.warning(
                    f"{self.friendly_id}: Did not recieve still image - retrying..."
                )
                return self.capture_still(timeout_ms)

        still_rgb = cv2.imdecode(in_still.getData(), cv2.IMREAD_UNCHANGED)

        return still_rgb

    def calibrate(self):
        """
        Calibrates the camera.
        """

        frame_rgb = self.capture_still()
        if frame_rgb is None:
            logger.error(f"{self.friendly_id}: Did not recieve still image")
            return

        # pose = self.estimate_pose_checkerboard(frame_rgb)
        pose = self.estimate_pose_charuco(frame_rgb)

        # Save the calibration data
        workspace_path = os.path.expanduser("~/pcl")
        self.calibration_path = os.path.join(
            workspace_path, config.calibration_data_dir
        )
        os.makedirs(self.calibration_path, exist_ok=True)

        try:
            np.savez(
                os.path.join(self.calibration_path, f"camera_{self.mxid}.npz"),
                world_to_cam=pose["world_to_cam"],
                cam_to_world=pose["cam_to_world"],
                trans_vec=pose["trans_vec"],
                rot_vec=pose["rot_vec"],
            )
        except Exception:
            logger.error(f"{self.friendly_id}: Could not save calibration data")

        # Save the calibration image
        image_path = os.path.join(self.calibration_path, "img")
        os.makedirs(image_path, exist_ok=True)

        # Draw the detected markers and origin on the image
        if pose is not None:
            cv2.aruco.drawDetectedMarkers(
                frame_rgb, pose["marker_corners"], pose["marker_ids"]
            )

            reprojection = self.draw_origin(frame_rgb, pose)

            try:
                cv2.imwrite(
                    os.path.join(image_path, f"camera_{self.mxid}.png"), reprojection
                )
            except Exception:
                logger.error(f"{self.friendly_id}: Could not save calibration image")

        self.load_calibration()

        logger.info(f"Saved calibration for camera {self.mxid}")

    def estimate_pose_charuco(self, image):
        """
        Estimates the pose of the camera using a Charuco board.
        """

        if image is None:
            return

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        size = config.checkerboard_size
        square_length = config.square_length
        marker_length = config.marker_length

        board = cv2.aruco.CharucoBoard(
            size=size,
            squareLength=square_length,
            markerLength=marker_length,
            dictionary=aruco_dict,
        )

        corners_world = board.getChessboardCorners()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        charuco_detector = cv2.aruco.CharucoDetector(board)

        # Detect Charuco markers
        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids,
        ) = charuco_detector.detectBoard(frame_gray)

        if charuco_corners is None or charuco_ids is None:
            print("No Charuco corners detected")
            return

        # Sort the object points and image points based on charuco_ids
        object_points = np.array(
            [corners_world[i] for i in charuco_ids.flatten()], dtype=np.float32
        )
        image_points = np.array(charuco_corners, dtype=np.float32)

        # Refine the corner locations
        image_points = cv2.cornerSubPix(
            frame_gray,
            image_points,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        intrinsic_mat = self.intrinsic_mat
        distortion_coef = self.distortion_coef

        # Estimate the pose of the Charuco board
        ret, rot_vec, trans_vec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsic_mat,
            distortion_coef,
        )

        # compute transformation from world to camera space and wise versa
        rotM = cv2.Rodrigues(rot_vec)[0]
        world_to_cam = np.vstack((np.hstack((rotM, trans_vec)), np.array([0, 0, 0, 1])))
        cam_to_world = np.linalg.inv(world_to_cam)

        pose = {
            "world_to_cam": world_to_cam,
            "cam_to_world": cam_to_world,
            "trans_vec": trans_vec,
            "rot_vec": rot_vec,
            "intrinsics": intrinsic_mat,
            "distortion": distortion_coef,
            "charuco_corners": charuco_corners,
            "charuco_ids": charuco_ids,
            "marker_corners": marker_corners,
            "marker_ids": marker_ids,
        }

        return pose

    def estimate_pose_checkerboard(self, image):
        """
        Estimates the pose of the camera using a checkerboard.
        """

        checkerboard_size = config.checkerboard_size
        checkerboard_inner_size = (
            checkerboard_size[0] - 1,
            checkerboard_size[1] - 1,
        )
        square_size = config.square_length
        corners_world = np.zeros(
            (
                1,
                checkerboard_inner_size[0] * checkerboard_inner_size[1],
                3,
            ),
            np.float32,
        )
        corners_world[0, :, :2] = np.mgrid[
            0 : checkerboard_inner_size[0],
            0 : checkerboard_inner_size[1],
        ].T.reshape(-1, 2)
        corners_world *= square_size

        if image is None:
            return

        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # find the checkerboard corners
        found, corners = cv2.findChessboardCorners(
            frame_gray,
            checkerboard_inner_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_FAST_CHECK
            + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            return

        # refine the corner locations
        corners = cv2.cornerSubPix(
            frame_gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )

        intrinsic_mat = self.intrinsic_mat
        distortion_coef = self.distortion_coef

        # compute the rotation and translation from the camera to the checkerboard
        ret, rot_vec, trans_vec = cv2.solvePnP(
            corners_world,
            corners,
            intrinsic_mat,
            distortion_coef,
        )

        # compute transformation from world to camera space and wise versa
        rotM = cv2.Rodrigues(rot_vec)[0]
        world_to_cam = np.vstack((np.hstack((rotM, trans_vec)), np.array([0, 0, 0, 1])))
        cam_to_world = np.linalg.inv(world_to_cam)

        pose = {
            "world_to_cam": world_to_cam,
            "cam_to_world": cam_to_world,
            "trans_vec": trans_vec,
            "rot_vec": rot_vec,
            "intrinsics": intrinsic_mat,
            "distortion": distortion_coef,
        }

        return pose

    def draw_origin(self, image, pose):
        """
        Draws the origin on the image.
        """

        # Define the 3D points of the coordinate system
        points = np.float32([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1]])

        # Project the 3D points onto the 2D image plane using the camera pose
        projected_points, _ = cv2.projectPoints(
            points,
            pose["rot_vec"],
            pose["trans_vec"],
            pose["intrinsics"],
            pose["distortion"],
        )

        # Draw the coordinate system on the image
        reprojection = image.copy()
        [p_0, p_x, p_y, p_z] = projected_points.astype(np.int64)
        reprojection = cv2.line(
            reprojection, tuple(p_0[0]), tuple(p_x[0]), (0, 0, 255), 2
        )
        reprojection = cv2.line(
            reprojection, tuple(p_0[0]), tuple(p_y[0]), (0, 255, 0), 2
        )
        reprojection = cv2.line(
            reprojection, tuple(p_0[0]), tuple(p_z[0]), (255, 0, 0), 2
        )

        return reprojection
