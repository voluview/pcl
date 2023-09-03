from typing import List

import depthai as dai
from loguru import logger

from calibrator.calibrator import Calibrator
from pointcloud.point_cloud_visualizer import PointCloudVisualizer
from pointcloud.pointcloud import Pointcloud


class Controller:
    def __init__(self):
        self.device_infos = dai.Device.getAllAvailableDevices()
        if len(self.device_infos) == 0:
            logger.exception("No devices found!")
            raise RuntimeError("No devices found!")
        else:
            logger.info(f"Found {str(len(self.device_infos))} devices")

        self.device_infos.sort(key=lambda x: x.getMxId(), reverse=True)

        self.show_calibrator = False
        self.show_pointcloud = False
        self.updating = False

        logger.info("Controller initialized")

    def init_calibrator(self):
        self.calibrators: List[Calibrator] = []

        friendly_id = 0
        for device_info in self.device_infos:
            friendly_id += 1
            calibrator = Calibrator(device_info, friendly_id)
            self.calibrators.append(calibrator)

        self.show_calibrator = True
        self.show_pointcloud = False

        logger.info("Calibrator initialized")

    def init_pointcloud(self):
        self.pointclouds: List[Pointcloud] = []

        friendly_id = 0
        for device_info in self.device_infos:
            friendly_id += 1
            pointcloud = Pointcloud(device_info, friendly_id)
            self.pointclouds.append(pointcloud)

        self.show_calibrator = False
        self.show_pointcloud = True

        self.pointcloudvisualizer = PointCloudVisualizer(self.pointclouds)

        # TODO: Fix: the way to update the pointcloud

        # while True:
        #     self.pointcloudvisualizer.update()

        logger.info("Pointcloud initialized")

    def update(self):
        if self.show_calibrator:
            for calibrator in self.calibrators:
                calibrator.update()

        if self.show_pointcloud:
            self.pointcloudvisualizer.update()

    def manual_update(self):
        while True:
            self.pointcloudvisualizer.update()

    def start_updating(self):
        logger.info("Start updating frames")
        self.updating = True

    def stop_updating(self):
        logger.info("Stop updating frames")
        self.updating = False

    def calibrate_cameras(self):
        logger.info("Calibrating cameras...")
        self.updating = False
        for calibrator in self.calibrators:
            calibrator.calibrate()
        logger.info("Cameras calibrated")

    def close(self):
        self.stop_updating()
        # for camera in self.cameras:
        #     camera.close()
        logger.info("Cameras closed")
