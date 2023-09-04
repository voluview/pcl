import os
import time
from copy import deepcopy
from typing import List

import numpy as np
import open3d as o3d
from loguru import logger

import main.config as config
from main.helpers import FPSCounter, remove_color_points
from pointcloud.pointcloud import Pointcloud


class PointCloudVisualizer:
    """
    Visualizes the point clouds.
    """

    def __init__(self, pointclouds: List[Pointcloud]):
        self.pointclouds = pointclouds
        self.point_cloud = o3d.geometry.PointCloud()

        self.point_cloud_window = o3d.visualization.VisualizerWithKeyCallback()
        self.point_cloud_window.register_key_callback(
            ord("A"), lambda vis: self.align_point_clouds()
        )
        self.point_cloud_window.register_key_callback(
            ord("S"), lambda vis: self.save_point_cloud_alignment()
        )
        self.point_cloud_window.register_key_callback(
            ord("R"), lambda vis: self.reset_alignment()
        )
        self.point_cloud_window.register_key_callback(
            ord("P"), lambda vis: self.toggle_play()
        )
        self.point_cloud_window.register_key_callback(
            ord("V"), lambda vis: self.toggle_save()
        )
        self.point_cloud_window.register_key_callback(ord("Q"), lambda vis: self.quit())
        self.point_cloud_window.create_window(window_name="Pointcloud")
        self.point_cloud_window.add_geometry(self.point_cloud)

        # Set point size
        opt = self.point_cloud_window.get_render_option()
        opt.point_size = config.point_size

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.15, origin=[0, 0, 0]
        )
        self.point_cloud_window.add_geometry(origin)

        view = self.point_cloud_window.get_view_control()
        view.set_constant_z_far(config.max_range * 2)

        self.fps_counter = FPSCounter(duration=10)
        self.frame_count = 0

        self.record_path = os.path.join(
            os.path.expanduser("~/pcl"),
            "records",
        )
        os.makedirs(self.record_path, exist_ok=True)

        self.cache = []
        self.save = config.save

        self.running = True
        while self.running:
            self.update()

    def update(self, show=config.show):
        """
        Updates the point cloud frames.
        """

        self.point_cloud.clear()

        for pointcloud in self.pointclouds:
            pointcloud.update()

            self.point_cloud += pointcloud.point_cloud

        if config.remove_color_live:
            self.point_cloud = remove_color_points(
                self.point_cloud, config.color_to_remove, config.color_treshold
            )

        if self.save:
            self.cache.append(deepcopy(self.point_cloud))

        print(
            "FPS: {:.2f}".format(self.fps_counter.show()),
            f" | Saved frames: {len(self.cache)}",
            end="\r",
        )

        if show:
            self.point_cloud_window.update_geometry(self.point_cloud)
            self.point_cloud_window.poll_events()
            self.point_cloud_window.update_renderer()

    def align_point_clouds(self):
        """
        Fine aligns the point clouds with ICP (Iterative Closest Point).
        """

        voxel_radius = config.voxel_radius
        max_iter = config.max_iter

        master_point_cloud = self.pointclouds[0].point_cloud

        for pointcloud in self.pointclouds[1:]:
            for iter, radius in zip(max_iter, voxel_radius):
                target_down = master_point_cloud.voxel_down_sample(radius)
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
                )

                source_down = pointcloud.point_cloud.voxel_down_sample(radius)
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30)
                )

                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down,
                    target_down,
                    radius,
                    pointcloud.point_cloud_alignment,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iter
                    ),
                )

                pointcloud.point_cloud_alignment = result_icp.transformation

    def reset_alignment(self):
        """
        Resets the ICP point cloud alignment.
        """

        for pointcloud in self.pointclouds:
            pointcloud.point_cloud_alignment = np.identity(4)
            pointcloud.save_point_cloud_alignment()

    def toggle_play(self):
        """
        Toggles the visualizer and saving of pointclouds from cache.
        """

        self.running = not self.running
        self.point_cloud_window.destroy_window()
        logger.info("Visualizer stopped")

        # Save pointclouds
        for pointloud in self.cache:
            timestamp = round(time.time() * 1000)
            o3d.io.write_point_cloud(
                f"{self.record_path}/{timestamp}.ply",
                pointloud,
            )

            self.frame_count += 1
            print(f"Frames processed: {self.frame_count} / {len(self.cache)}", end="\r")

    def toggle_save(self):
        """
        Toggles saving of pointclouds.
        """

        self.save = not self.save
        logger.info("Saving pointclouds: {}", "ON" if self.save else "OFF")

    def save_point_cloud_alignment(self):
        """
        Saves the point cloud alignments to files.
        """

        for pointcloud in self.pointclouds:
            pointcloud.save_point_cloud_alignment()

    def quit(self):
        """
        Quits the visualizer."""

        self.running = False
