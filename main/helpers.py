import time

import numpy as np
import open3d as o3d


class FPSCounter:
    """
    Calculates the average FPS over a given duration.
    """

    def __init__(self, duration=5):
        self.frame_timestamps = []
        self.duration = duration  # seconds

    def show(self):
        current_time = time.time()
        self.frame_timestamps.append(current_time)

        # Remove timestamps older than 'duration' seconds
        while self.frame_timestamps and (
            current_time - self.frame_timestamps[0] > self.duration
        ):
            self.frame_timestamps.pop(0)

        # Calculate average FPS over the remaining timestamps
        num_frames = len(self.frame_timestamps)
        avg_fps = num_frames / self.duration if self.frame_timestamps else 0

        # print("FPS: {:.2f}".format(avg_fps), end="\r")

        return avg_fps


def remove_color_points(pointcloud, target_color, threshold):
    """
    Removes shades of the specified color from the point cloud.
    """

    color_map = {
        "black": [0, 0, 0],
        "white": [255, 255, 255],
        "red": [255, 0, 0],
        "green": [0, 255, 0],
        "blue": [0, 0, 255],
        "cyan": [0, 255, 255],
        "magenta": [255, 0, 255],
        "yellow": [255, 255, 0],
    }

    if target_color not in color_map:
        raise ValueError(f"Invalid color. Choose from {', '.join(color_map.keys())}.")

    target_rgb = np.array(color_map[target_color]) / 255.0  # Normalize to [0, 1]

    colors = np.asarray(pointcloud.colors)
    distances = np.linalg.norm(colors - target_rgb, axis=1)
    color_indices = np.where(distances < threshold)[0]

    pointcloud.points = o3d.utility.Vector3dVector(
        np.delete(np.asarray(pointcloud.points), color_indices, axis=0)
    )
    pointcloud.colors = o3d.utility.Vector3dVector(
        np.delete(colors, color_indices, axis=0)
    )

    return pointcloud
