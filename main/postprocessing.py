import glob
import os

import config
import open3d
from helpers import remove_color_points


def main():
    """
    Processes the point cloud files.
    """

    directory = os.path.expanduser("~/view")
    input_folder = os.path.join(directory, "records")
    output_folder = os.path.join(directory, "records", "postprocessed")

    os.makedirs(output_folder, exist_ok=True)

    files = glob.glob("*.ply", root_dir=input_folder)
    counter = 1

    for file in files:
        print(f"Processing file {counter}/{len(files)}:", file, end="\r")
        counter += 1

        pcd = open3d.io.read_point_cloud(
            os.path.join(input_folder, file), remove_nan_points=True
        )

        if config.remove_color_post:
            pcd = remove_color_points(
                pcd, config.color_to_remove, config.color_treshold
            )

        if config.remove_statistical_outlier:
            pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.1)[0]

        open3d.io.write_point_cloud(os.path.join(output_folder, file), pcd)


if __name__ == "__main__":
    main()
