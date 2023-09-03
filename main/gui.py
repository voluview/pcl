import tkinter as tk
from tkinter import ttk

from loguru import logger

from main.controller import Controller


class Gui:
    """
    Main GUI class
    """

    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Camera GUI")
        # self.window.geometry("300x300")

        self.window.style = ttk.Style()
        self.window.style.theme_use("clam")

        self.window.columnconfigure([0, 1], weight=1)
        self.window.rowconfigure([0, 1, 2, 3], weight=1)

        self.init_calibrator_button = ttk.Button(
            self.window,
            text="Init Calibrator",
            command=self.initialise_calibrator,
            state=tk.DISABLED,
        )
        self.init_pointcloud_button = ttk.Button(
            self.window,
            text="Init Pointcloud",
            command=self.initialise_pointcloud,
        )
        self.start_button = ttk.Button(
            self.window, text="Start", command=self.start_video, state=tk.DISABLED
        )
        self.stop_button = ttk.Button(
            self.window, text="Stop", command=self.stop_video, state=tk.DISABLED
        )
        self.calibrate_button = ttk.Button(
            self.window,
            text="Calibrate",
            command=self.start_calibration,
            state=tk.DISABLED,
        )
        self.close_button = ttk.Button(
            self.window, text="Close", command=self.close_application
        )

        self.init_calibrator_button.grid(row=0, column=0, padx=10, pady=10)
        self.init_pointcloud_button.grid(row=0, column=1, padx=10, pady=10)
        self.calibrate_button.grid(row=1, column=0, padx=10, pady=10)
        self.start_button.grid(row=1, column=1, padx=10, pady=10)
        self.stop_button.grid(row=2, column=1, padx=10, pady=10)
        self.close_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.controller = None

        logger.info("App started")

        # self.initialise_pointcloud()  # Use to automatically start the pointcloud

    def run(self):
        """
        Runs the GUI.
        """

        self.window.mainloop()

    def initialise_calibrator(self):
        """
        Initialises the calibrator.
        """

        if self.controller is not None:
            self.controller = None

        self.controller = Controller()
        self.controller.init_calibrator()

        self.start_button["state"] = tk.NORMAL
        self.calibrate_button["state"] = tk.NORMAL

    def initialise_pointcloud(self):
        """
        Initialises the pointcloud.
        """

        if self.controller is not None:
            self.controller = None

        self.controller = Controller()
        self.controller.init_pointcloud()

        self.start_button["state"] = tk.NORMAL
        self.calibrate_button["state"] = tk.DISABLED

    def start_calibration(self):
        """
        Starts the calibration process.
        """

        if self.controller is not None:
            self.controller.calibrate_cameras()

            self.start_button["state"] = tk.NORMAL
            self.stop_button["state"] = tk.DISABLED

    def start_video(self):
        """
        Starts the video stream.
        """

        if self.controller is not None:
            self.controller.start_updating()

            self.start_button["state"] = tk.DISABLED
            self.stop_button["state"] = tk.NORMAL

            self.show_video()

    def stop_video(self):
        """
        Stops the video stream.
        """

        if self.controller is not None:
            self.controller.stop_updating()

            self.start_button["state"] = tk.NORMAL
            self.stop_button["state"] = tk.DISABLED

    def close_application(self):
        """
        Closes the application.
        """

        if self.controller is not None:
            self.controller.close()

        self.window.destroy()
        logger.info("App closed")

    def show_video(self):
        """
        Shows the video stream.
        """

        while self.controller.updating:
            self.controller.update()
            self.window.update()
