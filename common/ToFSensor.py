from DFRobot.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c
import time
from common.ToFData import ToFData
from common.ViewAngle import ViewAngle
from common.WorldCoord import WorldCoord
import logging
import threading

class ToFSensor:
    def __init__(self, address: int = 0x33):
        self.address = address
        self.status = "Uninitialized"
        self.sensor = None
        self.last_raw_data = None
        # self.last_cloud_points = None
        self.frame_cnt_total = 0
        self.frame_cnt_valid = 0
        self.view_angle = ViewAngle() 
        thread = threading.Thread(
            target=self.init_sensor,
            daemon=True
        )
        thread.start()
        self.fetch_thread = None
        self.keep_running = True

    @property
    def ready(self):
        return self.sensor is not None

    def init_sensor(self):
        logging.info("Initializing ToF Sensor...")
        self.status = "Initializing"
        tmp_sensor = DFRobot_matrixLidar_i2c(self.address)

        while tmp_sensor.begin() != 0:
            time.sleep(0.333)
        logging.info("Setting ranging mode.")
        self.status = "Setting Ranging Mode"
        
        while tmp_sensor.set_Ranging_Mode(8) != 0:
            time.sleep(0.333)

        self.sensor = tmp_sensor
        self.status = "Ready"
        logging.info("Lidar initialized successfully!")

        self.fetch_thread = threading.Thread(
            target=self.__fetch,
            daemon=True
        )
        self.fetch_thread.start()

    def __exit__(self, exc_type, exc, tb):
        logging.info("Cleaning up ToF Sensor resources...")
        self.keep_running = False
        if self.fetch_thread:
            self.fetch_thread.join(timeout=1.0)
        logging.info("ToF Sensor cleanup complete.")


    def __fetch(self):
        while self.keep_running:
            failed = True
            tof_data = None

            while failed:
                raw_data = self.sensor.get_all_data()
                self.frame_cnt_total += 1

                if len(raw_data) <= 0:
                    continue
                
                try:
                    tof_data = ToFData(raw_data)

                    # pts = WorldCoord(tof_data, self.view_angle)
                    failed = False
                    self.frame_cnt_valid += 1
                except Exception as e:
                    # logging.exception("Error processing ToF data: %s", e)
                    time.sleep(0.25)

            if not failed:
                self.last_raw_data = tof_data
            # self.last_cloud_points = pts

            time.sleep(0.15)