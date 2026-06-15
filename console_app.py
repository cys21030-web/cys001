from DFRobot.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c
import time
import json
import pathlib
from common.ToFData import ToFData
from common.ViewAngle import ViewAngle
from common.WorldCoord import WorldCoord

if __name__ == "__main__":
    # Initialize the Lidar sensor, at address 0x33.
    sensor = DFRobot_matrixLidar_i2c(0x33)
    while sensor.begin() != 0:
        print("Lidar initialization failed, retrying...")
        time.sleep(0.1)
    print("Lidar initialized successfully")
        
    while sensor.set_Ranging_Mode(8) != 0:
        print("Failed to set ranging mode, retrying...")
        time.sleep(0.1)
    print("Ranging mode set successfully")

    view_angle = ViewAngle() 
    while True:
        raw_data = sensor.get_all_data()
        if len(raw_data) <= 0:
            continue
        
        try:
            tof_data = ToFData(raw_data)
            tof_data.save()
            pts = WorldCoord(tof_data, view_angle)
            pts.plot()
            pts.save_as_ply()
        
        except Exception as e:
            print(str(e))
            continue
            
        break

    # while True:
    #     raw_data = sensor.get_all_data()
    #     print(f'{raw_data = }')
        
    #     tof_data = ToFData(raw_data)
    #     tof_data.disp()
        
    #     # data_saving
        
    #     print("---")
        
    #     time.sleep(0.25)



        #