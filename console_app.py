from DFRobot.DFRobot_matrixLidar import DFRobot_matrixLidar_i2c
import time
import json
import pathlib
from common.ToFData import ToFData
from common.ViewAngle import ViewAngle
from common.WorldCoord import WorldCoord

if __name__ == "__main__":
    try:
        # Initialize the Lidar sensor, at address 0x33.
        sensor = DFRobot_matrixLidar_i2c(0x33)
        while sensor.begin() != 0:
            print("LiDAR 初始化失敗，正在重試...")
            time.sleep(0.1)
        print("LiDAR 初始化成功")
            
        while sensor.set_Ranging_Mode(8) != 0:
            print("設定量測模式失敗，正在重試...")
            time.sleep(0.1)
        print("量測模式設定成功")

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
    except Exception as e:
        import traceback
        with open(".error.log", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        raise e

    # while True:
    #     raw_data = sensor.get_all_data()
    #     print(f'{raw_data = }')
        
    #     tof_data = ToFData(raw_data)
    #     tof_data.disp()
        
    #     # data_saving
        
    #     print("---")
        
    #     time.sleep(0.25)



        #