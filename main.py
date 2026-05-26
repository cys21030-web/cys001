from DFRobot_matrixLidar import DFRobot_matrixLidar_i2c
import time


class ToFData:
    width = 8
    height = 8
    def __init__(self, raw_data: list[int]):
        self.data = []

        for idx in range(0, len(raw_data), 2):
            a = raw_data[idx]
            b = raw_data[idx + 1]
            c = (b << 8) | a & 0xFFFF
            self.data.append(c)
        
        self.repair_data()
        

    def repair_data(self):
        reapired = []
        for idx in range(len(self.data)):
            val = self.data[idx]
            if not self.is_value_good(val):
                # bug data detected, try to repair
                sum, cnt = 0, 0
                x, y = self.get_x_y_from_idx(idx)
                for y2 in range(y - 1, y + 2):
                    for x2 in range(x - 1, x + 2):
                        if self.is_pos_good(x2, y2):
                            idx2 = self.get_idx_from_x_y(x2, y2)
                            val2 = self.data[idx2]
                            if self.is_value_good(val2):
                                sum += val2
                                cnt += 1
                    if cnt <= 0:
                        raise Exception("No good data to repair")
                    reapired.append(sum // cnt)
            else:
                # good point, just append
                reapired.append(self.data[idx])
        self.data = reapired
                
    
    def get_x_y_from_idx(self, idx: int) -> tuple[int, int]:
        x = idx % 8
        y = idx // 8
        return x, y
    
    def get_idx_from_x_y(self, x: int, y: int) -> int:
        return y * 8 + x

    def is_pos_good(self, x: int, y: int) -> bool:
        # Example condition - replace with actual logic
        # return 0 <= x < 8 and 0 <= y < 8
        return x >= 0 and x < 8 and y >= 0 and y < 8
    
    def is_value_good(self, value: int) -> bool:
        # Example condition - replace with actual logic
        return value <= 30000

    def disp(self):
        for idx in range(len(self.data)):
            if idx % 8 == 0 and idx > 0:
                print()
            print(f"{self.data[idx]:04d}", end = " ")

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

while True:
    raw_data = sensor.get_all_data()
    print(f'{raw_data = }')
    
    tof_data = ToFData(raw_data)
    tof_data.disp()
    
    if any(d > 30000 for d in tof_data.data):
        exit(1)
    print("---")
    
    time.sleep(0.25)



    #