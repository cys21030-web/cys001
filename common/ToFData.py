import time
import pathlib

class ToFData:
    """
    這個類別用來處理 ToF (Time of Flight) 資料，包含資料修復、顯示和儲存功能。
    ToF 資料通常是從感測器讀取的距離數據，可能會有一些錯誤或異常值。
    這個類別會嘗試修復這些錯誤值，並提供方法來顯示和儲存修復後的資料。
    """
    
    width = 8 # ToF 數據的寬度，通常是 8 個點
    height = 8 # ToF 數據的高度，通常是 8 個點
    def __init__(self, raw_data: list[int]):
        self.data = []

        # 將原始資料轉換為距離數據，假設每兩個字節代表一個距離值
        for idx in range(0, len(raw_data), 2):  
            a = raw_data[idx] # 低位字節
            b = raw_data[idx + 1] # 高位字節
            c = (b << 8) | a & 0xFFFF # 將兩個字節合併成一個距離值
            self.data.append(c) # 將距離值添加到資料列表中
        
        self.repair_data() # 修復資料中的錯誤值
        

    def repair_data(self):
        reapired = []
        for idx in range(len(self.data)):
            val = self.data[idx]
            
            # 檢查每個距離值，如果發現錯誤值，則嘗試使用周圍的有效值來修復它
            if not self.is_value_good(val):
                # bug data detected, try to repair
                sum, cnt = 0, 0
                
                # 從周圍的 3x3 區域中尋找有效的距離值，並計算它們的平均值來修復錯誤值
                x, y = self.get_x_y_from_idx(idx)
                for y2 in range(y - 1, y + 2):
                    for x2 in range(x - 1, x + 2):
                        # 不計算數據範圍之外的點
                        if self.is_pos_good(x2, y2):
                            idx2 = self.get_idx_from_x_y(x2, y2)
                            val2 = self.data[idx2]
                            
                            # 只考慮有效的距離值來修復錯誤值
                            if self.is_value_good(val2):
                                sum += val2
                                cnt += 1
                                
                    # 如果周圍沒有有效的距離值，則無法修復錯誤值，這裡可以選擇丟出異常或使用預設值
                    if cnt <= 0:
                        raise Exception("No good data to repair")
                    
                # 使用周圍有效距離值的平均值來修復錯誤值
                reapired.append(sum // cnt)
            else:
                # 如果距離值是有效的，則直接添加到修復後的資料列表中
                reapired.append(self.data[idx])
                
        # 最後將修復後的資料列表賦值給 self.data，以便後續使用
        print(f"{len(self.data)} -> {len(reapired)} data repaired")
        self.data = reapired
                
    
    def get_x_y_from_idx(self, idx: int) -> tuple[int, int]:
        # 根據索引計算對應的 x 和 y 坐標，假設資料是以行優先的方式存儲的
        x = idx % 8
        y = idx // 8
        return x, y
    
    def get_idx_from_x_y(self, x: int, y: int) -> int:
        # 根據 x 和 y 坐標計算對應的索引，假設資料是以行優先的方式存儲的
        return y * 8 + x

    def is_pos_good(self, x: int, y: int) -> bool:
        # Example condition - replace with actual logic
        # return 0 <= x < 8 and 0 <= y < 8
        return x >= 0 and x < 8 and y >= 0 and y < 8
    
    def is_value_good(self, value: int) -> bool:
        # Example condition - replace with actual logic
        return value <= 30000

    def disp(self):
        # 每8個一行
        for idx in range(len(self.data)):
            if idx % 8 == 0 and idx > 0:
                print()
            print(f"{self.data[idx]:04d}", end = " ")

    def save(self) -> str:
        now = time.time()
        now_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(now))
        print(f"{len(self.data)} data saved to tof_data_{now_str}.dat")
        with open(f"tof_data_{now_str}.dat", "w") as f:
            for idx in range(0, len(self.data)):
                f.write(f'{self.data[idx]:04d} ')
                if ((idx + 1)  % 8) == 0 and idx > 0:
                    f.write("\n")
            
    @staticmethod
    def from_dat(cls, path: pathlib.Path) -> 'ToFData':
        with open(path, "r") as f:
            data = []
            for line in f:
                for val_str in line.split():
                    val = int(val_str)
                    data.append(val)
            return cls(data)
