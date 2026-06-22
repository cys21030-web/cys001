import time
import pathlib
import numpy as np

class ToFData:
    """ToF 距離資料處理類別。

    此類別負責將原始感測器輸出轉換成 8x8 的距離矩陣，
    同時進行基本的異常值修復與格式整理，方便後續訓練與視覺化使用。
    """
    
    width = 8 # ToF 數據的寬度，通常是 8 個點
    height = 8 # ToF 數據的高度，通常是 8 個點
    min_valid = 0
    max_valid = 500
    def __init__(self, raw_data: list[float], is_raw_byte: bool = True):
        self.data = np.zeros(
            shape = (self.height, self.width),
            dtype = int,
        )
        self.repaired_data = np.zeros_like(self.data)

        if is_raw_byte:
            # 將原始資料轉換為距離數據，假設每兩個字節代表一個距離值
            for idx in range(0, len(raw_data), 2):  
                a = raw_data[idx] # 低位字節
                b = raw_data[idx + 1] # 高位字節
                c = ((b << 8) | a) & 0xFFFF # 將兩個字節合併成一個距離值
                x, y = self.get_x_y_from_idx(idx // 2)
                self.data[y, x] = c # 將距離值添加到資料列表中
        
            self.repair() # 修復資料中的錯誤值
        else:
            self.data = np.array(
                raw_data,
                dtype=int
                ).reshape((self.height, self.width))
            self.repaired_data = np.array(
                raw_data,
                dtype=int
                ).reshape((self.height, self.width))

        

    def repair(self):
        repaired_data = np.zeros_like(self.data)
        for y1 in range(self.data.shape[0]):
            for x1 in range(self.data.shape[1]):
                val = self.data[y1, x1]
                
                # 檢查每個距離值，如果發現錯誤值，則嘗試使用周圍的有效值來修復它
                if not self.is_value_good(val):
                    # bug data detected, try to repair
                    sum, cnt = 0, 0
                    
                    # 從周圍的 3x3 區域中尋找有效的距離值，並計算它們的平均值來修復錯誤值
                    for y2 in range(y1 - 1, y1 + 2):
                        for x2 in range(x1 - 1, x1 + 2):
                            if x2 == x1 and y2 == y1:
                                continue

                            # 不計算數據範圍之外的點
                            if not self.is_pos_good(x2, y2):
                                continue
                            val2 = self.data[y2, x2]
                            
                            # 只考慮有效的距離值來修復錯誤值
                            if not self.is_value_good(val2):
                                continue
                            sum += val2
                            cnt += 1
                                    
                    # 如果周圍沒有有效的距離值，則無法修復錯誤值，這裡可以選擇丟出異常或使用預設值
                    if cnt <= 0:
                        raise Exception("沒有可用的鄰域資料可供修復")
                        
                    # 使用周圍有效距離值的平均值來修復錯誤值
                    repaired_data[y1, x1] = sum / cnt
                else:
                    # 如果距離值是有效的，則直接添加到修復後的資料列表中
                    repaired_data[y1, x1] = val

        # 最後將修復後的資料列表賦值給 self.data，以便後續使用
        # print(f"{len(self.data)} -> {len(repaired_data)} data repaired")
        self.repaired_data = repaired_data
                
    
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
        return value >= self.min_valid and value <= self.max_valid

    def disp(self):
        # 每8個一行
        for y in range(self.data.shape[0]):
            for x in range(self.data.shape[1]):
                print(f"{self.data[y, x]:04d}", end = " ")
            print()

    def save(self, filename: str = 'dat.dat') -> str:
        print(f"已儲存 {len(self.data)} 筆資料至 {filename}")
        with open(filename, "w") as f:
            for y in range(self.repaired_data.shape[0]):
                for x in range(self.repaired_data.shape[1]):
                    f.write(f'{self.repaired_data[y, x]:04d} ')
                f.write('\n')
                
    @classmethod
    def from_dat(cls, path: pathlib.Path) -> 'ToFData':
        with open(path, "r") as f:
            data = []
            for line in f:
                for val_str in line.split():
                    val = float(val_str)
                    data.append(val)
            return cls(data, False)
