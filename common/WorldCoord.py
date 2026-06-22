import json

from common.ToFData import ToFData
from common.ViewAngle import ViewAngle

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:  # pragma: no cover - depends on runtime environment
    plt = None
    Axes3D = None

import numpy as np

try:
    from common.matplotlib_utils import configure_matplotlib_chinese
    configure_matplotlib_chinese()
except Exception:  # pragma: no cover - fallback for minimal environments
    pass


class WorldCoord:
    """將距離矩陣轉成 3D 點雲資料。

    這個類別會依據感測器視角與每個距離值，計算出對應的世界座標位置，
    以便在 3D 視圖中顯示感測器掃描到的空間分布。
    """
    def __init__(self, raw_data: np.ndarray, view_angles: ViewAngle):
        
        """
        將 ToF 資料映射到世界坐標系中，根據給定感應器的角度來計算每個距離值在世界坐標系中的位置。
        """
        self.xs = np.zeros((64,))
        self.ys = np.zeros((64,))
        self.zs = np.zeros((64,))
        self.view_angles = view_angles
        
        data_width = raw_data.shape[1]
        data_height = raw_data.shape[0]

        for y1 in range(data_height):
            for x1 in range(data_width):
                val = raw_data.data[y1, x1]
                idx = y1 * data_width + x1
                x, y, z = view_angles.get_coord(idx, val)
                self.xs[idx] = x
                self.ys[idx] = y
                self.zs[idx] = z
            
    def plot(self):
        # 使用 matplotlib 來繪製 3D 點雲
        if plt is None:
            return
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.xs, self.ys, self.zs)
        ax.set_xlabel('X 軸')
        ax.set_ylabel('Y 軸')
        ax.set_zlabel('Z 軸')
        plt.savefig('world_coordinates.png')
    
    def save(self, filename: str = 'world_coordinates.json'):
        # 將世界坐標數據保存為 JSON 文件
        data = {
            'x': self.xs,
            'y': self.ys,
            'z': self.zs
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
            
    def save_as_ply(self, filename: str = 'world_coordinates.ply'):
        # 將世界坐標數據保存為 PLY 文件
        with open(filename, 'w') as f:
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write(f'element vertex {len(self.xs)}\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')
            f.write('end_header\n')
            for x, y, z in zip(self.xs, self.ys, self.zs):
                f.write(f'{x} {y} {z}\n')
            
            
            
        
    