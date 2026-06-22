import math


class ViewAngle:
    """感測器視角與投影參數設定。

    此類別用來建立雷達感測器的觀測角度分布，並把每個距離點映射成 3D 空間座標。
    這些參數會影響 3D 點雲圖的呈現方式與推論資料的幾何意義。
    """

    def __init__(self, view_angle: float = 60, n_horiz: int = 8, n_vert: int = 8, sensor_pitch: float = -120.0):
        self.view_angle = view_angle
        self.n_horiz = n_horiz
        self.n_vert = n_vert
        self.sensor_pitch = sensor_pitch
        self.yaws = [
            math.radians(
                -view_angle / 2 + i * (view_angle / (self.n_horiz - 1))
                ) for i in range(self.n_horiz + 1)
            ]
        self.pitches = [
            math.radians(
                -view_angle / 2 + i * (view_angle / (self.n_vert - 1))
                ) for i in range(self.n_vert + 1)
            ]

    def get(self, idx: int) -> tuple[float, float]:
        x, y = idx % self.n_horiz, idx // self.n_horiz
        return self.yaws[x], self.pitches[y]
    
    def get_coord(self, idx: int, distance: float) -> tuple[float, float, float]:
        yaw, pitch = self.get(idx)
        pitch += math.radians(self.sensor_pitch)  # Apply sensor pitch offset
        x = distance * math.cos(pitch) * math.sin(yaw)
        y = distance * math.cos(pitch) * math.cos(yaw)
        z = distance * math.sin(pitch)
        return x, y, z