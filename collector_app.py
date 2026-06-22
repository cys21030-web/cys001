# -*- coding: utf-8 -*-
import logging
import random
from datetime import datetime
from pathlib import Path
import numpy as np

from imgui_bundle import imgui, hello_imgui, implot, implot3d

from ai.ToFDataLabel import ToFDataLabel
from common.ToFSensor import ToFSensor
from common.WorldCoord import WorldCoord
from AppBase import App as AppBase


class CollectorApp(AppBase):
    """資料採集模式。

    此模式用於從 ToF 感測器讀取原始距離數據，並將其整理成可訓練的樣本。
    使用者可以選擇標籤、啟動批次採集，並把結果保存到 snapshot/ 之下。
    """

    def __init__(self):
        super().__init__()
        
        # 初始化 ToF 傳感器驅動與採集狀態
        self.tof_sensor = ToFSensor()
        self.selected_label = 0

        # 用於追蹤各類別當前的採集樣本數與目標數量
        self.sample_count = np.zeros((ToFDataLabel.label_cnts), dtype=np.int32)
        self.target_count = np.zeros((ToFDataLabel.label_cnts), dtype=np.int32)
        self.is_collecting = False
        self.last_saved_frame = 0
        self.last_disp_frame = 0

        self.__win_title = "升降機平層誤差監察儀"

    @property
    def view_angle(self):
        if not self.sensor_ready:
            return 0.0
        return self.tof_sensor.view_angle.view_angle
    
    @view_angle.setter
    def view_angle(self, value):
        if not self.sensor_ready:
            return
        self.tof_sensor.view_angle.view_angle = value

    @property
    def sensor_pitch(self):
        if not self.sensor_ready:
            return 0.0
        return self.tof_sensor.view_angle.sensor_pitch
    
    @sensor_pitch.setter
    def sensor_pitch(self, value):
        if not self.sensor_ready:
            return
        self.tof_sensor.view_angle.sensor_pitch = value

    @property
    def sensor_ready(self):
        return self.tof_sensor.ready
    
    @property
    def sensor_frame_cnt_total(self) -> int:
        if not self.sensor_ready:
            return 1
        return self.tof_sensor.frame_cnt_total
    
    @property
    def sensor_frame_cnt_valid(self) -> int:
        if not self.sensor_ready:
            return 0
        return self.tof_sensor.frame_cnt_valid

    def gui(self):
        self.update_data()
        self.snapshot()

    def update_data(self):
        # 獲取傳感器最新影格，並套用簡單的卡爾曼濾波 (平滑權重 0.5) 進行數值降噪
        if self.tof_sensor.last_raw_data is None:
            return

        if self.last_disp_frame >= self.sensor_frame_cnt_total:
            return
        
        self.raw_data = self.raw_data * 0.5 + self.tof_sensor.last_raw_data.repaired_data * 0.5
        self.cloud_points = WorldCoord(self.raw_data, self.tof_sensor.view_angle)

    def gui_settings(self):
        # 渲染左側傳感器參數調整、硬體狀態與採集按鈕控制
        super().gui_settings_plot_view()
        imgui.separator_text("Sensor")

        if not self.sensor_ready:
            imgui.text("ToF 感測器尚未就緒。")
            return
        imgui.label_text('狀態', self.tof_sensor.status)
        imgui.label_text('影格', f'{self.sensor_frame_cnt_valid:04d} / {self.sensor_frame_cnt_total:04d}')

        imgui.separator_text("收集樣本")
        imgui.text_wrapped("先選一個標籤，再按『開始採集』。系統會把資料存成一筆一筆的樣本。")
        changed, new_label = imgui.combo(
            '標籤',
            self.selected_label,
            ToFDataLabel.combo_labels
        )
        if changed and not self.is_collecting:
            self.selected_label = new_label

        if self.is_collecting:
            if imgui.button('停止採集'):
                self.is_collecting = False
        else:
            if imgui.button('開始採集'):
                self.enable_snapshot()

        imgui.separator()

        # 顯示當前已收集的各標籤檔案數量
        for idx in range(ToFDataLabel.label_cnts):
            imgui.label_text(ToFDataLabel.labels[idx].name, f'{self.sample_count[idx]}')

    def enable_snapshot(self):
        # 啟動自動批次採集，每次點擊預設自動採集 100 影格
        self.target_count[self.selected_label] = self.sample_count[self.selected_label] + 100
        self.is_collecting = True

    def snapshot(self):
        # 批次數據自動儲存，限制儲存影格間隔 (加入少量隨機因子防抖)，確保數據多樣性
        if not self.is_collecting:
            return
        
        if self.sample_count[self.selected_label] >= self.target_count[self.selected_label]:
            self.is_collecting = False
            return
        
        if self.raw_data is None or self.cloud_points is None:
            return
        
        if self.sensor_frame_cnt_valid <= self.last_saved_frame + random.randint(0, 3) * 5:
            return
        self.last_saved_frame = self.sensor_frame_cnt_valid

        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d-%H-%M-%S.%f")[:-3]
        label_name = ToFDataLabel.labels[self.selected_label].name
        output_name = f'{label_name}-{now_str}'
        output_dir = self.snapshot_dir / f'{label_name}'

        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        self.save_raw_data(output_dir / f'{output_name}.dat')
        self.sample_count[self.selected_label] += 1

    def save_raw_data(self, filename: str = 'dat.dat') -> str:
        # 格式化輸出距離數據，寫入 .dat 檔案
        print(f"已儲存 {len(self.raw_data)} 筆資料至 {filename}")
        with open(filename, "w") as f:
            for y in range(self.raw_data.shape[0]):
                for x in range(self.raw_data.shape[1]):
                    f.write(f'{self.raw_data[y, x]:0.3f} ')
                f.write('\n')
