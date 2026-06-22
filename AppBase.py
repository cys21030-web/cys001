# -*- coding: utf-8 -*-
from imgui_bundle import imgui, hello_imgui, implot, implot3d
import logging
from ai.ToFDataLabel import ToFDataLabel
from common.ToFData import ToFData
from common.ToFSensor import ToFSensor
from datetime import datetime
from pathlib import Path
import math
import numpy as np
from common.WorldCoord import WorldCoord
import random
import colorsys

class App:
    def __init__(self):
        logging.info("初始化應用程式")
        
        # 設置基礎的 ImGui 與 ImPlot / ImPlot3D 記憶體上下文，保證渲染正常
        self.imgui_context = imgui.get_current_context()
        if self.imgui_context is None:
            self.imgui_context = imgui.create_context()
            imgui.set_current_context(self.imgui_context)

        self.implot_context = implot.get_current_context()
        if self.implot_context is None:
            self.implot_context = implot.create_context()

        self.implot3d_context = implot3d.get_current_context()
        if self.implot3d_context is None:
            self.implot3d_context = implot3d.create_context()

        # 預設 3D 圖表的觀察仰角與視角
        self.plot_box_elevation = 20
        self.plot_box_azimuth = 80

        # 當前幀的 Lidar 3D 世界座標點雲與 2D heatmap 距離矩陣數據
        self.cloud_points = None
        self.raw_data = np.zeros((8, 8), dtype=np.float32)

        self.window_title = "Base App"
        self.__view_angle = 0
        self.__sensor_pitch = 0
        self.__sensor_ready = True
        self.__frame_cnt_total = 0
        self.__frame_cnt_valid = 0
        self.snapshot_dir = Path('./snapshot')

        # Generate and register the custom colormap
        self.custom_colormap_id = self._register_custom_colormap()

    @property
    def win_title(self):
        return self.window_title

    @property
    def view_angle(self):
        return self.__view_angle
    
    @view_angle.setter
    def view_angle(self, value):
        self.__view_angle = value

    @property
    def sensor_pitch(self):
        return self.__sensor_pitch
    
    @sensor_pitch.setter
    def sensor_pitch(self, value):
        self.__sensor_pitch = value

    @property
    def sensor_ready(self):
        return self.__sensor_ready
    
    @property
    def sensor_frame_cnt_total(self) -> int:
        return self.__frame_cnt_total
    
    @property
    def sensor_frame_cnt_valid(self) -> int:
        return self.__frame_cnt_valid

    def load_custom_fonts(self):
        # 載入中文字型，防止渲染亂碼
        imgui.set_current_context(self.imgui_context)
        io = imgui.get_io()
        font_path = "LXGWWenKaiMonoTC-Regular.ttf" 
        font_size = 20.0
        try:
            io.fonts.add_font_from_file_ttf(font_path, font_size, None)
        except Exception:
            print(f"警告：無法載入字型 {font_path}")

    def config_docking(self) -> hello_imgui.DockingParams:
        # 子模式的預設視窗佈局設定
        docking_params = hello_imgui.DockingParams()

        split_left = hello_imgui.DockingSplit()
        split_left.initial_dock = 'MainDockSpace'
        split_left.new_dock = 'LeftSpace'
        split_left.direction = imgui.Dir.left
        split_left.ratio = 0.25
        
        split_bot = hello_imgui.DockingSplit()
        split_bot.initial_dock = 'MainDockSpace'
        split_bot.new_dock = 'BottomSpace'
        split_bot.direction = imgui.Dir.down
        split_bot.ratio = 0.6

        docking_params.docking_splits = [split_left, split_bot]

        # 註冊點雲圖、熱力圖與參數設定視窗
        w_3dplot = hello_imgui.DockableWindow(
            label_ = '點雲圖',
            dock_space_name_ = 'BottomSpace',
            gui_function_ = self.gui_3d_plot,
            is_visible_ = True,
            can_be_closed_= False
        )
        
        w_heatmap = hello_imgui.DockableWindow(
            label_ = '原始距離資料',
            dock_space_name_ = 'MainDockSpace',
            gui_function_ = self.gui_heat_map,
            is_visible_ = True,
            can_be_closed_ = False
        )

        w_settings = hello_imgui.DockableWindow(
            label_ = '設定',
            dock_space_name_ = 'LeftSpace',
            gui_function_ = self.gui_settings,
            is_visible_ = True,
            can_be_closed_ = False
        )

        docking_params.dockable_windows = [
            w_heatmap,
            w_3dplot,
            w_settings
        ]

        return docking_params

    def gui(self):
        pass

    def update_data(self):
        pass

    def gui_settings(self):
        self.gui_settings_plot_view()

    def gui_settings_plot_view(self):
        # 繪製傳感器方向滑動條與點雲觀察參數
        imgui.separator_text('感測器方向')
        changed, self.view_angle = imgui.slider_float("視角", self.view_angle, 0.0, 120.0)
        changed, self.sensor_pitch = imgui.slider_float("感測器俯仰角", self.sensor_pitch, -360, 360.0)

        imgui.separator_text('3D 圖表')
        changed, self.plot_box_elevation = imgui.slider_float('仰角', self.plot_box_elevation, -90.0, 90.0)
        changed, self.plot_box_azimuth = imgui.slider_float('方位角', self.plot_box_azimuth, -360.0, 360.0)

    def _register_custom_colormap(self, num_colors: int = 256):
        """
        Generate and register a custom colormap from blue to red.
        """
        colors = []
        for i in range(num_colors):
            hue = (240 - (i / (num_colors - 1)) * 240) / 360.0
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append((rgb[0], rgb[1], rgb[2], 1.0))
        
        return implot.add_colormap("custom_colormap", np.array(colors, dtype=np.float32), qual=False)

    def gui_heat_map(self):
        # 繪製 2D 距離熱力圖 (Heatmap)
        if not self.sensor_ready:
            return
        
        if self.raw_data is not None:
            implot.push_colormap(self.custom_colormap_id)
            if implot.begin_plot("原始距離資料"):
                implot.setup_legend(implot.Location_.east, implot.LegendFlags_.outside)
                implot.plot_heatmap(
                    "原始距離資料",
                    self.raw_data,
                    float(ToFData.min_valid),
                    float(ToFData.max_valid),
                    bounds_min = (0, 0),
                    bounds_max = (1.0, 1.0)
                )
            implot.end_plot()
            implot.pop_colormap()
    
    def gui_3d_plot(self):
        # 繪製 3D 立體點雲映射圖 (3D Scatter Plot)
        if implot3d.begin_plot("點雲圖"):
            implot3d.setup_legend(
                implot3d.Location_.east,
                implot3d.LegendFlags_.horizontal
            )
            implot3d.setup_box_rotation(
                self.plot_box_elevation,
                self.plot_box_azimuth,
                True,
                implot3d.Cond_.always
            )
            implot3d.setup_axes("X 軸", "Y 軸", "Z 軸")
            implot3d.setup_box_scale(1.0, 2.0, 1.0)
            implot3d.setup_axes_limits(-80.0, 80.0, 25.0, -250.0, -250.0, 50.0)
            if self.cloud_points is not None:
                implot3d.plot_scatter(
                    "world coord",
                    self.cloud_points.xs,
                    self.cloud_points.ys,
                    self.cloud_points.zs,
                )
            implot3d.end_plot()

    def save_raw_data(self, filename: str = 'dat.dat') -> str:
        # 格式化儲存 Lidar 數據
        print(f"已儲存 {len(self.raw_data)} 筆資料至 {filename}")
        with open(filename, "w") as f:
            for y in range(self.raw_data.shape[0]):
                for x in range(self.raw_data.shape[1]):
                    f.write(f'{self.raw_data[y, x]:0.3f} ')
                f.write('\n')
