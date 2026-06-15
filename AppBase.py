from imgui_bundle import imgui, hello_imgui, implot, implot3d
import logging
from ai.ToFDataLabel import ToFDataLabel
from common.ToFSensor import ToFSensor
from datetime import datetime
from pathlib import Path
import math
import numpy as np
from common.WorldCoord import WorldCoord
import random


class App:
    def __init__(self):
        logging.info("Initializing App")
        self.ctx_imgui = imgui.get_current_context()
        if self.ctx_imgui is None:
            self.ctx_imgui = imgui.create_context()
            imgui.set_current_context(self.ctx_imgui)

        self.ctx_implot = implot.get_current_context()
        if self.ctx_implot is None:
            self.ctx_implot = implot.create_context()

        self.ctx_implot3d = implot3d.get_current_context()
        if self.ctx_implot3d is None:
            self.ctx_implot3d = implot3d.create_context()

        self.plot_box_elvation = 20
        self.plot_box_azimuth = 80

        self.cloud_points = None
        self.raw_data = np.zeros((8, 8), dtype=np.float32)

        self.__win_title = "Base App"
        self.__view_angle = 0
        self.__sensor_pitch = 0
        self.__sensor_ready = True
        self.__frame_cnt_total = 0
        self.__frame_cnt_valid = 0
        self.snapshot_dir = Path('./snapshot')

    @property
    def win_title(self):
        return self.__win_title

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
        print(imgui.get_version())
        imgui.set_current_context(self.ctx_imgui)
        io = imgui.get_io()
        
        font_path = "LXGWWenKaiMonoTC-Regular.ttf" 
        font_size = 18.0
        
        try:
            io.fonts.add_font_from_file_ttf(font_path, font_size, None)
        except Exception:
            print(f"Warning: Could not load font from {font_path}")

    def config_docking(self) -> hello_imgui.DockingParams:
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

        w_3dplot = hello_imgui.DockableWindow(
            label_ = 'Point cloud',
            dock_space_name_ = 'BottomSpace',
            gui_function_ = self.gui_3d_plot,
            is_visible_ = True,
            can_be_closed_= False
            )
        
        w_heatmap = hello_imgui.DockableWindow(
            label_ = 'Raw Data',
            dock_space_name_ = 'MainDockSpace',
            gui_function_ = self.gui_heat_map,
            is_visible_ = True,
            can_be_closed_ = False
        )

        w_settings = hello_imgui.DockableWindow(
            label_ = 'Settings',
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
        imgui.separator_text('Sensor Orientation')
        changed, self.view_angle = imgui.slider_float("View Angle", self.view_angle, 0.0, 120.0)
        changed, self.sensor_pitch = imgui.slider_float("Sensor Pitch", self.sensor_pitch, -360, 360.0)

        imgui.separator_text('3D Plot')
        changed, self.plot_box_elvation = imgui.slider_float('Elevation', self.plot_box_elvation, -90.0, 90.0)
        changed, self.plot_box_azimuth = imgui.slider_float('Azimuth', self.plot_box_azimuth, -360.0, 360.0)


    def gui_heat_map(self):
        if not self.sensor_ready:
            return
        
        if self.raw_data is not None:
            if implot.begin_plot("Raw Data"):
                implot.setup_legend(implot.Location_.east, implot.LegendFlags_.outside)
                implot.plot_heatmap(
                    "Raw Data",
                    self.raw_data,
                    0.0,
                    2500.0,
                    bounds_min = (0, 0),
                    # bounds_max = (
                    #     self.tof_sensor.last_raw_data.width,
                    #     self.tof_sensor.last_raw_data.height),
                    bounds_max = (
                        1.0,
                        1.0)
                )
            implot.end_plot()
    
    def gui_3d_plot(self):
        if implot3d.begin_plot("Point Cloud"):
            implot3d.setup_legend(
                implot3d.Location_.east,
                implot3d.LegendFlags_.horizontal
            )
            implot3d.setup_box_rotation(
                # math.radians(self.plot_box_elvation),
                # math.radians(self.plot_box_azimuth),
                self.plot_box_elvation,
                self.plot_box_azimuth,
                True,
                implot3d.Cond_.always
                )
            implot3d.setup_axes("X", "Y", "Z")
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
        print(f"{len(self.raw_data)} data saved to {filename}")
        with open(filename, "w") as f:
            for y in range(self.raw_data.shape[0]):
                for x in range(self.raw_data.shape[1]):
                    f.write(f'{self.raw_data[y, x]:0.3f} ')
                f.write('\n')