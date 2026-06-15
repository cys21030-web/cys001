import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import enum
import logging
from imgui_bundle import imgui, hello_imgui, implot, implot3d
from collector_app import CollectorApp
from training_app import TrainingApp
from inference_app import InferenceApp

class AppMode(enum.Enum):
    COLLECTOR = "Collector"
    TRAINING = "Training"
    INFERENCE = "Inference"

class MainApp:
    def __init__(self):
        logging.info("Initializing MainApp")
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

        self.current_app = None
        self.app_mode = None

    def load_custom_fonts(self):
        io = imgui.get_io()
        font_path = "LXGWWenKaiMonoTC-Regular.ttf"
        font_size = 18.0
        try:
            io.fonts.add_font_from_file_ttf(font_path, font_size, None)
        except Exception:
            print(f"Warning: Could not load font from {font_path}")

    def gui_menu(self):
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Mode", True):
                clicked_collector, _ = imgui.menu_item(
                    AppMode.COLLECTOR.value, "", self.app_mode == AppMode.COLLECTOR
                )
                if clicked_collector:
                    if self.app_mode != AppMode.COLLECTOR:
                        self.app_mode = AppMode.COLLECTOR
                        self.current_app = CollectorApp()

                clicked_training, _ = imgui.menu_item(
                    AppMode.TRAINING.value, "", self.app_mode == AppMode.TRAINING
                )
                if clicked_training:
                    if self.app_mode != AppMode.TRAINING:
                        self.app_mode = AppMode.TRAINING
                        self.current_app = TrainingApp()

                clicked_inference, _ = imgui.menu_item(
                    AppMode.INFERENCE.value, "", self.app_mode == AppMode.INFERENCE
                )
                if clicked_inference:
                    if self.app_mode != AppMode.INFERENCE:
                        self.app_mode = AppMode.INFERENCE
                        self.current_app = InferenceApp()
                
                imgui.end_menu()
            imgui.end_main_menu_bar()

    def gui_settings(self):
        if self.current_app:
            self.current_app.gui_settings()

    def gui_heat_map(self):
        if self.current_app:
            self.current_app.gui_heat_map()
            
    def gui_3d_plot(self):
        if self.current_app:
            self.current_app.gui_3d_plot()

    def gui_training_metrics(self):
        if self.current_app and hasattr(self.current_app, "gui_training_metrics"):
            self.current_app.gui_training_metrics()
        else:
            imgui.text("Training metrics are only available in Training Mode.")

    def config_docking(self):
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
            gui_function_ = self.gui_3d_plot
        )
        
        w_heatmap = hello_imgui.DockableWindow(
            label_ = 'Raw Data',
            dock_space_name_ = 'MainDockSpace',
            gui_function_ = self.gui_heat_map
        )

        w_settings = hello_imgui.DockableWindow(
            label_ = 'Settings',
            dock_space_name_ = 'LeftSpace',
            gui_function_ = self.gui_settings
        )

        w_training = hello_imgui.DockableWindow(
            label_ = 'Training Metrics',
            dock_space_name_ = 'MainDockSpace',
            gui_function_ = self.gui_training_metrics
        )

        docking_params.dockable_windows = [
            w_heatmap,
            w_3dplot,
            w_settings,
            w_training
        ]
        return docking_params

    def gui(self):
        self.gui_menu()
        if self.current_app:
            self.current_app.gui()

    def run(self):
        runner_params = hello_imgui.RunnerParams()

        runner_params.callbacks.show_gui = self.gui
        runner_params.app_window_params.window_title = "Lidar App"
        runner_params.app_window_params.window_geometry.size = (1280, 960)
        runner_params.app_window_params.restore_previous_geometry = True
        runner_params.callbacks.load_additional_fonts = self.load_custom_fonts
        
        runner_params.imgui_window_params.default_imgui_window_type = (
            hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
        )
        
        runner_params.docking_params = self.config_docking()
        
        runner_params.fps_idling.fps_max = 20.0
        runner_params.fps_idling.vsync_to_monitor = True

        hello_imgui.run(runner_params)

def main():
    app = MainApp()
    app.run()

if __name__ == "__main__":
    main()
