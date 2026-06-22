# -*- coding: utf-8 -*-
import os

# 設置環境變量，限制 OpenMP 與 MKL 僅使用單線程，避免其與樹莓派 4B 的圖形驅動（Mesa/VC4）產生衝突而引發段錯誤 (Segmentation Fault)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import enum
import logging
from imgui_bundle import imgui, hello_imgui, implot, implot3d
from collector_app import CollectorApp
from inference_app import InferenceApp

# 定義應用程式支援的運作模式 (移除了不穩定的 GUI 訓練模式，改為純 TUI 終端訓練)
class AppMode(enum.Enum):
    COLLECTOR = "數據採集"
    INFERENCE = "實時數據分析"

class MainApp:
    def __init__(self):
        logging.info("初始化主應用程式")
        
        # 初始化 ImGui、ImPlot 與 ImPlot3D 上下文，確保繪圖組件正常運作
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
        # 載入自定義的中文字型，確保繁體中文介面正常顯示，不出現亂碼
        io = imgui.get_io()
        font_path = "LXGWWenKaiMonoTC-Regular.ttf"
        font_size = 18.0
        try:
            io.fonts.add_font_from_file_ttf(font_path, font_size, None)
        except Exception:
            print(f"警告：無法載入字型 {font_path}")

    def gui_menu(self):
        # 繪製主選單列，供使用者在數據採集器與實時推論監察模式之間自由切換
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("運作模式 (Mode)", True):
                # 數據採集模式切換
                clicked_collector, _ = imgui.menu_item(
                    AppMode.COLLECTOR.value, "", self.app_mode == AppMode.COLLECTOR
                )
                if clicked_collector:
                    if self.app_mode != AppMode.COLLECTOR:
                        self.app_mode = AppMode.COLLECTOR
                        self.current_app = CollectorApp()

                # 實時推論監察模式切換
                clicked_inference, _ = imgui.menu_item(
                    AppMode.INFERENCE.value, "", self.app_mode == AppMode.INFERENCE
                )
                if clicked_inference:
                    if self.app_mode != AppMode.INFERENCE:
                        self.app_mode = AppMode.INFERENCE
                        self.current_app = InferenceApp()
                
                imgui.end_menu()
            imgui.end_main_menu_bar()

    # 以下方法將視窗渲染邏輯委託給目前正啟用的子模式應用程式 (Delegation Pattern)
    def gui_settings(self):
        if self.current_app:
            self.current_app.gui_settings()

    def gui_heat_map(self):
        if self.current_app:
            self.current_app.gui_heat_map()
            
    def gui_3d_plot(self):
        if self.current_app:
            self.current_app.gui_3d_plot()

    def config_docking(self):
        # 設定 ImGui 的 Docking 佈局（左側設置欄、中央熱力圖、下方3D雲點圖）
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

        # 註冊點雲圖、熱力圖與參數設定視窗（轉換標題為繁體中文）
        w_3dplot = hello_imgui.DockableWindow(
            label_ = '三維空間點雲圖',
            dock_space_name_ = 'BottomSpace',
            gui_function_ = self.gui_3d_plot
        )
        
        w_heatmap = hello_imgui.DockableWindow(
            label_ = '距離熱力圖',
            dock_space_name_ = 'MainDockSpace',
            gui_function_ = self.gui_heat_map
        )

        w_settings = hello_imgui.DockableWindow(
            label_ = '參數控制',
            dock_space_name_ = 'LeftSpace',
            gui_function_ = self.gui_settings
        )

        docking_params.dockable_windows = [
            w_heatmap,
            w_3dplot,
            w_settings
        ]
        return docking_params

    def gui(self):
        self.gui_menu()
        if self.current_app:
            self.current_app.gui()

    def run(self):
        # 配置與啟動 hello_imgui 渲染引擎
        runner_params = hello_imgui.RunnerParams()

        runner_params.callbacks.show_gui = self.gui
        runner_params.app_window_params.window_title = "升降機平層誤差監察系統"
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
    try:
        main()
    except Exception as e:
        import traceback
        with open(".error.log", "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        raise e
