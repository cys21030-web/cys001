# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - depends on runtime environment
    torch = None
    F = None

from imgui_bundle import imgui, hello_imgui, implot, implot3d

from ai.ToFDataLabel import ToFDataLabel
from common.ToFData import ToFData
from common.ToFSensor import ToFSensor
from common.ViewAngle import ViewAngle
from common.WorldCoord import WorldCoord

from AppBase import App as AppBase


class InferenceApp(AppBase):
    """即時推論模式。

    此模式會載入已訓練好的模型，針對最新感測資料做前向推論，
    並在介面上顯示每個類別的機率與判定結果。
    """

    def __init__(self):
        super().__init__()
        
        self.__win_title = "實時推論監察儀"
        
        # 初始化 ToF 傳感器驅動
        self.tof_sensor = ToFSensor()
        self.last_disp_frame = 0
        
        # 實時推論變量
        self.models_dir = Path('./models')
        self.model_files = []
        self.selected_model_index = -1
        self.model = None
        self.device = "cpu"
        if self.torch_available() and torch.cuda.is_available():
            self.device = "cuda"
        
        self.probabilities = None
        self.predicted_label_index = -1
        self.model_error = None
        
        # 自動掃描本地已訓練模型
        self.scan_models()

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
        self.run_inference()

    def update_data(self):
        # 獲取最新影格並進行降噪與點雲映射
        if self.tof_sensor.last_raw_data is None:
            return

        if self.last_disp_frame >= self.sensor_frame_cnt_total:
            return
        
        self.raw_data = self.raw_data * 0.5 + self.tof_sensor.last_raw_data.repaired_data * 0.5
        self.cloud_points = WorldCoord(self.raw_data, self.tof_sensor.view_angle)

    def scan_models(self):
        # 掃描 models 目錄下的所有 .pth 檔案 (包括子目錄)
        if self.models_dir.exists():
            self.model_files = sorted(list(self.models_dir.rglob("*.pth")), key=lambda p: p.name, reverse=True)
            if self.model_files and self.selected_model_index == -1:
                self.selected_model_index = 0
                self.load_selected_model()

    def torch_available(self):
        return torch is not None and F is not None

    def load_selected_model(self):
        # 載入指定的 PyTorch 模型權重
        if 0 <= self.selected_model_index < len(self.model_files):
            model_path = self.model_files[self.selected_model_index]
            self.model_error = None
            if not self.torch_available():
                self.model_error = "目前環境中沒有可用的 PyTorch，無法載入模型。"
                self.model = None
                return
            try:
                # 載入權重並初始化分類網路
                checkpoint = torch.load(model_path, map_location=self.device)
                from ai.ToFTrainer import ToFClassifierModel
                self.model = ToFClassifierModel.from_config(checkpoint["model_kwargs"])
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model.to(self.device)
                self.model.eval()
                logging.info(f"模型載入成功：{model_path.name}")
            except Exception as e:
                logging.exception(f"模型載入失敗：{e}")
                self.model = None
                self.model_error = str(e)

    def run_inference(self):
        # 執行實時神經網路前向傳播與 softmax 置信度計算
        if self.model is None or self.raw_data is None or not self.torch_available():
            return
        
        try:
            # 數據正規化與展平 preprocessing
            tensor = torch.tensor(self.raw_data, dtype=torch.float32)
            normalized = tensor.clamp(min=ToFData.min_valid, max=ToFData.max_valid) / ToFData.max_valid
            
            # 展平為 1D 向量
            features = normalized.view(1, -1).to(self.device)
            
            with torch.no_grad():
                logits = self.model(features)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
            self.probabilities = probs
            self.predicted_label_index = int(np.argmax(probs))
        except Exception as e:
            logging.exception(f"即時推論執行失敗：{e}")

    def get_missing_model_message(self):
        if self.model_error:
            return self.model_error
        if not self.torch_available():
            return "目前環境中沒有可用的 PyTorch，請先安裝 PyTorch 後再啟動推論。"
        return "目前沒有可用的模型檔案，請提供已預先訓練好的模型後再啟動推論。"

    def get_prediction_summary(self):
        if self.model is None:
            return "尚未載入模型"
        if self.probabilities is None:
            return "正在等待資料"
        if not 0 <= self.predicted_label_index < len(ToFDataLabel.labels):
            return "尚未得到結果"

        label_name = ToFDataLabel.labels[self.predicted_label_index].name
        confidence = self.probabilities[self.predicted_label_index]
        return f"{label_name}（{confidence * 100:.0f}%）"

    def gui_settings(self):
        super().gui_settings_plot_view()
        
        # 傳感器狀態面版
        imgui.separator_text("感測器狀態")
        if not self.sensor_ready:
            imgui.text("ToF 感測器尚未就緒。")
        else:
            imgui.label_text('狀態', self.tof_sensor.status)
            imgui.label_text('影格', f'{self.sensor_frame_cnt_valid:04d} / {self.sensor_frame_cnt_total:04d}')
            
        # 選擇模型下拉式選單
        imgui.separator_text("模型選擇")
        imgui.text_wrapped("這裡會顯示目前最像哪一種情況。若還沒有模型，先收集資料再訓練一次即可。")
        self.scan_models() # 動態重新掃描
        
        if not self.model_files:
            imgui.text("./models/ 目錄中找不到任何模型。")
            imgui.text_wrapped(self.get_missing_model_message())
        else:
            combo_labels = [str(f.relative_to(self.models_dir)) for f in self.model_files]
            changed, new_index = imgui.combo("模型", self.selected_model_index, combo_labels)
            if changed:
                self.selected_model_index = new_index
                self.load_selected_model()
                
        # 實時推論監察面板
        imgui.separator_text("即時推論面板")
        if self.model is None:
            imgui.text("目前尚未載入分類模型。")
        elif self.probabilities is None:
            imgui.text("正在等待即時感測器資料…")
        else:
            imgui.text(f"目前判斷：{self.get_prediction_summary()}")
            imgui.spacing()
            imgui.separator()
            imgui.spacing()

            for label in ToFDataLabel.labels:
                probability = float(self.probabilities[label.index])
                imgui.text(label.name)
                imgui.same_line()
                imgui.progress_bar(probability, (0.0, 0.0), f"{probability * 100:0.0f}%")
                imgui.spacing()
