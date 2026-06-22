# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
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
        self.selected_model_idx = -1
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.probabilities = None
        self.predicted_label_idx = -1
        
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
            if self.model_files and self.selected_model_idx == -1:
                self.selected_model_idx = 0
                self.load_selected_model()

    def load_selected_model(self):
        # 載入指定的 PyTorch 模型權重
        if 0 <= self.selected_model_idx < len(self.model_files):
            model_path = self.model_files[self.selected_model_idx]
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

    def run_inference(self):
        # 執行實時神經網路前向傳播與 softmax 置信度計算
        if self.model is None or self.raw_data is None:
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
            self.predicted_label_idx = int(np.argmax(probs))
        except Exception as e:
            logging.exception(f"即時推論執行失敗：{e}")

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
        self.scan_models() # 動態重新掃描
        
        if not self.model_files:
            imgui.text("./models/ 目錄中找不到任何模型。")
            imgui.text("請先執行 tui_train.py 進行訓練。")
        else:
            combo_labels = [str(f.relative_to(self.models_dir)) for f in self.model_files]
            changed, new_idx = imgui.combo("模型", self.selected_model_idx, combo_labels)
            if changed:
                self.selected_model_idx = new_idx
                self.load_selected_model()
                
        # 實時推論監察面板
        imgui.separator_text("即時推論面板")
        if self.model is None:
            imgui.text("目前尚未載入分類模型。")
        elif self.probabilities is None:
            imgui.text("正在等待即時感測器資料…")
        else:
            # 配對標籤名稱、置信度與索引，以便事後排序
            labeled_probs = [
                (lbl.name, self.probabilities[lbl.index], lbl.index)
                for lbl in ToFDataLabel.labels
            ]
            # 依據置信度分數進行「降序」排序
            labeled_probs.sort(key=lambda x: x[1], reverse=True)
            
            # 使用動態臨界值公式 (1.5 * 1 / N)，當 N=3 時為 50% 臨界值
            num_classes = len(ToFDataLabel.labels)
            threshold = 1.5 * (1.0 / num_classes) if num_classes > 0 else 0.50
            highest_prob = labeled_probs[0][1]
            has_obvious_result = highest_prob >= threshold
            
            pred_label = ToFDataLabel.labels[self.predicted_label_idx].name
            imgui.text_disabled("偵測狀態：")
            imgui.same_line()
            
            # 如果有明顯結果，突顯當前預測狀態
            if has_obvious_result:
                if self.predicted_label_idx == 0:
                    imgui.text_colored((0.12, 0.53, 0.22, 1.0), f"{pred_label}（良好）")
                elif self.predicted_label_idx == 3:
                    imgui.text_colored((0.3, 0.5, 0.35, 1.0), f"{pred_label}（無法判定）")
                else:
                    imgui.text_colored((0.0, 0.5, 0.35, 1.0), f"{pred_label}（注意）")
            else:
                imgui.text_colored((0.6, 0.6, 0.6, 1.0), "不確定 / 無明顯結果")
                
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            # 渲染已降序排序的置信度條，並依據置信度是否「明顯」套用綠色/灰色樣式
            for rank, (name, prob, idx) in enumerate(labeled_probs):
                if has_obvious_result:
                    if rank == 0:
                        text_color = (0.12, 0.53, 0.22, 1.0) # 第一名顯示為深綠色，提升與白字的對比度
                        bar_color = (0.12, 0.53, 0.22, 1.0)   # 第一名置信度條顯示為深綠色，使內部白字清晰可見
                    else:
                        text_color = (0.6, 0.6, 0.6, 1.0) # 其他顯示為灰色
                        bar_color = (0.6, 0.6, 0.6, 1.0)   # 其他置信度條顯示為灰色
                else:
                    text_color = (0.6, 0.6, 0.6, 1.0)     # 若無明顯結果，全數顯示為灰色
                    bar_color = (0.6, 0.6, 0.6, 1.0)       # 若無明顯結果，全數置信度條顯示為灰色
                
                imgui.text_colored(text_color, f"{name:<12}")
                imgui.same_line()
                
                imgui.push_style_color(imgui.Col_.plot_histogram, bar_color)
                imgui.progress_bar(prob, (0.0, 0.0), f"{int(prob * 100)}%")
                imgui.pop_style_color()
