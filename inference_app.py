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
    def __init__(self):
        super().__init__()
        
        self.__win_title = "實時推論監察儀"
        
        # Initialize ToF Sensor
        self.tof_sensor = ToFSensor()
        self.last_disp_frame = 0
        
        # Inference variables
        self.models_dir = Path('./models')
        self.model_files = []
        self.selected_model_idx = -1
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.probabilities = None
        self.predicted_label_idx = -1
        
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
        if self.tof_sensor.last_raw_data is None:
            return

        if self.last_disp_frame >= self.sensor_frame_cnt_total:
            return
        
        self.raw_data = self.raw_data * 0.5 + self.tof_sensor.last_raw_data.repaired_data * 0.5
        self.cloud_points = WorldCoord(self.raw_data, self.tof_sensor.view_angle)

    def scan_models(self):
        if self.models_dir.exists():
            self.model_files = sorted(list(self.models_dir.glob("*.pth")), reverse=True)
            if self.model_files and self.selected_model_idx == -1:
                self.selected_model_idx = 0
                self.load_selected_model()

    def load_selected_model(self):
        if 0 <= self.selected_model_idx < len(self.model_files):
            model_path = self.model_files[self.selected_model_idx]
            try:
                # Load PyTorch checkpoint safely
                checkpoint = torch.load(model_path, map_location=self.device)
                from ai.ToFTrainer import ToFClassifierModel
                self.model = ToFClassifierModel.from_config(checkpoint["model_kwargs"])
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model.to(self.device)
                self.model.eval()
                logging.info(f"Model loaded successfully: {model_path.name}")
            except Exception as e:
                logging.exception(f"Error loading model: {e}")
                self.model = None

    def run_inference(self):
        if self.model is None or self.raw_data is None:
            return
        
        try:
            # Standardize and normalize raw_data (shape: 8x8) exactly like training sample
            tensor = torch.tensor(self.raw_data, dtype=torch.float32)
            normalized = tensor.clamp(min=ToFData.min_valid, max=ToFData.max_valid) / ToFData.max_valid
            
            # Flatten to 1D (shape: 64)
            features = normalized.view(1, -1).to(self.device)
            
            with torch.no_grad():
                logits = self.model(features)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
            self.probabilities = probs
            self.predicted_label_idx = int(np.argmax(probs))
        except Exception as e:
            logging.exception(f"Error running inference: {e}")

    def gui_settings(self):
        super().gui_settings_plot_view()
        
        # Sensor Status Panel
        imgui.separator_text("Sensor Status")
        if not self.sensor_ready:
            imgui.text("ToF Sensor is not ready yet.")
        else:
            imgui.label_text('Status', self.tof_sensor.status)
            imgui.label_text('Frames', f'{self.sensor_frame_cnt_valid:04d} / {self.sensor_frame_cnt_total:04d}')
            
        # Model Selection Panel
        imgui.separator_text("Model Selection")
        self.scan_models() # Rescan to pick up newly trained models
        
        if not self.model_files:
            imgui.text("No models found inside `./models/`.")
            imgui.text("Please run `tui_train.py` first.")
        else:
            combo_labels = [f.name for f in self.model_files]
            changed, new_idx = imgui.combo("Model", self.selected_model_idx, combo_labels)
            if changed:
                self.selected_model_idx = new_idx
                self.load_selected_model()
                
        # Real-time Inference Panel
        imgui.separator_text("Inference Panel")
        if self.model is None:
            imgui.text("No classifier model currently loaded.")
        elif self.probabilities is None:
            imgui.text("Awaiting live sensor stream...")
        else:
            pred_label = ToFDataLabel.labels[self.predicted_label_idx].name
            imgui.text_disabled("Detected State:")
            imgui.same_line()
            
            # Highlight state with clear indicators
            if self.predicted_label_idx == 0:
                imgui.text_colored((0.0, 1.0, 0.0, 1.0), f"{pred_label} (Normal)")
            elif self.predicted_label_idx == 1:
                imgui.text_colored((1.0, 0.8, 0.0, 1.0), f"{pred_label} (Upstairs)")
            else:
                imgui.text_colored((1.0, 0.4, 0.0, 1.0), f"{pred_label} (Downstairs)")
                
            imgui.spacing()
            imgui.separator()
            imgui.spacing()
            
            # Display confidence breakdown
            for lbl in ToFDataLabel.labels:
                prob = self.probabilities[lbl.index]
                imgui.text(f"{lbl.name:<12}")
                imgui.same_line()
                imgui.progress_bar(prob, (0.0, 0.0), f"{int(prob * 100)}%")
