# -*- coding: utf-8 -*-
import os

# 設置環境變量，限制 OpenMP 與 MKL 僅使用單線程，避免多線程庫在樹莓派4B上引起段錯誤
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

import logging
import random
import yaml
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np

import torch
from imgui_bundle import imgui, hello_imgui, implot, implot3d

from ai.ToFDataLabel import ToFDataLabel
from common.ToFData import ToFData
from common.ViewAngle import ViewAngle
from common.WorldCoord import WorldCoord

from AppBase import App as AppBase
from training import ToFSample, ToFDataset, ToFDataLoader


class TrainingApp(AppBase):
    def __init__(self):
        super().__init__()

        self.__win_title = '數據檢視器'
        self.selected_item = None

        self.samples = {}
        
        # 訓練狀態與指標變量
        self.is_training = False
        self.train_test_split_perc = 50.0
        self.current_epoch = 0
        self.epochs = 30
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.layer_weights = {
            "Layer 1": [],
            "Layer 2": [],
            "Layer 3": []
        }
        self.final_w1 = None
        self.final_w2 = None
        self.final_w3 = None

        # 持久化繪圖陣列（用以在記憶體中保持 C++ 數據指標有效，避免被 Python GC 釋放而引發段錯誤）
        self.plot_epochs_indices = np.array([], dtype=np.float32)
        self.plot_train_losses = np.array([], dtype=np.float32)
        self.plot_test_losses = np.array([], dtype=np.float32)
        self.plot_train_accuracies = np.array([], dtype=np.float32)
        self.plot_test_accuracies = np.array([], dtype=np.float32)
        self.plot_layer1 = np.array([], dtype=np.float32)
        self.plot_layer2 = np.array([], dtype=np.float32)
        self.plot_layer3 = np.array([], dtype=np.float32)

        # 啟動時在背景守護線程中自動掃描本地已收集數據，防止視窗載入卡頓
        import threading
        threading.Thread(
            target=self.data_pre_scan,
            daemon=True
        ).start()

    def gui(self):
        pass

    def gui_settings(self):
        super().gui_settings()
            
        imgui.separator_text('Loaded')

        # 渲染樹狀結構，列出當前本地載入的所有數據文件，點擊可即時預覽 heatmap 與 3D 雲點圖
        if imgui.tree_node_ex("Data Files", imgui.TreeNodeFlags_.default_open | imgui.TreeNodeFlags_.open_on_arrow | imgui.TreeNodeFlags_.open_on_double_click):
            for lbl in ToFDataLabel.labels:
                lbl_name = lbl.name
                cnt = 0
                if lbl_name in self.samples.keys():
                    cnt = len(self.samples[lbl_name])
                    if imgui.tree_node_ex(f'{lbl_name} [{cnt}]'):
                        
                        for sample in self.samples[lbl_name]:
                            dat_flag = imgui.TreeNodeFlags_.leaf | imgui.TreeNodeFlags_.no_tree_push_on_open
                            if sample == self.selected_item:
                                dat_flag |= imgui.TreeNodeFlags_.selected
                            imgui.tree_node_ex(sample.path.name, dat_flag)

                            if imgui.is_item_clicked():
                                self.selected_item = sample
                                self.raw_data = sample.data.repaired_data
                                self.cloud_points = sample.points

                        imgui.tree_pop()
            imgui.tree_pop()

        # 訓練控制面版（包含 Train Ratio 滑桿、與啟動按鈕）
        imgui.separator_text('Training Control')
        if self.is_training:
            imgui.begin_disabled()
            imgui.slider_float('Train Split (%)', self.train_test_split_perc, 10.0, 90.0, "%.1f")
            imgui.button('Training In Progress...')
            imgui.end_disabled()
        else:
            changed, self.train_test_split_perc = imgui.slider_float('Train Split (%)', self.train_test_split_perc, 10.0, 90.0, "%.1f")
            
            # 若當前未載入任何數據，禁用訓練按鈕
            total_samples_loaded = sum(len(deque) for deque in self.samples.values()) if self.samples else 0
            if total_samples_loaded == 0:
                imgui.begin_disabled()
                imgui.button('No Loaded Samples')
                imgui.end_disabled()
            else:
                if imgui.button('Start Training'):
                    self.is_training = True
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # 收集所有的採集數據
                    all_samples = []
                    for lbl_name, sample_deque in self.samples.items():
                        all_samples.extend(list(sample_deque))
                        
                    # 隨機打亂並依據 split 比例進行 Train/Test 分割
                    samples_by_label = {}
                    for s in all_samples:
                        samples_by_label.setdefault(s.label, []).append(s)
                        
                    train_samples = []
                    test_samples = []
                    split_perc = self.train_test_split_perc / 100.0
                    
                    for label, samples_list in samples_by_label.items():
                        shuffled = list(samples_list)
                        random.shuffle(shuffled)
                        split_idx = max(1, int(len(shuffled) * split_perc)) # 確保各類別至少有一個樣本供訓練
                        train_samples.extend(shuffled[:split_idx])
                        test_samples.extend(shuffled[split_idx:])
                        
                    if not train_samples and all_samples:
                        train_samples.append(all_samples[0])
                        test_samples = all_samples[1:]
                        
                    # 在主線程中執行同步模型訓練（單線程極簡穩定版，防崩潰）
                    self.run_training_loop(train_samples, test_samples, timestamp)

    def run_training_loop(self, train_samples, test_samples, timestamp):
        from torch import nn
        from torch.utils.data import DataLoader
        from training.dataset import ToFDataset
        from ai.ToFTrainer import ToFClassifierModel
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = ToFClassifierModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        train_dataset = ToFDataset(train_samples)
        test_dataset = ToFDataset(test_samples)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.layer_weights = {
            "Layer 1": [],
            "Layer 2": [],
            "Layer 3": []
        }
        
        self.current_epoch = 0
        
        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            correct = 0
            
            for features, labels in train_loader:
                # 將 2D [8, 8] 輸入展平成 1D [64] 傳入分類模型
                features = features.to(device).view(features.size(0), -1)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * features.size(0)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                
            train_loss = epoch_loss / len(train_dataset) if len(train_dataset) > 0 else 0.0
            train_acc = correct / len(train_dataset) if len(train_dataset) > 0 else 0.0
            
            # 在測試集上計算交叉熵與分類準確率
            model.eval()
            test_loss = 0.0
            test_correct = 0
            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(device).view(features.size(0), -1)
                    labels = labels.to(device)
                    logits = model(features)
                    loss = criterion(logits, labels)
                    test_loss += loss.item() * features.size(0)
                    
                    predictions = torch.argmax(logits, dim=1)
                    test_correct += (predictions == labels).sum().item()
            
            val_loss = test_loss / len(test_dataset) if len(test_dataset) > 0 else 0.0
            val_acc = test_correct / len(test_dataset) if len(test_dataset) > 0 else 0.0
            
            self.train_losses.append(train_loss)
            self.test_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(val_acc)
            
            # 提取並追蹤神經網路各層參數權重的平均值變化
            with torch.no_grad():
                w1 = model.network[0].weight.cpu().numpy()
                w2 = model.network[3].weight.cpu().numpy()
                w3 = model.network[6].weight.cpu().numpy()
                
                self.layer_weights["Layer 1"].append(float(w1.mean()))
                self.layer_weights["Layer 2"].append(float(w2.mean()))
                self.layer_weights["Layer 3"].append(float(w3.mean()))
                
                self.final_w1 = w1
                self.final_w2 = w2
                self.final_w3 = w3
                
            self.current_epoch = epoch
            
        # 轉換指標數值為持久化 NumPy 浮點陣列，鎖定 C++ 指標記憶體，完美杜絕記憶體 GC 導致的段錯誤
        self.plot_epochs_indices = np.array(range(1, len(self.train_losses) + 1), dtype=np.float32)
        self.plot_train_losses = np.array(self.train_losses, dtype=np.float32)
        self.plot_test_losses = np.array(self.test_losses, dtype=np.float32)
        self.plot_train_accuracies = np.array(self.train_accuracies, dtype=np.float32)
        self.plot_test_accuracies = np.array(self.test_accuracies, dtype=np.float32)
        
        self.plot_layer1 = np.array(self.layer_weights["Layer 1"], dtype=np.float32)
        self.plot_layer2 = np.array(self.layer_weights["Layer 2"], dtype=np.float32)
        self.plot_layer3 = np.array(self.layer_weights["Layer 3"], dtype=np.float32)

        # 儲存 PyTorch 權重檔案與 YAML 分割數據歷史紀錄
        models_dir = Path('./models')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # 儲存 .pth 權重結構
        model_path = models_dir / f"model_{timestamp}.pth"
        payload = {
            "model_kwargs": model.config(),
            "state_dict": model.state_dict(),
        }
        torch.save(payload, model_path)
        
        # 儲存 YAML 分割元數據與檔案紀錄，用以事後追溯
        yaml_path = models_dir / f"model_{timestamp}.yaml"
        yaml_data = {
            "timestamp": timestamp,
            "train_ratio": self.train_test_split_perc,
            "train_set_size": len(train_samples),
            "test_set_size": len(test_samples),
            "final_train_loss": float(self.train_losses[-1]),
            "final_test_loss": float(self.test_losses[-1]),
            "final_train_accuracy": float(self.train_accuracies[-1]),
            "final_test_accuracy": float(self.test_accuracies[-1]),
            "train_files": [str(s.path.resolve().relative_to(Path('.').resolve())) for s in train_samples if s.path],
            "test_files": [str(s.path.resolve().relative_to(Path('.').resolve())) for s in test_samples if s.path],
        }
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
            
        logging.info(f"Model and YAML metadata saved successfully.")
        self.is_training = False

    def gui_training_metrics(self):
        # 繪製訓練進度條與狀態
        if self.current_epoch > 0:
            progress = self.current_epoch / self.epochs if self.epochs > 0 else 0.0
            imgui.text(f"Status: Training Complete (30 / 30 Epochs)")
            imgui.progress_bar(progress, (0.0, 0.0), f"{int(progress * 100)}%")
            imgui.spacing()
        else:
            imgui.text("Click 'Start Training' in Settings to train a model.")
            return

        # 切換分頁，繪製損失函數曲線、分類準確率、各層權重變化與混淆權重矩陣
        if imgui.begin_tab_bar("TrainingTabBar"):
            selected_metrics, _ = imgui.begin_tab_item("Metrics Plots")
            if selected_metrics:
                if len(self.plot_train_losses) > 0:
                    if implot.begin_plot("Loss Rate"):
                        implot.setup_axes("Epoch", "Loss")
                        implot.plot_line("Train Loss", self.plot_epochs_indices, self.plot_train_losses)
                        if len(self.plot_test_losses) > 0:
                            implot.plot_line("Test Loss", self.plot_epochs_indices, self.plot_test_losses)
                        implot.end_plot()
                        
                    if implot.begin_plot("Accuracy Rate"):
                        implot.setup_axes("Epoch", "Accuracy")
                        implot.plot_line("Train Acc", self.plot_epochs_indices, self.plot_train_accuracies)
                        if len(self.plot_test_accuracies) > 0:
                            implot.plot_line("Test Acc", self.plot_epochs_indices, self.plot_test_accuracies)
                        implot.end_plot()
                else:
                    imgui.text("Awaiting metrics...")
                imgui.end_tab_item()

            selected_weights, _ = imgui.begin_tab_item("Layer Weights (Epoch Mean)")
            if selected_weights:
                if len(self.plot_layer1) > 0:
                    if implot.begin_plot("Layer Parameter Weights"):
                        implot.setup_axes("Epoch", "Mean Parameter Weight")
                        implot.plot_line("Layer 1 (64->128)", self.plot_epochs_indices, self.plot_layer1)
                        implot.plot_line("Layer 2 (128->64)", self.plot_epochs_indices, self.plot_layer2)
                        implot.plot_line("Layer 3 (64->3)", self.plot_epochs_indices, self.plot_layer3)
                        implot.end_plot()
                else:
                    imgui.text("Awaiting weight data...")
                imgui.end_tab_item()

            selected_matrices, _ = imgui.begin_tab_item("Weight Matrices")
            if selected_matrices:
                if self.final_w3 is not None:
                    imgui.text("Final Dense Layer Weight Matrix (3 Classes x 64 Features)")
                    if implot.begin_plot("Final Layer Weights Heatmap"):
                        implot.setup_legend(implot.Location_.east, implot.LegendFlags_.outside)
                        implot.plot_heatmap(
                            "Weights",
                            self.final_w3,
                            -0.5,
                            0.5,
                            bounds_min=(0.0, 0.0),
                            bounds_max=(1.0, 1.0)
                        )
                        implot.end_plot()
                else:
                    imgui.text("Weight matrix visualization will load when training finishes.")
                imgui.end_tab_item()

            imgui.end_tab_bar()

    def gui_heatmap(self):
        pass

    def data_pre_scan(self):
        # 自動掃描本地已採集的歷史數據包 (.dat) 並載入佇列中
        for lbl in ToFDataLabel.labels:
            dir = self.snapshot_dir / lbl.name
            if not dir.exists():
                continue

            self.samples[lbl.name] = deque()

            for root, dir_names, file_names in dir.walk(True):
                for file_name in file_names:
                    file = root / file_name
                    if file.suffix != '.dat':
                        continue

                    self.data_scan_one(lbl, file)

    def data_scan_one(self, lbl: ToFDataLabel.Label, file: Path):
        try:
            self.samples[lbl.name].append(
                ToFSample.from_data_file(lbl.index, file)
            )
        except Exception as exc:
            logging.exception(exc)
