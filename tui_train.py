#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import yaml
import logging
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from ai.ToFDataLabel import ToFDataLabel
from common.ToFData import ToFData
from training.sample import ToFSample
from training.dataset import ToFDataset
from ai.ToFTrainer import ToFClassifierModel

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_header():
    print("=" * 70)
    print("      ToF Lidar Classifier - Terminal Training System (TUI)")
    print("=" * 70)

def scan_datasets(snapshot_dir: Path):
    # 掃描本地 snapshot/ 資料夾下的所有影格樣本
    print("[*] 正在掃描本地影格數據...")
    samples = {}
    total_count = 0
    
    for lbl in ToFDataLabel.labels:
        dir_path = snapshot_dir / lbl.name
        if not dir_path.exists():
            samples[lbl.name] = []
            continue
            
        samples[lbl.name] = []
        for file in dir_path.glob("*.dat"):
            try:
                sample = ToFSample.from_data_file(lbl.index, file)
                samples[lbl.name].append(sample)
                total_count += 1
            except Exception as e:
                pass
                
        print(f"  - 標籤 {lbl.name:<10} [索引 {lbl.index}]: 成功載入 {len(samples[lbl.name]):>3} 份文件")
        
    print(f"[*] 數據集載入完成，共計 {total_count} 份樣本。")
    return samples, total_count

def main():
    clear_terminal()
    draw_header()
    
    # 確認 snapshot 資料夾存在
    snapshot_dir = Path('./snapshot')
    if not snapshot_dir.exists():
        print(f"[Error] 未找到 snapshot 資料夾: '{snapshot_dir.resolve()}'")
        sys.exit(1)
        
    # 執行數據掃描
    samples, total_count = scan_datasets(snapshot_dir)
    if total_count == 0:
        print("[Error] snapshot 資料夾內沒有找到任何 .dat 數據。請先採集數據。")
        sys.exit(1)
        
    print("-" * 70)
    
    # 1. 設置訓練集/測試集分割比例（預設 50%）
    try:
        split_input = input("請輸入訓練集分割比例 % (10 到 90, 預設為 50): ").strip()
        if split_input == "":
            split_perc = 50.0
        else:
            split_perc = float(split_input)
            if split_perc < 10 or split_perc > 90:
                print("[!] 輸入超出範圍，套用預設值 50.0%")
                split_perc = 50.0
    except ValueError:
        print("[!] 無效輸入，套用預設值 50.0%")
        split_perc = 50.0
        
    print(f"[*] 訓練集比例: {split_perc}% | 測試集比例: {100.0 - split_perc}%")
    print("-" * 70)
    
    input("按 ENTER 鍵開始訓練 PyTorch 分類模型...")
    
    print("\n[*] 正在執行數據集隨機分割...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 整合與隨機分流
    all_samples = []
    for s_list in samples.values():
        all_samples.extend(s_list)
        
    samples_by_label = {}
    for s in all_samples:
        samples_by_label.setdefault(s.label, []).append(s)
        
    train_samples = []
    test_samples = []
    ratio = split_perc / 100.0
    
    for label, samples_list in samples_by_label.items():
        shuffled = list(samples_list)
        random.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * ratio))
        train_samples.extend(shuffled[:split_idx])
        test_samples.extend(shuffled[split_idx:])
        
    if not train_samples:
        train_samples.append(all_samples[0])
        test_samples = all_samples[1:]
        
    print(f"[*] 分割完成: 訓練集樣本數 = {len(train_samples)}, 測試集樣本數 = {len(test_samples)}")
    print("-" * 70)
    
    # 準備 PyTorch 分類網路
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 計算執行裝置: {device.upper()}")
    
    model = ToFClassifierModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train_dataset = ToFDataset(train_samples)
    test_dataset = ToFDataset(test_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    layer_weights = {
        "Layer 1": [],
        "Layer 2": [],
        "Layer 3": []
    }
    
    epochs = 30
    print(f"[*] 開始執行 {epochs} 個 Epoch 訓練...")
    print("=" * 70)
    
    # 執行 PyTorch 訓練迴圈
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        
        for features, labels in train_loader:
            # 將 2D [8, 8] 輸入展平成 1D [64]
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
        
        # 測試集評估
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
        
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(val_acc)
        
        # 追蹤網路參數變化
        with torch.no_grad():
            w1 = model.network[0].weight.cpu().numpy()
            w2 = model.network[3].weight.cpu().numpy()
            w3 = model.network[6].weight.cpu().numpy()
            
            layer_weights["Layer 1"].append(float(w1.mean()))
            layer_weights["Layer 2"].append(float(w2.mean()))
            layer_weights["Layer 3"].append(float(w3.mean()))
            
        # 繪製終端機動態 ASCII 進度條
        bar_length = 20
        progress_ratio = epoch / epochs
        filled_length = int(bar_length * progress_ratio)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"Epoch {epoch:>2}/{epochs} [{bar}] {int(progress_ratio*100):>3}% | "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.1f}% | "
              f"Test Loss: {val_loss:.4f} | Test Acc: {val_acc*100:.1f}%")
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  -> 層參數平均值: Layer 1: {w1.mean():.4f} | Layer 2: {w2.mean():.4f} | Layer 3: {w3.mean():.4f}")
            
    print("=" * 70)
    print("[*] 模型神經網路訓練成功完成！")
    
    # 儲存神經網路與元數據 YAML 記錄
    models_dir = Path('./models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 儲存權重檔
    model_path = models_dir / f"model_{timestamp}.pth"
    payload = {
        "model_kwargs": model.config(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, model_path)
    
    # 儲存元數據 YAML 歷史對照檔案，供即時推論前追溯分割來源
    yaml_path = models_dir / f"model_{timestamp}.yaml"
    yaml_data = {
        "timestamp": timestamp,
        "train_ratio": split_perc,
        "train_set_size": len(train_samples),
        "test_set_size": len(test_samples),
        "final_train_loss": float(train_losses[-1]),
        "final_test_loss": float(test_losses[-1]),
        "final_train_accuracy": float(train_accuracies[-1]),
        "final_test_accuracy": float(test_accuracies[-1]),
        "train_files": [str(s.path.resolve().relative_to(Path('.').resolve())) for s in train_samples if s.path],
        "test_files": [str(s.path.resolve().relative_to(Path('.').resolve())) for s in test_samples if s.path],
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_data, f, default_flow_style=False, allow_unicode=True)
        
    print("\n" + "=" * 70)
    print("      TRAINING METADATA & ARTIFACT RECORD (訓練元數據報告)")
    print("=" * 70)
    print(f"  模型存檔路徑:   {model_path.resolve()}")
    print(f"  YAML 歷史日誌:  {yaml_path.resolve()}")
    print(f"  訓練集準確率:   {train_accuracies[-1]*100:.2f}%  (損失率: {train_losses[-1]:.4f})")
    print(f"  測試集準確率:   {test_accuracies[-1]*100:.2f}%  (損失率: {test_loss:.4f})")
    print("=" * 70)
    print("[*] 終端機訓練流程正常退出。已就緒，可立即在 GUI 推論模式中載入此模型進行實時預測！\n")

if __name__ == "__main__":
    main()
