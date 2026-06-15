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

# 設定系統日誌格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clear_terminal():
    """
    清除終端機螢幕以保持 TUI 介面乾淨。
    """
    os.system('cls' if os.name == 'nt' else 'clear')


def draw_header():
    """
    繪製 TUI 應用程式主標頭。
    """
    print("=" * 70)
    print("      ToF Lidar Classifier - Terminal Training System (TUI)")
    print("=" * 70)


def scan_datasets(snapshot_dir: Path) -> tuple[dict[str, list[ToFSample]], int]:
    """
    掃描指定的快照資料夾，並載入所有合格的 .dat 影格樣本。
    """
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
            except Exception:
                pass
                
        print(f"  - 標籤 {lbl.name:<10} [索引 {lbl.index}]: 成功載入 {len(samples[lbl.name]):>3} 份文件")
        
    print(f"[*] 數據集載入完成，共計 {total_count} 份樣本。")
    return samples, total_count


def get_split_percentage() -> float:
    """
    向使用者請求 Train/Test 的分割比例，並提供輸入驗證與預設值 (50%)。
    """
    try:
        split_input = input("請輸入訓練集分割比例 % (10 到 90, 預設為 50): ").strip()
        if split_input == "":
            return 50.0
        
        split_perc = float(split_input)
        if split_perc < 10 or split_perc > 90:
            print("[!] 輸入超出範圍，套用預設值 50.0%")
            return 50.0
        return split_perc
    except ValueError:
        print("[!] 無效輸入，套用預設值 50.0%")
        return 50.0


def split_dataset(samples: dict[str, list[ToFSample]], split_perc: float) -> tuple[list[ToFSample], list[ToFSample]]:
    """
    依據指定的分割比例，對各類別數據進行隨機、比例一致的 Stratified 分流。
    """
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
        split_idx = max(1, int(len(shuffled) * ratio)) # 確保訓練集有數據
        train_samples.extend(shuffled[:split_idx])
        test_samples.extend(shuffled[split_idx:])
        
    # 保底機制，避免極端數值分割失敗
    if not train_samples and all_samples:
        train_samples.append(all_samples[0])
        test_samples = all_samples[1:]
        
    return train_samples, test_samples


def build_dataloader(train_samples: list[ToFSample], test_samples: list[ToFSample]) -> tuple[DataLoader, DataLoader]:
    """
    使用自定義 ToFDataset 建立 PyTorch DataLoader。
    """
    train_dataset = ToFDataset(train_samples)
    test_dataset = ToFDataset(test_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return train_loader, test_loader


def create_network_model(device: str, num_classes: int) -> ToFClassifierModel:
    """
    建立自定義的神經網路架構。
    根據指定，我們建構具有 3 個線性層 (Linear Layers) 的深度網路：
    - Layer 1 (第一線性層): 64 節點 -> 64 隱藏節點 (搭配 ReLU)
    - Layer 2 (第二線性層): 64 隱藏節點 -> 64 隱藏節點 (搭配 ReLU)
    - Layer 3 (第三線性層): 64 隱藏節點 -> nLabels 輸出類別
    不使用 Dropout (dropout=0.0) 以保持精確與簡單。
    """
    model = ToFClassifierModel(
        input_size=64,
        hidden_sizes=(64, 64),
        num_classes=num_classes,
        dropout=0.0
    ).to(device)
    return model


def train_single_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> tuple[float, float]:
    """
    執行單個 Epoch 的神經網路前向與反向傳播訓練。
    """
    model.train()
    epoch_loss = 0.0
    correct = 0
    total_count = 0
    
    for features, labels in loader:
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
        total_count += labels.size(0)
        
    avg_loss = epoch_loss / total_count if total_count > 0 else 0.0
    accuracy = correct / total_count if total_count > 0 else 0.0
    return avg_loss, accuracy


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: str) -> tuple[float, float]:
    """
    在測試/驗證集上執行前向傳播，評估模型的泛化損失率與準確率。
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    total_count = 0
    
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device).view(features.size(0), -1)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            
            test_loss += loss.item() * features.size(0)
            predictions = torch.argmax(logits, dim=1)
            test_correct += (predictions == labels).sum().item()
            total_count += labels.size(0)
            
    avg_loss = test_loss / total_count if total_count > 0 else 0.0
    accuracy = test_correct / total_count if total_count > 0 else 0.0
    return avg_loss, accuracy


def execute_training_session(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str, total_epochs: int) -> tuple[list[float], list[float], list[float], list[float], dict[str, list[float]]]:
    """
    管理整場多 Epoch 的神經網路訓練工作流，並實時輸出進度條與統計指標。
    """
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    layer_weights = {"Layer 1": [], "Layer 2": [], "Layer 3": []}
    
    print(f"[*] 開始執行 {total_epochs} 個 Epoch 訓練...")
    print("=" * 70)
    
    for epoch in range(1, total_epochs + 1):
        # 訓練與評估
        train_loss, train_acc = train_single_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
        
        # 紀錄指標
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(val_acc)
        
        # 紀錄權重參數平均值變化
        with torch.no_grad():
            w1 = model.network[0].weight.cpu().numpy()
            w2 = model.network[3].weight.cpu().numpy()
            w3 = model.network[6].weight.cpu().numpy()
            
            layer_weights["Layer 1"].append(float(w1.mean()))
            layer_weights["Layer 2"].append(float(w2.mean()))
            layer_weights["Layer 3"].append(float(w3.mean()))
            
        # 繪製終端機 ASCII 動態進度條
        bar_length = 20
        progress_ratio = epoch / total_epochs
        filled_length = int(bar_length * progress_ratio)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"Epoch {epoch:>2}/{total_epochs} [{bar}] {int(progress_ratio*100):>3}% | "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.1f}% | "
              f"Test Loss: {val_loss:.4f} | Test Acc: {val_acc*100:.1f}%")
        
        if epoch % 10 == 0 or epoch == total_epochs:
            print(f"  -> 層參數平均值: Layer 1: {w1.mean():.4f} | Layer 2: {w2.mean():.4f} | Layer 3: {w3.mean():.4f}")
            
    print("=" * 70)
    print("[*] 模型神經網路訓練成功完成！")
    return train_losses, test_losses, train_accuracies, test_accuracies, layer_weights


def save_training_artifacts(model: nn.Module, train_samples: list[ToFSample], test_samples: list[ToFSample], train_losses: list[float], test_losses: list[float], train_accuracies: list[float], test_accuracies: list[float], split_perc: float, timestamp: str) -> tuple[Path, Path]:
    """
    將模型權重（.pth）與本次數據分割詳細歷程與對照（.yaml）序列化儲存至本地磁碟。
    """
    models_dir = Path('./models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 儲存 .pth 分類結構與權重檔
    model_path = models_dir / f"model_{timestamp}.pth"
    payload = {
        "model_kwargs": model.config(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, model_path)
    
    # 儲存元數據 YAML 日誌
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
        
    return model_path, yaml_path


def print_final_report(model_path: Path, yaml_path: Path, train_accuracies: list[float], test_accuracies: list[float], train_losses: list[float], test_losses: list[float]):
    """
    輸出格式美觀的終端數據統計分析報告。
    """
    print("\n" + "=" * 70)
    print("      TRAINING METADATA & ARTIFACT RECORD (訓練元數據報告)")
    print("=" * 70)
    print(f"  模型存檔路徑:   {model_path.resolve()}")
    print(f"  YAML 歷史日誌:  {yaml_path.resolve()}")
    print(f"  訓練集準確率:   {train_accuracies[-1]*100:.2f}%  (損失率: {train_losses[-1]:.4f})")
    print(f"  測試集準確率:   {test_accuracies[-1]*100:.2f}%  (損失率: {test_losses[-1]:.4f})")
    print("=" * 70)
    print("[*] 終端機訓練流程正常退出。已就緒，可立即在 GUI 推論模式中載入此模型進行實時預測！\n")


def main():
    clear_terminal()
    draw_header()
    
    # 掃描本地快照數據集
    snapshot_dir = Path('./snapshot')
    if not snapshot_dir.exists():
        print(f"[Error] 未找到 snapshot 快照資料夾: '{snapshot_dir.resolve()}'")
        sys.exit(1)
        
    samples, total_count = scan_datasets(snapshot_dir)
    if total_count == 0:
        print("[Error] snapshot 資料夾內沒有找到任何合格的數據。請先至 GUI 採集數據。")
        sys.exit(1)
        
    print("-" * 70)
    
    # 獲取分割比例、分割數據並配置 DataLoader
    split_perc = get_split_percentage()
    print(f"[*] 訓練集分割比率: {split_perc}% | 測試集比率: {100.0 - split_perc}%")
    print("-" * 70)
    
    input("按 ENTER 鍵開始訓練 PyTorch 3 層神經網路模型...")
    
    print("\n[*] 正在進行數據集分割與洗牌...")
    train_samples, test_samples = split_dataset(samples, split_perc)
    print(f"[*] 數據集分割完成。訓練樣本數: {len(train_samples)} | 測試樣本數: {len(test_samples)}")
    print("-" * 70)
    
    train_loader, test_loader = build_dataloader(train_samples, test_samples)
    
    # 建立多層線性分類網路 (3 Layers, Shape: 64 -> 64 -> 64 -> nLabels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] 計算執行裝置: {device.upper()}")
    
    num_classes = len(ToFDataLabel.labels)
    model = create_network_model(device, num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 執行模型訓練
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_epochs = 30
    
    train_losses, test_losses, train_accuracies, test_accuracies, _ = execute_training_session(
        model, train_loader, test_loader, criterion, optimizer, device, total_epochs
    )
    
    # 儲存神經網絡與 YAML 元數據紀錄
    model_path, yaml_path = save_training_artifacts(
        model, train_samples, test_samples, train_losses, test_losses, train_accuracies, test_accuracies, split_perc, timestamp
    )
    
    # 輸出最終對照報告
    print_final_report(model_path, yaml_path, train_accuracies, test_accuracies, train_losses, test_losses)


if __name__ == "__main__":
    main()
