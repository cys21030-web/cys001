#!/usr/bin/env python
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

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def draw_header():
    print("=" * 70)
    print("      ToF Lidar Classifier - Terminal Training System (TUI)")
    print("=" * 70)

def scan_datasets(snapshot_dir: Path):
    print("[*] Scanning snapshot directories...")
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
                
        print(f"  - Label {lbl.name:<10} [Index {lbl.index}]: Loaded {len(samples[lbl.name]):>3} files")
        
    print(f"[*] Total dataset size loaded: {total_count} samples.")
    return samples, total_count

def main():
    clear_terminal()
    draw_header()
    
    # Identify snapshot directory
    snapshot_dir = Path('./snapshot')
    if not snapshot_dir.exists():
        print(f"[Error] Snapshot directory '{snapshot_dir.resolve()}' not found.")
        sys.exit(1)
        
    # Scan datasets
    samples, total_count = scan_datasets(snapshot_dir)
    if total_count == 0:
        print("[Error] No .dat samples found inside the snapshot folders. Please collect data first.")
        sys.exit(1)
        
    print("-" * 70)
    
    # 1. Slider/Prompt to set split percentage (Default 50%)
    try:
        split_input = input("Enter training split ratio % (10 to 90, default 50): ").strip()
        if split_input == "":
            split_perc = 50.0
        else:
            split_perc = float(split_input)
            if split_perc < 10 or split_perc > 90:
                print("[!] Out of bounds, using default 50.0%")
                split_perc = 50.0
    except ValueError:
        print("[!] Invalid input, using default 50.0%")
        split_perc = 50.0
        
    print(f"[*] Configured Train/Test ratio: {split_perc}% / {100.0 - split_perc}%")
    print("-" * 70)
    
    input("Press ENTER to start the PyTorch training pipeline...")
    
    print("\n[*] Initializing split...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Gather and split
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
        
    print(f"[*] Split completed: Train subset = {len(train_samples)}, Test subset = {len(test_samples)}")
    print("-" * 70)
    
    # Setup PyTorch components natively
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Compute Device: {device.upper()}")
    
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
    print(f"[*] Commencing {epochs} training epochs...")
    print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        
        for features, labels in train_loader:
            # Flatten 2D (8, 8) input to 1D (64)
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
        
        # Evaluate
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
        
        # Track Layer Weights Mean
        with torch.no_grad():
            w1 = model.network[0].weight.cpu().numpy()
            w2 = model.network[3].weight.cpu().numpy()
            w3 = model.network[6].weight.cpu().numpy()
            
            layer_weights["Layer 1"].append(float(w1.mean()))
            layer_weights["Layer 2"].append(float(w2.mean()))
            layer_weights["Layer 3"].append(float(w3.mean()))
            
        # Draw dynamic ASCII progress bar
        bar_length = 20
        progress_ratio = epoch / epochs
        filled_length = int(bar_length * progress_ratio)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        print(f"Epoch {epoch:>2}/{epochs} [{bar}] {int(progress_ratio*100):>3}% | "
              f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.1f}% | "
              f"Test Loss: {val_loss:.4f} | Test Acc: {val_acc*100:.1f}%")
        
        # Track weight variations for Layer 3 (Final Classifier weights) in the trace
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  -> Weight Mean Stats: Layer 1: {w1.mean():.4f} | Layer 2: {w2.mean():.4f} | Layer 3: {w3.mean():.4f}")
            
    print("=" * 70)
    print("[*] Training pipeline completed!")
    
    # Save model and Output YAML record
    models_dir = Path('./models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / f"model_{timestamp}.pth"
    payload = {
        "model_kwargs": model.config(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, model_path)
    
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
    print("      TRAINING METADATA & ARTIFACT RECORD")
    print("=" * 70)
    print(f"  Model Saved to:  {model_path.resolve()}")
    print(f"  YAML Log Saved:  {yaml_path.resolve()}")
    print(f"  Train Acc:       {train_accuracies[-1]*100:.2f}%  (Loss: {train_losses[-1]:.4f})")
    print(f"  Test Acc:        {test_accuracies[-1]*100:.2f}%  (Loss: {test_losses[-1]:.4f})")
    print("=" * 70)
    print("[*] TUI pipeline session finished. Ready to load into Inference mode!\n")

if __name__ == "__main__":
    main()
