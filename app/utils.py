"""Utility functions for the Flask app."""
import pathlib
import json
from typing import Optional
import numpy as np

from ai.ToFTrainer import ToFSample, ToFLabels
from common.ToFData import ToFData


def load_tof_data(dat_file: pathlib.Path) -> Optional[ToFData]:
    """Load ToF data from a .dat file."""
    try:
        if not dat_file.exists():
            return None
        return ToFData.from_dat(dat_file)
    except Exception as e:
        print(f"Error loading {dat_file}: {e}")
        return None


def normalize_matrix(data: list[int], min_val: int = 0, max_val: int = 30000) -> list[float]:
    """Normalize distance values to 0-1 range for display."""
    return [(min(max(val, min_val), max_val) - min_val) / (max_val - min_val) for val in data]


def create_tof_sample(dat_file: pathlib.Path, label: str) -> Optional[ToFSample]:
    """Create a ToFSample from a dat file."""
    try:
        label_map = {
            "Normal": ToFLabels.Normal,
            "Upstairs": ToFLabels.Upstairs,
            "Downstairs": ToFLabels.Downstairs
        }
        
        if label not in label_map:
            return None
        
        tof_data = load_tof_data(dat_file)
        if tof_data is None:
            return None
        
        return ToFSample.from_tof_data(tof_data, label_map[label])
    except Exception as e:
        print(f"Error creating sample from {dat_file}: {e}")
        return None


def save_model_metadata(model_path: pathlib.Path, metadata: dict) -> None:
    """Save model metadata to JSON file."""
    meta_path = model_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))


def load_model_metadata(model_path: pathlib.Path) -> Optional[dict]:
    """Load model metadata from JSON file."""
    meta_path = model_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text())
    except:
        return None


def compute_confusion_matrix_display(confusion_dict: dict) -> dict:
    """Convert confusion matrix dict to displayable format."""
    label_names = {0: "Normal", 1: "Upstairs", 2: "Downstairs"}
    display = {}
    for true_label, predictions in confusion_dict.items():
        display[label_names.get(true_label, str(true_label))] = {
            label_names.get(pred_label, str(pred_label)): count
            for pred_label, count in predictions.items()
        }
    return display
