import threading
import math
import random
import logging
import numpy as np

from datetime import datetime
from pathlib import Path
from imgui_bundle import imgui, hello_imgui, implot, implot3d

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from ai.ToFDataLabel import ToFDataLabel
from dataclasses import dataclass

from common.ToFData import ToFData
from common.ViewAngle import ViewAngle
from common.WorldCoord import WorldCoord

from AppBase import App as AppBase
from collections import deque

@dataclass
class ToFSample:
    label: int
    path: Path
    data: ToFData
    points: WorldCoord

    def __post_init__(self) -> None:
        expected_height, expected_width = 8, 8
        actual_height, actual_width = self.data.repaired_data.shape

        if actual_height != expected_height or actual_width != expected_width:
            raise ValueError(
                f'Expect {expected_height} rows * {expected_width} cols. Got {actual_height} rows * {actual_width} cols.'
            )
        
    def to_sensor(self) -> torch.Tensor:
        tensor = torch.tensor(self.data, dtype = torch.float32)
        return tensor.clamp(min=ToFData.min_valid, max=ToFData.max_valid) / ToFData.max_valid
    
    @classmethod
    def from_data_file(cls, label: int, path: Path) -> 'ToFSample':
        tof_data = ToFData.from_dat(path)
        return cls(
            label,
            path,
            tof_data,
            WorldCoord(
                tof_data.repaired_data,
                ViewAngle()
            ))
    
    @classmethod
    def from_tof_data(cls, label: int, tof_data: ToFData) -> 'ToFSample':
        return cls(label, tof_data.repaired_data)
    
class ToFDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)

class ToFDataLoader(DataLoader):
    pass

class App(AppBase):
    def __init__(self):
        super().__init__()

        self.__win_title = '數據檢視器'
        self.selected_item = None

        self.samples = {}
        threading.Thread(
            target=self.data_pre_scan,
            daemon=True
        ).start()

    def gui(self):
        pass

    def gui_settings(self):
        super().gui_settings()
            
        imgui.separator_text('Loaded')

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


    def gui_heatmap(self):
        pass

    def data_pre_scan(self):
        for lbl in ToFDataLabel.labels:
            dir = self.__snapshot_dir / lbl.name
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
        
        





if __name__ == '__main__':
    app = App()
    app.run()