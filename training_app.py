import threading
import logging
from pathlib import Path
from imgui_bundle import imgui, hello_imgui, implot, implot3d

from ai.ToFDataLabel import ToFDataLabel

from common.ToFData import ToFData
from common.ViewAngle import ViewAngle
from common.WorldCoord import WorldCoord

from AppBase import App as AppBase
from collections import deque

from training import ToFSample, ToFDataset, ToFDataLoader

    

class TrainingApp(AppBase):
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
        
        


