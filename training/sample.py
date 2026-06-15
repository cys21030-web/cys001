from dataclasses import dataclass
from pathlib import Path
import torch

from common.ToFData import ToFData
from common.ViewAngle import ViewAngle
from common.WorldCoord import WorldCoord

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
        tensor = torch.tensor(self.data.repaired_data, dtype = torch.float32)
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
    def from_tof_data(cls, label: int, tof_data: ToFData, path: Path = None) -> 'ToFSample':
        return cls(
            label,
            path,
            tof_data,
            WorldCoord(tof_data.repaired_data, ViewAngle())
        )
