from torch.utils.data import Dataset, DataLoader
from .sample import ToFSample

class ToFDataset(Dataset):
    def __init__(self, samples: list[ToFSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return sample.to_sensor(), sample.label

class ToFDataLoader(DataLoader):
    """
    Custom DataLoader for ToF data.
    Currently behaves like a standard DataLoader, but can be extended
    for custom batching or sampling logic.
    """
    pass
