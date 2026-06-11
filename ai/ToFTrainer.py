import json
import pathlib
import random
from dataclasses import asdict, dataclass
from typing import Iterable

from common.ToFData import ToFData

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for ai.ToFTrainer. Install it with `pip install torch`."
    ) from exc


class ToFLabels:
    Normal = 0
    Upstairs = 1
    Downstairs = 2


@dataclass
class ToFSample:
    label: int
    data: list[int]

    def __post_init__(self) -> None:
        expected = ToFData.width * ToFData.height
        if len(self.data) != expected:
            raise ValueError(
                f"ToFSample requires {expected} distance values, got {len(self.data)}"
            )

    def to_tensor(self) -> torch.Tensor:
        tensor = torch.tensor(self.data, dtype=torch.float32)
        return tensor.clamp(min=0.0, max=30000.0) / 30000.0

    @classmethod
    def from_tof_data(cls, tof_data: ToFData, label: int) -> "ToFSample":
        return cls(label=label, data=list(tof_data.data))

    @classmethod
    def from_raw_bytes(cls, raw_data: list[int], label: int) -> "ToFSample":
        return cls.from_tof_data(ToFData(raw_data), label)

    @classmethod
    def from_int_list(cls, data: list[int], label: int) -> "ToFSample":
        return cls(label=label, data=list(data))

    def as_dict(self) -> dict:
        return {"label": self.label, "data": list(self.data)}

    @staticmethod
    def from_dict(payload: dict) -> "ToFSample":
        return ToFSample(label=int(payload["label"]), data=[int(x) for x in payload["data"]])


class ToFDataset(Dataset):
    def __init__(self, samples: Iterable[ToFSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        return sample.to_tensor(), sample.label


class ToFClassifierModel(nn.Module):
    def __init__(
        self,
        input_size: int = ToFData.width * ToFData.height,
        hidden_sizes: tuple[int, ...] = (128, 64),
        num_classes: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout = dropout

        layers: list[nn.Module] = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            current_size = hidden_size
        layers.append(nn.Linear(current_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: dict) -> "ToFClassifierModel":
        return cls(
            input_size=config.get("input_size", ToFData.width * ToFData.height),
            hidden_sizes=tuple(config.get("hidden_sizes", (128, 64))),
            num_classes=config.get("num_classes", 3),
            dropout=config.get("dropout", 0.2),
        )


class ToFTrainer:
    def __init__(self, model: ToFClassifierModel | None = None, device: str | None = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model or ToFClassifierModel()).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer: torch.optim.Optimizer | None = None
        self.is_trained = False

    def fit(
        self,
        samples: Iterable[ToFSample],
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        validation_samples: Iterable[ToFSample] | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("At least one training sample is required")

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        dataset = ToFDataset(sample_list)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        history = {"loss": [], "val_loss": []}
        validation_loader = None
        if validation_samples is not None:
            validation_loader = DataLoader(
                ToFDataset(list(validation_samples)), batch_size=batch_size, shuffle=False
            )

        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * features.size(0)

            epoch_loss /= len(dataset)
            history["loss"].append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch}/{epochs}: loss={epoch_loss:.4f}")

            if validation_loader is not None:
                val_loss = self._evaluate_loss(validation_loader)
                history["val_loss"].append(val_loss)
                if verbose:
                    print(f"  validation_loss={val_loss:.4f}")

        self.is_trained = True
        return history

    def _evaluate_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(features)
                loss = self.criterion(logits, labels)
                total_loss += loss.item() * features.size(0)
                total_count += features.size(0)
        self.model.train()
        return total_loss / total_count if total_count else 0.0

    def predict(self, samples: Iterable[ToFSample]) -> list[int]:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        dataset = ToFDataset(list(samples))
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        self.model.eval()
        predictions: list[int] = []
        with torch.no_grad():
            for features, _ in loader:
                features = features.to(self.device)
                logits = self.model(features)
                labels = torch.argmax(logits, dim=1)
                predictions.extend(labels.cpu().tolist())
        self.model.train()
        return predictions

    def evaluate(self, samples: Iterable[ToFSample]) -> dict[str, float | dict[int, int]]:
        sample_list = list(samples)
        if not sample_list:
            return {"accuracy": 0.0, "confusion_matrix": {}}

        predictions = self.predict(sample_list)
        correct = 0
        confusion: dict[int, dict[int, int]] = {}
        for sample, predicted in zip(sample_list, predictions):
            if predicted == sample.label:
                correct += 1
            confusion.setdefault(sample.label, {})
            confusion[sample.label][predicted] = confusion[sample.label].get(predicted, 0) + 1

        return {
            "accuracy": correct / len(sample_list),
            "confusion_matrix": confusion,
        }

    def save_model(self, path: pathlib.Path) -> None:
        payload = {
            "model_kwargs": self.model.config(),
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, path)

    @staticmethod
    def load_model(path: pathlib.Path, device: str | None = None) -> "ToFTrainer":
        checkpoint = torch.load(path, map_location=device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = ToFClassifierModel.from_config(checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        trainer = ToFTrainer(model=model, device=device)
        trainer.is_trained = True
        return trainer

    @staticmethod
    def save_dataset(path: pathlib.Path, samples: Iterable[ToFSample]) -> None:
        payload = [sample.as_dict() for sample in samples]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load_dataset(path: pathlib.Path) -> list[ToFSample]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [ToFSample.from_dict(item) for item in payload]

