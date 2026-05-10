from dataclasses import dataclass, field
import torch


@dataclass
class DataConfig:
    json_path: str = "data/dataset.json"
    label_studio_json: str = "data/label_studio_export.json"
    images_dir: str = "data/images"
    masks_dir: str = "data/masks"
    target_size: tuple = (256, 256)
    sigma: float = 3.0
    keypoint_names: list = field(default_factory=lambda: ["nose", "shoulder", "tail"])


@dataclass
class ModelConfig:
    num_classes: int = 2
    num_keypoints: int = 3


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-4
    loss_seg_weight: float = 1
    # Scale factor to balance MSE vs CE magnitudes
    loss_pose_weight: float = 1
    pck_threshold: float = 0.05
    checkpoint_interval: int = 10
    output_dir: str = "models/checkpoints"
    num_workers: int = 4

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
