"""Cornell Grasp Dataset loader.

The Cornell dataset stores each object as an RGB image, a depth image, and a
text file containing positive grasp rectangles. This loader converts those
rectangles into dense maps so a fully convolutional network can learn:

- grasp quality at each pixel
- grasp angle at each pixel
- gripper opening width at each pixel
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

InputMode = Literal["rgb", "depth", "rgbd"]
Split = Literal["train", "val", "test", "all"]


@dataclass(frozen=True)
class GraspRectangle:
    """One oriented grasp rectangle using image coordinates as ``(x, y)``."""

    points: np.ndarray

    def __post_init__(self) -> None:
        if self.points.shape != (4, 2):
            raise ValueError("GraspRectangle points must have shape (4, 2).")

    @property
    def center(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def angle(self) -> float:
        """Return the grasp axis angle in radians, normalized to [-pi/2, pi/2]."""

        edge = self.points[1] - self.points[0]
        angle = float(np.arctan2(edge[1], edge[0]))
        return normalize_grasp_angle(angle)

    @property
    def width(self) -> float:
        """Approximate gripper opening in pixels.

        Cornell rectangles are annotated with four corners. For this simple MVP
        we use the first edge as the gripper opening direction.
        """

        return float(np.linalg.norm(self.points[1] - self.points[0]))

    def scaled(self, sx: float, sy: float) -> "GraspRectangle":
        scale = np.array([sx, sy], dtype=np.float32)
        return GraspRectangle((self.points * scale).astype(np.float32))


@dataclass(frozen=True)
class CornellRecord:
    rgb_path: Path
    depth_path: Path
    grasp_path: Path


def normalize_grasp_angle(angle: float) -> float:
    """Normalize antipodal grasp angles to [-pi/2, pi/2]."""

    while angle > np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    return float(angle)


def load_grasp_rectangles(path: Path) -> list[GraspRectangle]:
    """Read positive Cornell grasp rectangles from a ``*cpos.txt`` file."""

    raw_points: list[list[float]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.replace("\t", " ").split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        if np.isfinite(x) and np.isfinite(y):
            raw_points.append([x, y])

    rectangles: list[GraspRectangle] = []
    for start in range(0, len(raw_points), 4):
        block = raw_points[start : start + 4]
        if len(block) == 4:
            rectangles.append(GraspRectangle(np.asarray(block, dtype=np.float32)))
    return rectangles


class CornellGraspDataset(Dataset[dict[str, torch.Tensor]]):
    """PyTorch dataset for Cornell RGB-D grasp prediction."""

    def __init__(
        self,
        root: str | Path,
        input_mode: InputMode = "rgbd",
        image_size: int = 300,
        split: Split = "train",
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.input_mode = input_mode
        self.image_size = int(image_size)
        self.split = split
        self.val_split = float(val_split)
        self.test_split = float(test_split)
        self.seed = int(seed)
        self.augment = bool(augment) and split == "train"

        if self.input_mode not in {"rgb", "depth", "rgbd"}:
            raise ValueError("input_mode must be one of: rgb, depth, rgbd")

        records = self._find_records(self.root)
        if not records:
            raise FileNotFoundError(
                f"No Cornell samples found under {self.root}. Expected files like "
                "pcd0100r.png, pcd0100d.tiff, and pcd0100cpos.txt."
            )

        rng = np.random.default_rng(self.seed)
        indices = np.arange(len(records))
        rng.shuffle(indices)

        if split == "all":
            chosen = indices
        else:
            if self.val_split < 0 or self.test_split < 0 or self.val_split + self.test_split >= 1:
                raise ValueError("val_split and test_split must be non-negative and sum to less than 1.")
            val_count = int(round(len(indices) * self.val_split))
            test_count = int(round(len(indices) * self.test_split))
            if len(indices) >= 3:
                val_count = max(1, val_count)
                test_count = max(1, test_count)
            if val_count + test_count >= len(indices):
                raise ValueError("Dataset is too small for the requested train/val/test split fractions.")
            val_indices = indices[:val_count]
            test_indices = indices[val_count:val_count + test_count]
            train_indices = indices[val_count + test_count:]
            chosen = {"train": train_indices, "val": val_indices, "test": test_indices}[split]

        self.records = [records[int(i)] for i in chosen]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rgb, depth, original_size = self._load_images(index)
        image = self._build_input_tensor(rgb, depth)
        rectangles = self.get_rectangles(index, original_size=original_size)
        quality, angle_cos, angle_sin, width = self._build_target_maps(rectangles)

        sample = {
            "image": image,
            "quality": torch.from_numpy(quality).unsqueeze(0),
            "angle_cos": torch.from_numpy(angle_cos).unsqueeze(0),
            "angle_sin": torch.from_numpy(angle_sin).unsqueeze(0),
            "width": torch.from_numpy(width).unsqueeze(0),
            "index": torch.tensor(index, dtype=torch.long),
        }
        if self.augment:
            sample = self._augment_sample(sample)
        return sample

    @property
    def in_channels(self) -> int:
        if self.input_mode == "rgb":
            return 3
        if self.input_mode == "depth":
            return 1
        return 4

    def get_rgb_image(self, index: int) -> Image.Image:
        """Return the resized RGB image for visualization."""

        record = self.records[index]
        return Image.open(record.rgb_path).convert("RGB").resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )

    def get_rectangles(
        self,
        index: int,
        original_size: tuple[int, int] | None = None,
    ) -> list[GraspRectangle]:
        """Return grasp rectangles scaled to the configured square image size."""

        record = self.records[index]
        rectangles = load_grasp_rectangles(record.grasp_path)
        if original_size is None:
            with Image.open(record.rgb_path) as img:
                original_size = img.size
        original_w, original_h = original_size
        sx = self.image_size / float(original_w)
        sy = self.image_size / float(original_h)
        return [rect.scaled(sx=sx, sy=sy) for rect in rectangles]

    @staticmethod
    def _find_records(root: Path) -> list[CornellRecord]:
        records: list[CornellRecord] = []
        for grasp_path in sorted(root.rglob("*cpos.txt")):
            base = grasp_path.name.replace("cpos.txt", "")
            parent = grasp_path.parent
            rgb_path = _first_existing(
                [
                    parent / f"{base}r.png",
                    parent / f"{base}r.jpg",
                    parent / f"{base}r.jpeg",
                ]
            )
            depth_path = _first_existing(
                [
                    parent / f"{base}d.tiff",
                    parent / f"{base}d.tif",
                    parent / f"{base}d.png",
                    parent / f"{base}d.jpg",
                ]
            )
            if rgb_path is not None and depth_path is not None:
                records.append(
                    CornellRecord(
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        grasp_path=grasp_path,
                    )
                )
        return records

    def _load_images(self, index: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
        record = self.records[index]
        rgb_img = Image.open(record.rgb_path).convert("RGB")
        original_size = rgb_img.size
        depth_img = Image.open(record.depth_path)

        rgb_img = rgb_img.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )
        depth_img = depth_img.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR,
        )

        rgb = np.asarray(rgb_img, dtype=np.float32) / 255.0
        depth = np.asarray(depth_img, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = _normalize_depth(depth)
        return rgb, depth, original_size

    def _build_input_tensor(self, rgb: np.ndarray, depth: np.ndarray) -> torch.Tensor:
        rgb_chw = np.transpose((rgb - 0.5) / 0.5, (2, 0, 1))
        depth_chw = ((depth - 0.5) / 0.5)[None, ...]

        if self.input_mode == "rgb":
            image = rgb_chw
        elif self.input_mode == "depth":
            image = depth_chw
        else:
            image = np.concatenate([rgb_chw, depth_chw], axis=0)
        return torch.from_numpy(image.astype(np.float32))

    def _build_target_maps(
        self,
        rectangles: list[GraspRectangle],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        quality = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        angle_cos = np.zeros_like(quality)
        angle_sin = np.zeros_like(quality)
        width = np.zeros_like(quality)

        for rect in rectangles:
            polygon = [tuple(point) for point in rect.points.tolist()]
            mask_img = Image.new("L", (self.image_size, self.image_size), 0)
            draw = ImageDraw.Draw(mask_img)
            draw.polygon(polygon, fill=1)
            mask = np.asarray(mask_img, dtype=bool)

            quality[mask] = 1.0
            angle_cos[mask] = np.cos(2.0 * rect.angle)
            angle_sin[mask] = np.sin(2.0 * rect.angle)
            width[mask] = np.clip(rect.width / float(self.image_size), 0.0, 1.0)

        return quality, angle_cos, angle_sin, width

    def _augment_sample(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply label-preserving square-image augmentations for antipodal grasps."""

        image = sample["image"]
        quality = sample["quality"]
        angle_cos = sample["angle_cos"]
        angle_sin = sample["angle_sin"]
        width = sample["width"]

        # Horizontal or vertical flips negate sin(2 theta) and keep cos(2 theta).
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[2])
            quality = torch.flip(quality, dims=[2])
            angle_cos = torch.flip(angle_cos, dims=[2])
            angle_sin = -torch.flip(angle_sin, dims=[2])
            width = torch.flip(width, dims=[2])
        if torch.rand(()) < 0.5:
            image = torch.flip(image, dims=[1])
            quality = torch.flip(quality, dims=[1])
            angle_cos = torch.flip(angle_cos, dims=[1])
            angle_sin = -torch.flip(angle_sin, dims=[1])
            width = torch.flip(width, dims=[1])

        # 90-degree rotations shift theta by k*pi/2, flipping both cos/sin for odd k.
        k = int(torch.randint(0, 4, ()).item())
        if k:
            image = torch.rot90(image, k=k, dims=[1, 2])
            quality = torch.rot90(quality, k=k, dims=[1, 2])
            angle_cos = torch.rot90(angle_cos, k=k, dims=[1, 2])
            angle_sin = torch.rot90(angle_sin, k=k, dims=[1, 2])
            if k % 2:
                angle_cos = -angle_cos
                angle_sin = -angle_sin
            width = torch.rot90(width, k=k, dims=[1, 2])

        sample["image"] = image
        sample["quality"] = quality
        sample["angle_cos"] = angle_cos
        sample["angle_sin"] = angle_sin
        sample["width"] = width
        return sample


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    """Normalize depth to [0, 1] per image using robust percentiles."""

    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth[depth > 0]
    if valid.size == 0:
        valid = depth.reshape(-1)
    lo, hi = np.percentile(valid, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return depth.astype(np.float32)
