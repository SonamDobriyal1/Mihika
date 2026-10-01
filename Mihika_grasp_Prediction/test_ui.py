"""Simple desktop UI for testing grasp predictions on uploaded images.

The trained baseline normally expects RGB-D input. This UI lets the user upload
an RGB image and, optionally, a matching depth image. If no depth image is
provided for an RGB-D checkpoint, a neutral zero-depth channel is used so the UI
can still produce a prediction.
"""

from __future__ import annotations

import argparse
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk

from models import GGCNNBaseline
from utils.prediction import grasp_from_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload an image and visualize grasp prediction.")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--max_display_size",
        type=int,
        default=720,
        help="Maximum displayed image width/height in pixels.",
    )
    return parser.parse_args()


def checkpoint_angle_representation(checkpoint: dict[str, Any]) -> str:
    state = checkpoint.get("model_state", {})
    if "angle_head.weight" in state:
        return "direct"
    return "cos_sin"


def load_model(checkpoint_path: Path) -> tuple[GGCNNBaseline, dict[str, Any], torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint does not contain config.")

    input_mode = config["data"].get("input_mode", "rgbd")
    in_channels = {"rgb": 3, "depth": 1, "rgbd": 4}[input_mode]
    model = GGCNNBaseline(
        in_channels=in_channels,
        angle_representation=checkpoint_angle_representation(checkpoint),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, device


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = depth[depth > 0]
    if valid.size == 0:
        valid = depth.reshape(-1)
    lo, hi = np.percentile(valid, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_input_tensor(
    rgb_image: Image.Image | None,
    depth_image: Image.Image | None,
    input_mode: str,
    image_size: int,
) -> torch.Tensor:
    if input_mode in {"rgb", "rgbd"}:
        if rgb_image is None:
            raise ValueError("This checkpoint requires an RGB image.")
        rgb = np.asarray(
            rgb_image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        rgb = rgb / 255.0
        rgb_chw = np.transpose((rgb - 0.5) / 0.5, (2, 0, 1))

    if input_mode in {"depth", "rgbd"}:
        if depth_image is None:
            depth = np.zeros((image_size, image_size), dtype=np.float32)
        else:
            depth_raw = np.asarray(
                depth_image.resize((image_size, image_size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            if depth_raw.ndim == 3:
                depth_raw = depth_raw[..., 0]
            depth = normalize_depth(depth_raw)
        depth_chw = ((depth - 0.5) / 0.5)[None, ...]

    if input_mode == "rgb":
        image = rgb_chw
    elif input_mode == "depth":
        image = depth_chw
    else:
        image = np.concatenate([rgb_chw, depth_chw], axis=0)
    return torch.from_numpy(image.astype(np.float32)).unsqueeze(0)


def draw_prediction(
    display_image: Image.Image,
    pred_rect: np.ndarray,
    model_image_size: int,
) -> Image.Image:
    output = display_image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    sx = output.width / float(model_image_size)
    sy = output.height / float(model_image_size)
    points = [(float(x) * sx, float(y) * sy) for x, y in pred_rect]
    closed = points + [points[0]]
    draw.line(closed, fill=(255, 0, 0), width=max(3, output.width // 180))
    for x, y in points:
        radius = max(3, output.width // 240)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))
    return output


def resize_for_display(image: Image.Image, max_size: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(max_size / image.width, max_size / image.height, 1.0)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.BILINEAR)


class GraspPredictionUI:
    def __init__(
        self,
        root: tk.Tk,
        model: GGCNNBaseline,
        config: dict[str, Any],
        device: torch.device,
        max_display_size: int,
    ) -> None:
        self.root = root
        self.model = model
        self.config = config
        self.device = device
        self.max_display_size = max_display_size
        self.rgb_path: Path | None = None
        self.depth_path: Path | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.root.title("Grasp Prediction Test UI")
        self.root.geometry("900x820")

        toolbar = tk.Frame(root)
        toolbar.pack(fill=tk.X, padx=12, pady=10)

        tk.Button(toolbar, text="Upload RGB Image", command=self.select_rgb).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(toolbar, text="Upload Depth Image (Optional)", command=self.select_depth).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(toolbar, text="Run Prediction", command=self.run_prediction).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Upload an RGB image to begin.")
        tk.Label(root, textvariable=self.status_var, anchor="w", justify=tk.LEFT).pack(fill=tk.X, padx=12)

        self.metrics_var = tk.StringVar(value="")
        tk.Label(root, textvariable=self.metrics_var, anchor="w", justify=tk.LEFT, font=("Consolas", 10)).pack(
            fill=tk.X,
            padx=12,
            pady=(4, 8),
        )

        self.preview_label = tk.Label(root, bg="#222222")
        self.preview_label.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    def select_rgb(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose RGB image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.rgb_path = Path(path)
        self.show_image(Image.open(self.rgb_path))
        self.status_var.set(f"RGB image: {self.rgb_path}")

    def select_depth(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose depth image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.depth_path = Path(path)
        self.status_var.set(f"Depth image: {self.depth_path}")

    def show_image(self, image: Image.Image) -> None:
        display = resize_for_display(image, self.max_display_size)
        self.preview_photo = ImageTk.PhotoImage(display)
        self.preview_label.configure(image=self.preview_photo)

    def run_prediction(self) -> None:
        input_mode = self.config["data"].get("input_mode", "rgbd")
        image_size = int(self.config["data"].get("image_size", 300))

        if input_mode in {"rgb", "rgbd"} and self.rgb_path is None:
            messagebox.showerror("Missing RGB Image", "Please upload an RGB image first.")
            return
        if input_mode == "depth" and self.depth_path is None:
            messagebox.showerror("Missing Depth Image", "This checkpoint requires a depth image.")
            return

        try:
            rgb_image = Image.open(self.rgb_path).convert("RGB") if self.rgb_path else None
            depth_image = Image.open(self.depth_path) if self.depth_path else None
            tensor = build_input_tensor(rgb_image, depth_image, input_mode, image_size).to(self.device)

            with torch.no_grad():
                outputs = self.model(tensor)

            center_x, center_y, angle, width_pixels, score = grasp_from_output(outputs, image_size)
            from utils.geometry import rectangle_from_grasp

            pred_rect = rectangle_from_grasp(center_x, center_y, angle, width_pixels)
            base_image = rgb_image if rgb_image is not None else depth_image.convert("RGB")
            overlay = draw_prediction(resize_for_display(base_image, self.max_display_size), pred_rect, image_size)
            self.show_image(overlay)

            depth_note = ""
            if input_mode == "rgbd" and self.depth_path is None:
                depth_note = " | depth: neutral fallback used"
            self.metrics_var.set(
                f"score={score:.4f} | center=({center_x:.1f}, {center_y:.1f}) | "
                f"angle={math.degrees(angle):.2f} deg | width={width_pixels:.1f}px{depth_note}"
            )
            self.status_var.set("Prediction complete. Red rectangle shows predicted grasp.")
        except Exception as exc:
            messagebox.showerror("Prediction Failed", str(exc))


def main() -> None:
    args = parse_args()
    model, config, device = load_model(Path(args.checkpoint))
    root = tk.Tk()
    GraspPredictionUI(root, model, config, device, args.max_display_size)
    root.mainloop()


if __name__ == "__main__":
    main()
