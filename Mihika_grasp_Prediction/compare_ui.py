"""Compare old and improved grasp checkpoints on one uploaded image."""

from __future__ import annotations

import argparse
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import torch
from PIL import Image, ImageTk

from test_ui import build_input_tensor, draw_prediction, load_model, resize_for_display
from utils.geometry import rectangle_from_grasp
from utils.prediction import grasp_from_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two grasp prediction checkpoints.")
    parser.add_argument("--old_checkpoint", required=True, help="Path to old baseline checkpoint.")
    parser.add_argument("--new_checkpoint", default="checkpoints/best.pt", help="Path to improved checkpoint.")
    parser.add_argument("--max_display_size", type=int, default=420, help="Maximum display size per image.")
    return parser.parse_args()


class CheckpointRunner:
    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.model, self.config, self.device = load_model(checkpoint_path)

    @property
    def input_mode(self) -> str:
        return str(self.config["data"].get("input_mode", "rgbd"))

    @property
    def image_size(self) -> int:
        return int(self.config["data"].get("image_size", 300))

    def predict(
        self,
        rgb_image: Image.Image | None,
        depth_image: Image.Image | None,
    ) -> tuple[Image.Image, dict[str, float]]:
        tensor = build_input_tensor(rgb_image, depth_image, self.input_mode, self.image_size).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)

        center_x, center_y, angle, width_pixels, score = grasp_from_output(outputs, self.image_size)
        pred_rect = rectangle_from_grasp(center_x, center_y, angle, width_pixels)
        base_image = rgb_image if rgb_image is not None else depth_image.convert("RGB")
        overlay = draw_prediction(base_image, pred_rect, self.image_size)
        return overlay, {
            "score": score,
            "center_x": center_x,
            "center_y": center_y,
            "angle_deg": math.degrees(angle),
            "width_pixels": width_pixels,
        }


class CompareUI:
    def __init__(
        self,
        root: tk.Tk,
        old_runner: CheckpointRunner,
        new_runner: CheckpointRunner,
        max_display_size: int,
    ) -> None:
        self.root = root
        self.old_runner = old_runner
        self.new_runner = new_runner
        self.max_display_size = max_display_size
        self.rgb_path: Path | None = None
        self.depth_path: Path | None = None
        self.old_photo: ImageTk.PhotoImage | None = None
        self.new_photo: ImageTk.PhotoImage | None = None

        self.root.title("Old vs Improved Grasp Prediction")
        self.root.geometry("980x760")

        toolbar = tk.Frame(root)
        toolbar.pack(fill=tk.X, padx=12, pady=10)
        tk.Button(toolbar, text="Upload RGB Image", command=self.select_rgb).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(toolbar, text="Upload Depth Image (Optional)", command=self.select_depth).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(toolbar, text="Compare Predictions", command=self.compare).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Upload one RGB image, then compare both checkpoints.")
        tk.Label(root, textvariable=self.status_var, anchor="w").pack(fill=tk.X, padx=12)

        image_frame = tk.Frame(root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        old_panel = tk.Frame(image_frame)
        old_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        new_panel = tk.Frame(image_frame)
        new_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        tk.Label(old_panel, text="Old baseline checkpoint").pack(anchor="w")
        self.old_image_label = tk.Label(old_panel, bg="#222222")
        self.old_image_label.pack(fill=tk.BOTH, expand=True)
        self.old_metrics = tk.StringVar(value="")
        tk.Label(old_panel, textvariable=self.old_metrics, anchor="w", justify=tk.LEFT, font=("Consolas", 9)).pack(fill=tk.X)

        tk.Label(new_panel, text="Improved checkpoint").pack(anchor="w")
        self.new_image_label = tk.Label(new_panel, bg="#222222")
        self.new_image_label.pack(fill=tk.BOTH, expand=True)
        self.new_metrics = tk.StringVar(value="")
        tk.Label(new_panel, textvariable=self.new_metrics, anchor="w", justify=tk.LEFT, font=("Consolas", 9)).pack(fill=tk.X)

    def select_rgb(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose RGB image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.rgb_path = Path(path)
            self.status_var.set(f"RGB image: {self.rgb_path}")

    def select_depth(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose depth image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.depth_path = Path(path)
            self.status_var.set(f"Depth image: {self.depth_path}")

    def compare(self) -> None:
        if self.rgb_path is None and (self.old_runner.input_mode != "depth" or self.new_runner.input_mode != "depth"):
            messagebox.showerror("Missing RGB Image", "Please upload an RGB image first.")
            return

        try:
            rgb_image = Image.open(self.rgb_path).convert("RGB") if self.rgb_path else None
            depth_image = Image.open(self.depth_path) if self.depth_path else None

            old_overlay, old_metrics = self.old_runner.predict(rgb_image, depth_image)
            new_overlay, new_metrics = self.new_runner.predict(rgb_image, depth_image)

            old_display = resize_for_display(old_overlay, self.max_display_size)
            new_display = resize_for_display(new_overlay, self.max_display_size)
            self.old_photo = ImageTk.PhotoImage(old_display)
            self.new_photo = ImageTk.PhotoImage(new_display)
            self.old_image_label.configure(image=self.old_photo)
            self.new_image_label.configure(image=self.new_photo)
            self.old_metrics.set(format_metrics(old_metrics, self.old_runner))
            self.new_metrics.set(format_metrics(new_metrics, self.new_runner))
            self.status_var.set("Comparison complete. Red rectangles show each checkpoint prediction.")
        except Exception as exc:
            messagebox.showerror("Comparison Failed", str(exc))


def format_metrics(metrics: dict[str, float], runner: CheckpointRunner) -> str:
    depth_note = ""
    if runner.input_mode == "rgbd":
        depth_note = "\nRGB-D model: use a matching depth image for best results."
    return (
        f"checkpoint: {runner.checkpoint_path}\n"
        f"score: {metrics['score']:.4f}\n"
        f"center: ({metrics['center_x']:.1f}, {metrics['center_y']:.1f})\n"
        f"angle: {metrics['angle_deg']:.2f} deg\n"
        f"width: {metrics['width_pixels']:.1f}px"
        f"{depth_note}"
    )


def main() -> None:
    args = parse_args()
    old_path = Path(args.old_checkpoint)
    new_path = Path(args.new_checkpoint)
    if old_path.resolve() == new_path.resolve():
        raise ValueError("old_checkpoint and new_checkpoint must be different files.")

    old_runner = CheckpointRunner(old_path)
    new_runner = CheckpointRunner(new_path)
    root = tk.Tk()
    CompareUI(root, old_runner, new_runner, args.max_display_size)
    root.mainloop()


if __name__ == "__main__":
    main()
