# RGB-D Robotic Grasp Prediction MVP

This is a small PyTorch research codebase for vision-based robotic grasp prediction on the Cornell Grasp Dataset. It predicts dense maps for:

- grasp quality
- grasp angle
- grasp gripper opening width

The MVP is intentionally simple: no ROS, no robot hardware, and no simulator dependency. PyBullet can be added later for synthetic data or closed-loop testing.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For NVIDIA GPU training on Windows, install the CUDA-enabled PyTorch build in
the project environment:

```powershell
..\venv\Scripts\python.exe -m pip install --upgrade torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
```

Training uses CUDA automatically when available and retains the configured
batch size and full precision. New GPU-run checkpoints are saved under
`checkpoints_gpu/`, preserving any existing checkpoints.

## Dataset Preparation

Download and extract the Cornell Grasp Dataset, then place or symlink it under:

```text
data/cornell/
```

The loader recursively searches for standard Cornell files:

```text
pcd0100r.png
pcd0100d.tiff
pcd0100cpos.txt
```

If your dataset is somewhere else, update `data.root` in `configs/baseline.yaml`.

## Train

```bash
..\venv\Scripts\python.exe train.py --config configs/baseline.yaml
```

The default split is 70% training, 15% validation, and 15% test. It is
deterministic with seed `42`. Training uses only the training subset, and
validation selects the best checkpoint. The test subset is held out for final
evaluation. Training keeps batch size 8 and the existing full precision setup.

Training writes:

```text
checkpoints_gpu/best.pt
checkpoints_gpu/last.pt
outputs/train_log_gpu.csv
```

## Evaluate

```bash
..\venv\Scripts\python.exe evaluate.py --checkpoint checkpoints_gpu/best.pt --split test
```

Use `--split val` to review validation metrics during development. Keep the
test split for final evaluation after model selection.

Accuracy is counted as correct when the best predicted grasp rectangle passes both:

- rectangle IoU threshold, default `0.25`
- angle threshold, default `30` degrees

These settings live in `configs/baseline.yaml`.

## Visualize

```bash
python visualize.py --checkpoint checkpoints/best.pt --sample_id 0
```

This saves an overlay image to:

```text
outputs/prediction_sample_0.png
```

The red rectangle is the model prediction. Green rectangles are Cornell ground-truth positive grasps.

## Smoke Test

The smoke test uses fake tensors, so it does not require the Cornell dataset:

```bash
python tests/smoke_test.py
```

## Project Structure

```text
data/
  cornell.py          Cornell dataset parser and dense target-map generation
models/
  ggcnn.py            Lightweight GG-CNN-inspired baseline CNN
utils/
  config.py           YAML config helpers
  geometry.py         Rectangle, IoU, and angle utilities
  prediction.py       Converts dense maps into one grasp proposal
configs/
  baseline.yaml       Default experiment config
train.py              Training entry point
evaluate.py           Evaluation entry point
visualize.py          Visualization entry point
tests/smoke_test.py   Minimal fake-data smoke test
```

## Notes

The model predicts grasp angle as `cos(2theta)` and `sin(2theta)` to handle antipodal grasp symmetry and angle wraparound. Width is predicted as a normalized value in `[0, 1]`. Training also supports lightweight square-image augmentation through flips and 90-degree rotations. A stronger next version would add richer random rotations/crops, evaluate multiple seeds or folds, and compare RGB-only, depth-only, and RGB-D ablations.

The Cornell samples are split reproducibly into 70% train, 15% validation, and
15% test using seed 42. Training and best-checkpoint selection use only train
and validation data. Evaluate the held-out test set after training with:

```bash
python evaluate.py --checkpoint checkpoints_gpu/best.pt --split test
```
