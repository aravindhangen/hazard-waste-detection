Run 2 — YOLO11s-Seg experimental challenger
===========================================

This directory is for OPTIONAL model comparison only.
Run 4 (YOLOv5s) is the production model for deployment.

Frozen inputs (shared with all runs)
-----------------------------------
  hazard_dataset_clean/   393 images, 275/79/39 split

Run 2 outputs (isolated)
------------------------
  train/                  Ultralytics training artifacts
  weights/best_yolo11s.pt Copied best checkpoint after training
  evaluation/             Run 2 val/test reports

Commands
--------
  pip install -r requirements-run2.txt
  python 09_train_yolo11_run2.py --device 0
  python 09_train_yolo11_run2.py --device 0 --skip-train   # eval only
  python scripts/training/12_compare_all_runs.py

Do NOT overwrite
----------------
  runs/yolov5s_run4/weights/best_yolov5s.pt
  api/ or dashboard/
