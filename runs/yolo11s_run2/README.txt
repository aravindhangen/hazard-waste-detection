Run 2 — YOLO11s-Seg experimental challenger
===========================================

This directory is for OPTIONAL model comparison only.
Run 1 (YOLOv9) remains the official baseline for submission and deployment.

Frozen inputs (shared with Run 1)
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
  python 10_compare_runs.py

Do NOT overwrite
----------------
  yolov9/runs/train-seg/hazard_waste_seg/weights/best.pt
  evaluation_reports/final_evaluation_report.txt
  api/ or dashboard/
