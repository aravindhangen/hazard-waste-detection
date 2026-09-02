Run 4 — YOLOv5s-Seg academic baseline / production model.

Train:
  python scripts/training/13_train_yolov5_run4.py --device 0

Evaluate only (after training):
  python scripts/training/13_train_yolov5_run4.py --skip-train --device 0

Resume:
  python scripts/training/13_train_yolov5_run4.py --device 0 --resume

Outputs:
  runs/yolov5s_run4/weights/best_yolov5s.pt
  runs/yolov5s_run4/evaluation/run4_evaluation_report.json

Then refresh comparison metrics:
  python scripts/training/12_compare_all_runs.py
