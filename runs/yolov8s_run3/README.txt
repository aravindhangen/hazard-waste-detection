Run 3 — YOLOv8s-Seg experimental baseline.

Train:
  python scripts/training/11_train_yolov8_run3.py --device 0

Evaluate only (after training):
  python scripts/training/11_train_yolov8_run3.py --skip-train --device 0

Resume:
  python scripts/training/11_train_yolov8_run3.py --device 0 --resume

Outputs:
  runs/yolov8s_run3/weights/best_yolov8s.pt
  runs/yolov8s_run3/evaluation/run3_evaluation_report.txt

Does NOT modify Run 4 (YOLOv5s production) artifacts.
