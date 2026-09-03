# Hazardous Waste Detection — Presentation (12 Slides)

---

## Slide 1 — Title

**Hazardous Waste Detection in Scrap Yards**

Instance Segmentation with YOLOv5s-Seg

*Your Name · Institution · Date*

---

## Slide 2 — Problem Statement

- Scrap yards contain **pressurized cylinders** and **shock absorbers** with hydraulic oil
- Manual sorting is slow, inconsistent, and hazardous
- Need automated **detection + segmentation** before compression/recycling
- Missing a cylinder can cause **explosion risk** under compaction

---

## Slide 3 — Objectives

1. Build a clean, leakage-free segmentation dataset
2. Train and compare YOLO segmentation models
3. Prioritize **Cylinder recall** (safety-critical)
4. Deploy a working API + dashboard demo

---

## Slide 4 — Proposed Solution

```text
Camera / Upload → Dashboard → FastAPI → YOLOv5s-Seg → Masks → Hazard Alert
```

- **Instance segmentation** (not just bounding boxes)
- Two classes: **Cylinder** (explosive), **Shock Absorber** (toxic)
- Real-time inference on consumer GPU (~63 FPS for YOLOv5s)

---

## Slide 5 — Dataset: 512 → 393

| Stage | Count |
|-------|------:|
| Raw merged (Roboflow) | 512 |
| After deduplication | **393** |
| Train / Val / Test | **275 / 79 / 39** |

- Two Roboflow exports merged
- Perceptual-hash deduplication
- Stratified split, zero cross-split leakage

---

## Slide 6 — Data Cleaning & Leakage Prevention

✅ 0 duplicate images across splits  
✅ 0 malformed polygon labels  
✅ Image–label pairing validated  
✅ Class balance ~1.17:1 (no oversampling)  
✅ Frozen dataset: `hazard_dataset_clean/`

---

## Slide 7 — Model Selection

**Considered (theoretical):** Mask R-CNN, RT-DETR

**Experimentally benchmarked (same frozen test set):**

- YOLOv5s-Seg (Run 4) — **production**
- YOLO11s-Seg (Run 2)
- YOLOv8s-Seg (Run 3)

Selection based on **held-out test set**, deployment constraints, and safety-oriented criteria.

---

## Slide 8 — Experimental Model Comparison

> **YOLOv8s achieved the highest test accuracy; YOLOv5s was selected for production deployment.**

| Model | mAP@50 | Cylinder Recall | FPS | Role |
|-------|-------:|----------------:|----:|------|
| **YOLOv5s** | 58.9% | 71.0% | **63.3** | **Production** |
| YOLO11s | 64.3% | 71.4% | 26.6 | Tested |
| YOLOv8s | **65.6%** | **72.0%** | 68.7 | Tested |

**Selection priority:** Cylinder Recall → mAP@50 → Recall → Deployability

YOLOv5s provides a standard academic baseline with compact weights and strong cloud CPU performance.

---

## Slide 9 — System Architecture

**Production path:** Dashboard → FastAPI → YOLOv5s `best_yolov5s.pt` → Hazard Alert

**Comparison path:** Same frozen test set → YOLOv5s vs YOLO11s vs YOLOv8s → Benchmark table

*(Use architecture diagram from `docs/FINAL_REPORT.md` Section 7)*

---

## Slide 10 — Live Dashboard Demo

**Demo flow:**

1. Upload prepared test image
2. YOLOv5s detection + segmentation masks
3. Confidence scores + hazard classification
4. Switch to **Compare Models** → all three models side-by-side
5. Show benchmark table

**URL:** `http://localhost:8000/dashboard/`

---

## Slide 11 — Limitations & Future Work

- YOLOv5s Cylinder recall 71.0% — human oversight still required
- Dataset domain-specific (scrap-yard screenshots)
- Future: more data, optional YOLOv8s for accuracy-critical edge deployments

---

## Slide 12 — Conclusion

- Built end-to-end hazardous-waste detection prototype
- **393-image frozen dataset** with rigorous QA
- **YOLOv5s deployed** with YOLO11s and YOLOv8s for academic comparison
- Deployed API + dashboard with live inference and model comparison
- Ready for Docker / Render deployment and academic evaluation

**Production model:** YOLOv5s-Seg · mAP@50 **0.589** · Cylinder recall **0.710**
