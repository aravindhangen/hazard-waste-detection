# Hazardous Waste Detection — Presentation (12 Slides)

---

## Slide 1 — Title

**Hazardous Waste Detection in Scrap Yards**

Instance Segmentation with YOLOv9 GELAN-C-SEG

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
Camera / Upload → Dashboard → FastAPI → YOLOv9-Seg → Masks → Hazard Alert
```

- **Instance segmentation** (not just bounding boxes)
- Two classes: **Cylinder** (explosive), **Shock Absorber** (toxic)
- Real-time inference on consumer GPU (~23 FPS)

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

**Considered (theoretical):** Mask R-CNN, YOLOv5-Seg, RT-DETR

**Experimentally benchmarked (same frozen test set):**

- YOLOv8s-Seg (Run 3)
- YOLOv9 GELAN-C-SEG (Run 1)
- YOLO11s-Seg (Run 2)

Selection based on **held-out test set** and safety-oriented criteria, not model novelty.

---

## Slide 8 — Experimental Model Comparison

> **YOLOv9 achieved the best safety-oriented detection performance.**

| Model | mAP@50 | Cylinder Recall | FPS |
|-------|-------:|----------------:|----:|
| YOLOv8s | 65.6% | 72.0% | **68.7** |
| **YOLOv9** | **73.2%** | **73.4%** | 22.9 **🏆** |
| YOLO11s | 64.3% | 71.4% | 26.6 |

**Selection priority:** Cylinder Recall → mAP@50 → Recall → Speed

Although YOLOv8s exceeded the 30 FPS design target, YOLOv9 was selected because hazardous-cylinder detection reliability is the primary requirement.

---

## Slide 9 — System Architecture

**Production path:** Dashboard → FastAPI → YOLOv9 `best.pt` → Hazard Alert

**Experimental path:** Same frozen test set → YOLOv8s vs YOLOv9 vs YOLO11s → Benchmark comparison

*(Use architecture diagram from `docs/FINAL_REPORT.md` Section 7)*

---

## Slide 10 — Live Dashboard Demo

**Demo flow:**

1. Upload prepared test image
2. YOLOv9 detection + segmentation masks
3. Confidence scores + hazard classification
4. Switch to **Compare Models** → all three models side-by-side
5. Show benchmark table

**URL:** `http://localhost:8000/dashboard/`

---

## Slide 11 — Limitations & Future Work

- Measured YOLOv9 FPS ~22.9 (below 30 FPS design target; YOLOv8s reached 68.7 FPS)
- Dataset domain-specific (scrap-yard screenshots)
- Future: more data, edge deployment with speed-optimized models where appropriate

---

## Slide 12 — Conclusion

- Built end-to-end hazardous-waste detection prototype
- **393-image frozen dataset** with rigorous QA
- **YOLOv9 selected** after three-model experimental comparison (YOLOv8s, YOLOv9, YOLO11s)
- Deployed API + dashboard with live inference and model comparison
- Ready for Docker deployment and academic evaluation

**Production model:** YOLOv9 GELAN-C-SEG · mAP@50 **0.732** · Cylinder recall **0.734**
