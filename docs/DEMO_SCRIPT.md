# Demo Rehearsal Script

Use this script for presentation practice. **Do not rely on webcam alone** — keep prepared images as backup.

**Dashboard:** http://127.0.0.1:8000/dashboard/  
**Start server:** `.\run_server.ps1`

---

## Prepared test images

All paths relative to project root: `hazard_dataset_clean/images/test/`

| Scenario | Filename | Ground truth |
|----------|----------|--------------|
| Clear cylinder | `scrap_Screenshot-2025-02-16-235555_png_png.rf.7080243146f9569ecf603df1b410b26e.jpg` | Cylinder |
| Multiple cylinders | `scrap_Screenshot-2025-02-16-235240_png_png.rf.8d46165a65f6509b22cc8cfb27efa157.jpg` | 2+ Cylinders |
| Shock absorber | `shock_Screenshot-2025-02-16-221206_png_png.rf.ea6981effb5404d65e7618e57c69a37d.jpg` | Shock Absorber |
| Mixed scrap | `scrap_Screenshot-2025-02-17-050237_png_png.rf.3ac3768175bc86757383ffe71c9bec95.jpg` | Cylinder + Shock |
| Busy scene | `scrap_Screenshot-2025-02-09-215322_png_png.rf.1c51e717893cc8d4ea3019ed7e8b770a.jpg` | Multiple objects |

Copy these to a `demo/` folder before presentation if you want shorter paths:

```powershell
New-Item -ItemType Directory -Force demo
Copy-Item "hazard_dataset_clean\images\test\scrap_Screenshot-2025-02-16-235555_png_png.rf.7080243146f9569ecf603df1b410b26e.jpg" demo\01_clear_cylinder.jpg
Copy-Item "hazard_dataset_clean\images\test\scrap_Screenshot-2025-02-16-235240_png_png.rf.8d46165a65f6509b22cc8cfb27efa157.jpg" demo\02_multiple_cylinders.jpg
Copy-Item "hazard_dataset_clean\images\test\shock_Screenshot-2025-02-16-221206_png_png.rf.ea6981effb5404d65e7618e57c69a37d.jpg" demo\03_shock_absorber.jpg
Copy-Item "hazard_dataset_clean\images\test\scrap_Screenshot-2025-02-17-050237_png_png.rf.3ac3768175bc86757383ffe71c9bec95.jpg" demo\04_mixed_scrap.jpg
Copy-Item "hazard_dataset_clean\images\test\scrap_Screenshot-2025-02-09-215322_png_png.rf.1c51e717893cc8d4ea3019ed7e8b770a.jpg" demo\05_busy_scene.jpg
```

---

## Demo sequence (~5 minutes)

### 1. Introduction (30 s)

> "This system detects hazardous cylinders and shock absorbers in scrap-yard images using instance segmentation."

### 2. Upload — Clear cylinder (60 s)

- Mode: **Single Model** → YOLOv9
- Upload `01_clear_cylinder.jpg`
- Point out: mask overlay, confidence score, **EXPLOSIVE** hazard badge

### 3. Upload — Mixed / busy scene (60 s)

- Upload `04_mixed_scrap.jpg` or `05_busy_scene.jpg`
- Show multiple detections and hazard summary list

### 4. Compare Models (90 s)

- Switch to **Compare Models**
- Ensure YOLOv9 + YOLO11s checked
- Upload same image again
- Explain: same image, same threshold, different model outputs
- Reference benchmark table: YOLOv9 higher mAP@50 and Cylinder recall

### 5. Benchmark table (45 s)

| Model | Status | mAP@50 | Cylinder Recall | FPS |
|-------|--------|-------:|----------------:|----:|
| YOLOv9 | Production | 0.732 | 0.734 | 22.9 |
| YOLO11s | Tested | 0.643 | 0.714 | 26.6 |
| YOLOv8s | Candidate | — | — | — |

> "YOLOv8s is a considered baseline but was not trained — no metrics shown."

### 6. Webcam (optional, 60 s)

- Only if camera works reliably
- If it fails → say "backup demo uses prepared images" and continue with upload

### 7. Architecture (30 s)

- Production: Dashboard → FastAPI → YOLOv9 only
- Experimental: comparison runs through model manager for demo purposes

---

## Talking points if asked

**Why YOLOv9 over YOLO11s?**  
Higher cylinder recall (0.734 vs 0.714) and mAP@50 (0.732 vs 0.643). Speed difference (~4 FPS) is not worth the recall drop for safety-critical detection.

**Why not YOLOv8?**  
Considered but not experimentally trained. Two-model comparison was sufficient to justify YOLOv9.

**Is this production-ready?**  
Academic prototype. Human oversight still required — 26% cylinder miss rate on test set.

**Dataset size?**  
393 unique images after dedup. Test set is only 39 images — metrics have uncertainty.

---

## Pre-demo checklist

- [ ] Server running (`.\run_server.ps1`)
- [ ] `demo/` folder with 5 images copied
- [ ] Browser open to dashboard
- [ ] Confidence threshold at 0.25
- [ ] GPU available (or CPU mode for Docker)
- [ ] Webcam tested once; prepared images ready as fallback
