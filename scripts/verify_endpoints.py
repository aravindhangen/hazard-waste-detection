"""Verify canonical API endpoints on the configured port (default 8000)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hazard_detection.config import CLEAN_IMAGE_DIR  # noqa: E402

PORT = os.environ.get("PORT", os.environ.get("HAZARD_API_PORT", "8000"))
BASE = f"http://127.0.0.1:{PORT}"

CHECKS = [
    ("GET", "/", {307, 308}, None),
    ("GET", "/dashboard/", {200}, "text/html"),
    ("GET", "/health", {200}, "application/json"),
    ("GET", "/models", {200}, "application/json"),
    ("GET", "/docs", {200}, "text/html"),
    ("GET", "/api", {200}, "application/json"),
]


def main() -> int:
    test_image = CLEAN_IMAGE_DIR / "test"
    images = sorted(test_image.glob("*.jpg"))
    if not images:
        print("No test image found; skipping POST /predict check.")
        images = []

    failed = 0
    with httpx.Client(timeout=120.0, follow_redirects=False) as client:
        for method, path, expected_status, content_type in CHECKS:
            url = f"{BASE}{path}"
            response = client.request(method, url)
            ok = response.status_code in expected_status
            if content_type and ok:
                ok = content_type in response.headers.get("content-type", "")
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {method} {path} -> {response.status_code}")
            if not ok:
                failed += 1

        if images:
            with images[0].open("rb") as handle:
                response = client.post(
                    f"{BASE}/predict",
                    files={"file": ("test.jpg", handle, "image/jpeg")},
                    params={"include_annotated_image": "false", "model_id": "yolov9"},
                )
            ok = response.status_code == 200 and response.json().get("hazard_detected") is not None
            status = "OK" if ok else "FAIL"
            print(f"[{status}] POST /predict -> {response.status_code}")
            if not ok:
                failed += 1

            with images[0].open("rb") as handle:
                response = client.post(
                    f"{BASE}/predict/compare",
                    files={"file": ("test.jpg", handle, "image/jpeg")},
                    params={
                        "include_annotated_image": "false",
                        "model_ids": "yolov9,yolo11s",
                    },
                )
            ok = response.status_code == 200 and len(response.json().get("models", [])) >= 2
            status = "OK" if ok else "FAIL"
            print(f"[{status}] POST /predict/compare -> {response.status_code}")
            if not ok:
                failed += 1

    print(f"\nBase URL: {BASE}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
