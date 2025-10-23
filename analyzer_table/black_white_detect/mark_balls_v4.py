#!/usr/bin/env python3
"""
🎯 Full pipeline for automatic billiard ball detection.
Steps:
1️⃣ Generate blue-felt mask
2️⃣ Detect connected components (CC)
3️⃣ Refine with HoughCircles
4️⃣ Draw green circles and labels on the original image
5️⃣ Save debug & output files
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analyzer_table.launcher_helper.json_models import Ball


# ================= Helper Functions ================= #

def preprocess_roi(roi_gray):
    """מחזק ניגודיות בתמונת ROI"""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq = clahe.apply(roi_gray)
    den = cv2.bilateralFilter(eq, d=7, sigmaColor=50, sigmaSpace=7)
    blur = cv2.GaussianBlur(den, (5, 5), 0)
    return blur


def edge_support_ratio(edges, cx, cy, r, step_deg=10, tol=2.5):
    """בודק כמה מהנקודות על ההיקף באמת נופלות על קצוות"""
    H, W = edges.shape[:2]
    ok, total = 0, 0
    for a in range(0, 360, step_deg):
        total += 1
        rad = np.deg2rad(a)
        x = int(cx + r * np.cos(rad))
        y = int(cy + r * np.sin(rad))
        x1, y1 = max(0, x - int(tol)), max(0, y - int(tol))
        x2, y2 = min(W - 1, x + int(tol)), min(H - 1, y + int(tol))
        if edges[y1:y2 + 1, x1:x2 + 1].max() > 0:
            ok += 1
    return ok / max(1, total)


def refine_with_hough(gray, x, y, w, h, pad=20):
    """מריץ HoughCircles בתוך ROI ומחזיר (cx, cy, r) אם נמצא עיגול תקין"""
    H, W = gray.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    roi_prep = preprocess_roi(roi)
    edges = cv2.Canny(roi_prep, 60, 160)
    r_est = 0.5 * min(w, h)

    def try_hough(p2, r_lo_mul, r_hi_mul):
        min_r = max(6, int(r_lo_mul * r_est))
        max_r = max(min_r + 2, int(r_hi_mul * r_est))
        return cv2.HoughCircles(
            roi_prep, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=max(10, int(0.8 * r_est)),
            param1=120, param2=p2,
            minRadius=min_r, maxRadius=max_r
        )

    circles = try_hough(22, 0.7, 1.35)
    if circles is None:
        circles = try_hough(18, 0.45, 1.9)
    if circles is None:
        return None

    best, best_score = None, -1.0
    for c in circles[0]:
        cx, cy, r = c
        if r < 0.4 * r_est:  # כנראה מספר על הכדור
            continue
        try:    
            cov = edge_support_ratio(edges, cx, cy, r)
        except:
            cov = 0

            
        score = cov * r
        if score > best_score:
            best_score = score
            best = (cx, cy, r)

    if best is None:
        return None

    cx, cy, r = best
    return int(x1 + cx), int(y1 + cy), int(r)


def touches_border(bbox, w, h, pad=3):
    x, y, bw, bh = bbox
    return x <= pad or y <= pad or (x + bw) >= (w - 1 - pad) or (y + bh) >= (h - 1 - pad)


def balls_from_cc(mask_bin, gray):
    """מאתר בלובים לבנים במסכה, מריץ Hough refining ומחזיר [(cx, cy, r), ...]"""
    h, w = mask_bin.shape[:2]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    balls = []
    img_area = h * w
    min_area_cc = int(0.00015 * img_area)
    max_area_cc = int(0.0085 * img_area)

    for label in range(1, num):
        x, y, bw, bh, area = stats[label]
        if area < min_area_cc or area > max_area_cc:
            continue
        if touches_border((x, y, bw, bh), w, h):
            continue

        ref = refine_with_hough(gray, x, y, bw, bh)
        if ref:
            balls.append(ref)
        else:
            cx, cy = centroids[label]
            rr = int(0.5 * (bw + bh) / 2)
            balls.append((int(cx), int(cy), max(6, rr)))

    return sorted(balls, key=lambda item: item[2], reverse=True)


def detect_balls_as_dataclasses(mask_bin, gray) -> List[Ball]:
    """ממיר את רשימת ה-(cx,cy,r) למחלקות Ball"""
    raw_balls = balls_from_cc(mask_bin, gray)
    return [Ball(center=(int(cx), int(cy)), radius=int(r)) for (cx, cy, r) in raw_balls]


# ================= Main Pipeline ================= #

def detect_balls_full_pipeline(input_path: str):
    """
    ⚙️ Detects all balls from a single image.
    Returns: List[Ball]
    Saves:
      - /debug/01_felt_mask.jpg
      - /output/output_marked_balls.jpg
    """
    BASE = Path(__file__).resolve().parent
    DEBUG_DIR = BASE / "debug"
    OUTPUT_DIR = BASE / "output"
    MASK_OUTPUT_PATH = DEBUG_DIR / "01_felt_mask.jpg"
    OUTPUT_PATH = OUTPUT_DIR / "output_marked_balls.jpg"

    DEBUG_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    orig = cv2.imread(input_path)
    if orig is None:
        raise FileNotFoundError(f"❌ Could not read input image: {input_path}")

    # === Create felt mask ===
    hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([80, 40, 30])
    upper_blue = np.array([125, 255, 255])
    mask_felt = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_inv = cv2.bitwise_not(mask_felt)
    _, mask_bin = cv2.threshold(mask_inv, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(str(MASK_OUTPUT_PATH), mask_bin)

    print(f"🖤 Mask saved to: {MASK_OUTPUT_PATH}")

    # === Detect balls ===
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    ball_objects = detect_balls_as_dataclasses(mask_bin, gray)
    print(f"🎱 Found {len(ball_objects)} balls.")

    # === Draw results ===
    out = orig.copy()
    for b in ball_objects:
        cx, cy = b.center
        r = b.radius
        draw_r = max(8, int(r))
        thick = max(2, draw_r // 5)
        cv2.circle(out, (cx, cy), draw_r, (0, 255, 0), thick)
        cv2.circle(out, (cx, cy), max(3, draw_r // 6), (0, 0, 255), -1)
        label = f"({cx},{cy})"
        cv2.putText(out, label, (cx + draw_r + 6, cy - draw_r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(OUTPUT_PATH), out)
    print(f"✅ Final image saved to: {OUTPUT_PATH}")

    return ball_objects


# ================= Run Example ================= #
if __name__ == "__main__":
    BASE = Path(__file__).resolve().parent
    example_image = BASE.parent / "test_image9.jpeg"
    balls = detect_balls_full_pipeline(str(example_image))
    print("[OK] Example finished, first 3 balls:", [b.center for b in balls[:3]])
