#!/usr/bin/env python3
"""
black_and_white_launcher.py
מריץ את האלגוריתם של mark_balls_v4 ומחזיר רשימת Ball
"""

import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "black_white_detect"))

from analyzer_table.black_white_detect.mark_balls_v4 import detect_balls_full_pipeline
from analyzer_table.launcher_helper.json_models import Ball


def run_ball_detection(image_path: str):
    """
    מקבל תמונה כקלט (נתיב) ומחזיר list של Ball
    """
    # טוען את התמונה
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"❌ לא נמצא קובץ תמונה: {image_path}")

    # ממיר ל־HSV ו־GRAY (לפי הצורך של הפונקציה)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = (80, 40, 30)
    upper_blue = (125, 255, 255)
    mask_felt = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_inv = cv2.bitwise_not(mask_felt)
    _, mask_bin = cv2.threshold(mask_inv, 127, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    # מפעיל את פונקציית הזיהוי מהמודול שלך
    balls: list[Ball] = detect_balls_full_pipeline(image_path)

    # הדפסה קצרה
    print(f"✅ Detected {len(balls)} balls:")
    for i, b in enumerate(balls, start=1):
        print(f"{i:02d}. Center={b.center}, Radius={b.radius}")

    return balls


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python black_and_white_launcher.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    balls = run_ball_detection(image_path)
