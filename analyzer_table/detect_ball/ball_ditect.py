import cv2
import numpy as np
import json
from .Debugger import Debugger
import os
from analyzer_table.launcher_helper.json_models import PhotoData, Origin, Rectangle, Ball  # ✅ נוספו המחלקות החדשות

MIN_BALL_RADIUS = 9
MAX_BALL_RADIUS = 21

def detect_balls_opencv(input_dir, output_dir, parts_info, full_w, full_h):
    Debugger.log("Starting OpenCV detection (with radius filtering)")
    os.makedirs(output_dir, exist_ok=True)
    total_balls = 0

    all_photos = []  # ✅ כאן נשמור את כל אובייקטי PhotoData

    for part_info in parts_info:
        file = part_info["file_name"]
        path = os.path.join(input_dir, file)
        Debugger.log(f"[OpenCV] Processing {path}")

        img = cv2.imread(path)
        if img is None:
            print("❌ התמונה לא נטענה!")
        else:
            height, width = img.shape[:2]
            print(f"📏 Image size: width={width}, height={height}")

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv[:, :, 2] = v
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=25,
            param1=60, param2=18, minRadius=10, maxRadius=90
        )

        found_balls = 0
        balls = []

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for x, y, r in circles[0, :]:
                if not (MIN_BALL_RADIUS <= r <= MAX_BALL_RADIUS):
                    Debugger.warn(f"[OpenCV] ❌ Ignored circle r={r} (out of valid range [{MIN_BALL_RADIUS}, {MAX_BALL_RADIUS}]) in {file}")
                    continue

                cv2.circle(img, (x, y), r, (0, 0, 255), 3)
                cv2.putText(
                    img,
                    f"{r}",
                    (x - 15, y - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                found_balls += 1

                global_x = part_info["origin_x"] + x
                global_y = part_info["origin_y"] + y

                balls.append(Ball(center=(int(global_x), int(global_y)), radius=int(r)))

        # שמירת התמונה
        output_img_path = os.path.join(output_dir, f"detect_{file}")
        cv2.imwrite(output_img_path, img)
        Debugger.log(f"[OpenCV] {found_balls} balls detected, saved {output_img_path}")
        total_balls += found_balls

        # ✅ בניית האובייקט של PhotoData
        x0, y0 = part_info["origin_x"], part_info["origin_y"]
        x1 = x0 + part_info["width"]
        y1 = y0 + part_info["height"]

        photo = PhotoData(
            cut_name=file,
            origin=Origin(x=x0, y=y0),
            rectangle=Rectangle(
                top_left=(x0, y1),
                top_right=(x1, y1),
                bottom_left=(x0, y0),
                bottom_right=(x1, y0)
            ),
            balls=balls
        )

        # ✅ שמירה ל-JSON מתוך המחלקה
        json_path = os.path.join(output_dir, f"{file.replace('.png', '.json')}")
        photo.save_json(json_path)
        Debugger.log(f"[OpenCV] Saved metadata: {json_path}")

        all_photos.append(photo)  # ✅ מוסיפים את האובייקט לרשימה

    Debugger.warn(f"[OpenCV] Total detected balls across all parts: {total_balls}")

    return all_photos  # ✅ מחזיר את כל האובייקטים עם המידע
