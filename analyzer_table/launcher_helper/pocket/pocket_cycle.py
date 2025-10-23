import os
import cv2
import numpy as np
from analyzer_table.launcher_helper.json_models import table_pockets, Pocket
from analyzer_table.detect_ball.Debugger import Debugger
from const_numbers import CROP_HALF_SIZE


# =====================
# 🧩 פונקציות עזר
# =====================

def _load_pocket_image(pocket: Pocket):
    """טוען תמונה של חור ומוודא שהיא תקינה."""
    img_path = pocket.pocket_img_path
    if not os.path.exists(img_path):
        Debugger.log(f"⚠️ Image not found for pocket {pocket.pocket_id}: {img_path}")
        return None, None
    img = cv2.imread(img_path)
    if img is None:
        Debugger.log(f"❌ Failed to load pocket image: {img_path}")
        return None, None
    return img, img_path


def _detect_circle(img) -> tuple[int, int, int] | None:
    """מזהה עיגול בתמונה באמצעות HoughCircles ומחזיר (x, y, r) אם נמצא."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=100,
        param2=15,
        minRadius=5,
        maxRadius=int(min(img.shape[:2]) / 2)
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0, 0]
        return int(x), int(y), int(r)
    return None


def _draw_and_save_circle(img, circle_data, img_path):
    """מסמן עיגול על התמונה ושומר גרסה חדשה עם הסיומת _cycle."""
    base, ext = os.path.splitext(img_path)
    new_path = f"{base}_cycle{ext}"

    if circle_data is not None:
        x, y, r = circle_data
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)  # ירוק – היקף
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)  # אדום – מרכז
        Debugger.log(f"✅ Circle drawn at ({x},{y}), radius={r}")
    else:
        Debugger.log("⚠️ No circle detected — saving original image")

    cv2.imwrite(new_path, img)
    Debugger.log(f"💾 Saved result: {new_path}")
    return new_path


# =====================
# 🎯 פונקציה ראשית
# =====================

def mark_pocket_circles(all_pockets: table_pockets, half_size: int = CROP_HALF_SIZE) -> None:
    Debugger.log("🎱 Starting pocket circle marking...")

    for pocket in all_pockets.pocket_list:
        img, path = _load_pocket_image(pocket)
        if img is None:
            continue

        circle_data = _detect_circle(img)
        new_path = _draw_and_save_circle(img, circle_data, path)
        pocket.pocket_img_path = new_path

        if circle_data is not None:
            local_x, local_y, local_r = circle_data

            # ⚙️ גבולות החיתוך המדויקים כמו ב-analyze_table_pockets
            cx, cy = pocket.pocket_center
            x1, y1 = max(0, cx - half_size), max(0, cy - half_size)

            # ✅ המרכז הגלובלי = נקודת החיתוך + מיקום המעגל בתוך ה-ROI
            global_x = int(x1 + local_x)
            global_y = int(y1 + local_y)

            adjusted_radius = int(local_r * 0.8)

            pocket.pocket_img_cordinates_on_table = (global_x, global_y)
            pocket.pocker_radius = adjusted_radius

            Debugger.log(
                f"📍 Pocket {pocket.pocket_id}: local=({local_x},{local_y}), "
                f"crop_origin=({x1},{y1}) → global=({global_x},{global_y})"
            )

        else:
            pocket.pocket_img_cordinates_on_table = pocket.pocket_center
            Debugger.log(
                f"⚠️ No circle found — using original center for pocket {pocket.pocket_id}: "
                f"{pocket.pocket_center}"
            )

    Debugger.log("✅ Finished marking pocket circles and updating global coordinates.")
