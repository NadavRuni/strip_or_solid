import cv2
import numpy as np
from typing import Tuple, Optional, Union
from analyzer_table.launcher_helper.json_models import Rectangle
from analyzer_table.detect_ball.Debugger import Debugger

def find_table_rectangle(image_or_path: Union[str, np.ndarray]) -> Rectangle:
    """
    מזהה את גבולות המלבן המרכזי של השולחן בתמונה או מנתיב קובץ.
    אם לא מזוהה שולחן — מחזיר את גבולות התמונה כמלבן.
    """
    # --- טעינת תמונה ---
    if isinstance(image_or_path, str):
        Debugger.log(f"🖼 Loading image from path: {image_or_path}")
        image = cv2.imread(image_or_path)
        if image is None:
            Debugger.error(f"❌ Failed to load image from {image_or_path}")
            raise ValueError(f"Cannot load image from path: {image_or_path}")
    else:
        image = image_or_path
        Debugger.log("🧠 Received image object directly")

    height, width = image.shape[:2]

    # --- עיבוד תמונה ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 50, 150)

    # --- איתור קווים ---
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=100,
        maxLineGap=30
    )

    if lines is None:
        Debugger.warn("⚠️ No lines detected — using full image bounds as rectangle.")
        return Rectangle(
            top_left=(0, 0),
            top_right=(width - 1, 0),
            bottom_left=(0, height - 1),
            bottom_right=(width - 1, height - 1)
        )

    horizontals = []
    verticals = []

    for (x1, y1, x2, y2) in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:  # קווים אופקיים
            horizontals.append((y1 + y2) / 2)
        elif 80 < abs(angle) < 100:  # קווים אנכיים
            verticals.append((x1 + x2) / 2)

    if len(horizontals) < 2 or len(verticals) < 2:
        Debugger.warn("⚠️ Not enough horizontal/vertical lines — using full image bounds.")
        return Rectangle(
            top_left=(0, 0),
            top_right=(width - 1, 0),
            bottom_left=(0, height - 1),
            bottom_right=(width - 1, height - 1)
        )

    horizontals.sort()
    verticals.sort()

    top_y = int(horizontals[0])
    bottom_y = int(horizontals[-1])
    left_x = int(verticals[0])
    right_x = int(verticals[-1])

    Debugger.log(f"✅ Rectangle edges found (OpenCV): left={left_x}, right={right_x}, top={top_y}, bottom={bottom_y}")

    return Rectangle(
        top_left=(left_x, top_y),
        top_right=(right_x, top_y),
        bottom_left=(left_x, bottom_y),
        bottom_right=(right_x, bottom_y)
    )
