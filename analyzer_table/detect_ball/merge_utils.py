from analyzer_table.launcher_helper.json_models import PhotoData, Ball, Origin, Rectangle
from analyzer_table.detect_ball.Debugger import Debugger
import math
import cv2
import numpy as np

# קבועים לקביעת גבולות מיזוג
MERGE_MAX_X_DIFF = 15     # מרחק מותר בציר X
MERGE_MAX_Y_DIFF = 15     # מרחק מותר בציר Y
def mergeData(main_photo: PhotoData, sub_photos: list[PhotoData], black_and_white_list: list[Ball],  table_rectangle: Rectangle) -> PhotoData:
    """
    מאחד את כל הכדורים מהתמונה הראשית וכל שאר התמונות.
    כדורים קרובים (מתחת לסף מרחק) נחשבים כאותו כדור.
    כדורים שאינם לפחות 80% בתוך גבולות השולחן — לא ייכללו.
    """
    Debugger.log("🔄 Starting mergeData process with table filtering")

    # === שלב 1: אתחול רשימת הכדורים מהתמונה הראשית ===
    merged_balls = [Ball(center=b.center, radius=b.radius) for b in main_photo.balls]
    Debugger.log(f"Initialized with {len(merged_balls)} balls from main image")


    
    # === שלב 3: איחוד כל הכדורים משאר התמונות ===
    added, skipped, duplicates = 0, 0, 0

    for photo in sub_photos:
        for b in photo.balls:
            if not is_inside_table(b , table_rectangle):
                skipped += 1
                continue

            if _ball_exists(merged_balls, b):
                duplicates += 1
                continue

            merged_balls.append(Ball(center=b.center, radius=b.radius))
            added += 1

    # merge black and white balls        
    for b in black_and_white_list:
        if not is_inside_table(b , table_rectangle):
            skipped += 1
            continue

        if _ball_exists(merged_balls, b):
            duplicates += 1
            continue

        Debugger.log(f"Adding black_white ball at {b.center} with radius {b.radius}")
        merged_balls.append(Ball(center=b.center, radius=b.radius))
        added += 1

    # === שלב 4: יצירת אובייקט מאוחד ===
    finall_balls = [] 
    for ball in merged_balls:
        if not is_inside_table(ball , table_rectangle):
            skipped += 1
            continue
        finall_balls.append(ball)

        

         
    merged_photo = PhotoData(
        cut_name="merged_all.png",
        origin=Origin(0, 0),
        rectangle=main_photo.rectangle,
        balls=finall_balls
    )

    # === שלב 5: סיכום תוצאות יפה ===
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    print(f"\n{BOLD}{CYAN}========== ⚪ MERGE SUMMARY =========={RESET}")
    print(f"Total balls from main image:        {len(main_photo.balls)}")
    print(f"Balls added from sub-images:        {added}")
    print(f"Balls skipped (outside table):      {skipped}")
    print(f"Duplicate balls ignored:            {duplicates}")
    print(f"{GREEN}{BOLD}Final unique balls:                 {len(merged_balls)}{RESET}\n")

    Debugger.log(f"✅ Merge complete — {len(merged_balls)} unique balls retained")

    return merged_photo


def _ball_exists(merged_balls: list[Ball], new_ball: Ball) -> bool:
    """בודק אם כדור כבר קיים (לפי קרבה גאומטרית ברדיוס ובמיקום)."""
     
    for existing in merged_balls:

        dx = abs(existing.center[0] - new_ball.center[0])
        dy = abs(existing.center[1] - new_ball.center[1])



        # אם הם קרובים מאוד — נחשב אותו כדור
        if dx <= MERGE_MAX_X_DIFF and dy <= MERGE_MAX_Y_DIFF :

            return True
    return False
def is_inside_table(ball: Ball, rect: Rectangle) -> bool:
    """
    בודקת אם כל הכדור נמצא בתוך גבולות המלבן.
    מערכת OpenCV: (0,0) = שמאל-למעלה, Y גדל כלפי מטה.
    """
    x, y = ball.center
    r = ball.radius

    # --- גבולות לפי מערכת קואורדינטות של OpenCV ---
    min_x = min(rect.top_left[0], rect.bottom_left[0])     # שמאל
    max_x = max(rect.top_right[0], rect.bottom_right[0])   # ימין
    min_y = min(rect.top_left[1], rect.top_right[1])       # למעלה
    max_y = max(rect.bottom_left[1], rect.bottom_right[1]) # למטה


# --- קבוע בטיחות (למשל 5 פיקסלים) ---
    safety_margin = 10

    # --- בדיקה שהכדור כולו בתוך הגבולות ---
    inside_x = (min_x + r + safety_margin) < x < (max_x - r - safety_margin)
    inside_y = (min_y + r + safety_margin) < y < (max_y - r - safety_margin)
    inside = inside_x and inside_y

    # 🎯 Debugging
    if inside:
            Debugger.log(
                f"🟢 INSIDE ball r={r} → center=({x}, {y}) | bounds X:[{min_x},{max_x}] Y:[{min_y},{max_y}]"
            )
    else:
            Debugger.warn(
                f"🔴 OUTSIDE ball r={r} → center=({x}, {y}) | bounds X:[{min_x},{max_x}] Y:[{min_y},{max_y}]"
            )

    return inside
