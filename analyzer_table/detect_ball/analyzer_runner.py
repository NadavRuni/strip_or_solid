import os
from PIL import Image
from .Debugger import Debugger
from .ball_ditect import detect_balls_opencv
from analyzer_table.launcher_helper.json_models import PhotoData  # ✅ נוספה לשימוש בסוג ההחזרה

def run_full_analysis(path):
    Debugger.log("Starting main function")

    if not path:
        Debugger.error("Path does not exist")
        return None, None  # ✅ החזרה ריקה במקרה שאין קובץ
    Debugger.log(f"Found image path: {path}")
    return analyzer(path)  # ✅ הפונקציה מחזירה את התוצאות

def analyzer(path):
    base_dir = os.path.dirname(__file__)

    output_dir = os.path.join(base_dir, "analyzer")
    detect_dir_opencv = os.path.join(base_dir, "detect_analyzer_opencv")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(detect_dir_opencv, exist_ok=True)
    Debugger.log("Created output directories")

    img = Image.open(path)
    w, h = img.size

    # ✅ שמירה של התמונה הראשית כחלק נוסף בתהליך
    main_cut_path = os.path.join(output_dir, "cut_main.png")
    img.save(main_cut_path)
    Debugger.log("Saved main image as cut_main.png for detection")

    # חיתוך ל-16 חלקים
    parts_info = split_image(img, overlap_ratio=0.25)
    Debugger.log(f"Image split into {len(parts_info)} parts")

    # שמירה של כל החתיכות
    for i, part_info in enumerate(parts_info, start=1):
        filename = os.path.join(output_dir, f"cut_{i}.png")
        part_info["image"].save(filename)
        part_info["file_name"] = f"cut_{i}.png"
        part_info.pop("image")

    Debugger.log("All parts saved successfully (including metadata)")

    main_info = {
        "file_name": "cut_main.png",
        "origin_x": 0,
        "origin_y": 0,
        "width": w,
        "height": h
    }
    parts_info.append(main_info)
    Debugger.log("Added main image to detection list")

    # ✅ קבלת כל אובייקטי ה-PhotoData מהזיהוי
    photo_data_list = detect_balls_opencv(output_dir, detect_dir_opencv, parts_info, w, h)

    Debugger.warn("✅ Finished analyzing main image and all parts with OpenCV")

    # ✅ חילוץ נפרד של התמונה הראשית מהרשימה
    main_photo = next((p for p in photo_data_list if p.cut_name == "cut_main.png"), None)
    sub_photos = [p for p in photo_data_list if p.cut_name != "cut_main.png"]
    import shutil

    # ✅ בסוף הפונקציה detect_balls_opencv
    shutil.rmtree(output_dir, ignore_errors=True)
    shutil.rmtree(detect_dir_opencv, ignore_errors=True)



    

    return sub_photos, main_photo  # ✅ החזרה של שתי התוצאות


def split_image(img, overlap_ratio=0.25):
    Debugger.log(f"Splitting image with overlap ratio {overlap_ratio}")
    w, h = img.size
    rows, cols = 4, 6
    part_w = w // cols
    part_h = h // rows
    overlap_w = int(part_w * overlap_ratio)
    overlap_h = int(part_h * overlap_ratio)

    parts_info = []

    for row in range(rows):
        for col in range(cols):
            left = max(0, col * part_w - overlap_w)
            right = min(w, (col + 1) * part_w + overlap_w)
            bottom = max(0, row * part_h - overlap_h)
            top = min(h, (row + 1) * part_h + overlap_h)


            part = img.crop((left, bottom, right, top))

            part_info = {
                "image": part,
                "origin_x": left,
                "origin_y": bottom,
                "width": right - left,
                "height": top - bottom
            }
            parts_info.append(part_info)

            Debugger.log(f"Created part ({row},{col}) -> origin=({left},{bottom}) size=({right-left},{top-bottom})")

    



    return parts_info
