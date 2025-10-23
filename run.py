import os
import shutil
import time
from analyzer_table.detect_ball.Debugger import Debugger
from analyzer_table.balls_from_image import full_analyzer_pipeline
import cv2


def run_first_input_image():
    """
    🎬 מריץ את full_analyzer_pipeline על התמונה הראשונה בתיקייה 'input'
    מבצע ניקוי תיקיות, שינוי שמות קבצים, מחיקת pockets,
    ולבסוף מוחק את התמונה שהורצה.
    """
    # 📂 נתיבי בסיס
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, "input")
    out_dir = os.path.join(base_dir, "out")
    balls_dir = os.path.join(out_dir, "balls")
    pockets_dir = os.path.join(out_dir, "pockets")

    # ==============================
    # 🧹 שלב 1: ניקוי תיקיות ישנות
    # ==============================
    Debugger.log("🧹 Cleaning previous output folders...")
    for folder in [balls_dir, pockets_dir]:
        if os.path.exists(folder):
            Debugger.log(f"🗑️ Removing old folder: {folder}")
            shutil.rmtree(folder)
    os.makedirs(balls_dir, exist_ok=True)
    os.makedirs(pockets_dir, exist_ok=True)

    # ==============================
    # 📸 שלב 2: מציאת התמונה הראשונה
    # ==============================
    if not os.path.exists(input_dir):
        Debugger.error(f"❌ תיקיית input לא קיימת: {input_dir}")
        return

    images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not images:
        Debugger.error(f"❌ לא נמצאו תמונות בתיקייה {input_dir}")
        return

    first_image = sorted(images)[0]
    image_path = os.path.join(input_dir, first_image)
    Debugger.log(f"📸 Found first image: {first_image}")

    # ==============================
    # 🚀 שלב 3: הרצת הפייפליין
    # ==============================
    result = full_analyzer_pipeline(image_path)

    # ==============================
    # 🧾 שלב 4: שינוי שמות לקבצי כדורים
    # ==============================
    Debugger.log("📝 Renaming ball images with timestamp...")
    if os.path.exists(balls_dir):
        for filename in os.listdir(balls_dir):
            old_path = os.path.join(balls_dir, filename)
            if not os.path.isfile(old_path):
                continue

            # צור timestamp ייחודי
            timestamp = int(time.time() * 1000)
            base_name = f"ball_{timestamp}"
            new_name = f"{base_name}.png"
            new_path = os.path.join(balls_dir, new_name)

            # אם כבר קיים – הוסף סיומת _1, _2 וכו'
            counter = 0
            while os.path.exists(new_path):
                new_name = f"{base_name}_{counter}.png"
                new_path = os.path.join(balls_dir, new_name)
                counter += 1

            os.rename(old_path, new_path)
            Debugger.log(f"🔄 Renamed {filename} → {new_name}")
            time.sleep(0.001)  # עיכוב קטן למניעת אותו timestamp

    # ==============================
    # 🧹 שלב 5: מחיקת תיקיית pockets
    # ==============================
    if os.path.exists(pockets_dir):
        Debugger.log(f"🗑️ Removing pockets folder: {pockets_dir}")
        shutil.rmtree(pockets_dir)

    # ==============================
    # 🗑️ שלב 6: מחיקת התמונה מה-input
    # ==============================
    try:
        os.remove(image_path)
        Debugger.log(f"🗑️ Deleted input image after processing: {first_image}")
    except Exception as e:
        Debugger.error(f"⚠️ Failed to delete input image: {e}")

    # ==============================
    # ✅ שלב 7: סיכום
    # ==============================
    Debugger.log(f"✅ Completed analysis for {first_image}")
    Debugger.log(f"⚪ White: {result.white.center if result.white else 'N/A'}")
    Debugger.log(f"⚫ Black: {result.black.center if result.black else 'N/A'}")
    Debugger.log(f"🎱 Total balls detected: {len(result.balls)}")
    Debugger.log(f"🕳️ Pockets detected: {len(result.Pockets.pocket_list)}")

    return result


# ✅ הרצה ישירה
if __name__ == "__main__":
    run_first_input_image()
