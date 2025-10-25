import os
from predict_all_models import predict_ensemble
from analyzer_table.launcher_helper.json_models import Ball_Color

# ===========================
# ⚙️ CONFIG
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VAL_DIR_MELEIEM = os.path.join(BASE_DIR, "out", "val_Meleiem")
VAL_DIR_PASSIM = os.path.join(BASE_DIR, "out", "val_Passim")

# ===========================
# 🧩 EVALUATION FUNCTION
# ===========================
def evaluate_predictions():
    total = 0
    correct = 0
    results = {"solid": {"correct": 0, "total": 0}, "striped": {"correct": 0, "total": 0}}

    print("\n==========================")
    print("🔍 STARTING VALIDATION")
    print("==========================")

    # --- Meleiem (SOLID) ---
    if os.path.exists(VAL_DIR_MELEIEM):
        for fname in sorted(os.listdir(VAL_DIR_MELEIEM)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(VAL_DIR_MELEIEM, fname)
            pred = predict_ensemble(img_path)
            predicted_label = pred["majority_vote_label"]

            expected_label = Ball_Color.SOLID
            is_correct = predicted_label == expected_label

            results["solid"]["total"] += 1
            total += 1
            if is_correct:
                results["solid"]["correct"] += 1
                correct += 1

            status = "✅" if is_correct else "❌"
            print(f"📸 {fname:<30} | expected={expected_label:<8} | predicted={predicted_label:<8} {status}")

    # --- Passim (STRIPED) ---
    if os.path.exists(VAL_DIR_PASSIM):
        for fname in sorted(os.listdir(VAL_DIR_PASSIM)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(VAL_DIR_PASSIM, fname)
            pred = predict_ensemble(img_path)
            predicted_label = pred["majority_vote_label"]

            expected_label = Ball_Color.STRIPED
            is_correct = predicted_label == expected_label

            results["striped"]["total"] += 1
            total += 1
            if is_correct:
                results["striped"]["correct"] += 1
                correct += 1

            status = "✅" if is_correct else "❌"
            print(f"📸 {fname:<30} | expected={expected_label:<8} | predicted={predicted_label:<8} {status}")

    # ===========================
    # 📊 SUMMARY
    # ===========================
    print("\n==========================")
    print("🎯 EVALUATION RESULTS")
    print("==========================")

    solid_acc = (results["solid"]["correct"] / results["solid"]["total"]) if results["solid"]["total"] > 0 else 0
    striped_acc = (results["striped"]["correct"] / results["striped"]["total"]) if results["striped"]["total"] > 0 else 0
    total_acc = (correct / total) if total > 0 else 0

    print(f"🟤 SOLID (Meleiem):   {results['solid']['correct']} / {results['solid']['total']}  → {solid_acc:.2%}")
    print(f"⚪ STRIPED (Passim): {results['striped']['correct']} / {results['striped']['total']}  → {striped_acc:.2%}")
    print("-------------------------------------")
    print(f"🏆 OVERALL ACCURACY: {correct} / {total}  → {total_acc:.2%}")

    return {
        "solid_acc": solid_acc,
        "striped_acc": striped_acc,
        "total_acc": total_acc,
        "total_images": total,
    }

# ===========================
# 🏁 MAIN
# ===========================
if __name__ == "__main__":
    results = evaluate_predictions()
