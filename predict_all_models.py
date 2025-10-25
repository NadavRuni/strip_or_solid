import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image

# ===========================
# ⚙️ CONFIG
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# משקולות לכל מודל (ניתן לשנות בעתיד)
MODEL_WEIGHTS = {
    "mobilenet": 1.0,
    "efficientnet": 1.0,
    "vit": 1.0
}

# ===========================
# 🧩 TRANSFORM
# ===========================
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===========================
# 🧠 MODEL LOADING
# ===========================
def create_model(model_name):
    if model_name == "mobilenet":
        model = models.mobilenet_v3_small(pretrained=False)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif model_name == "vit":
        model = models.vit_b_16(pretrained=False)
        model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(DEVICE)

def load_model(model_name):
    model = create_model(model_name)
    model_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Missing model file: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# ===========================
# 🔮 ENSEMBLE PREDICTION
# ===========================
def predict_ensemble(image_path: str) -> dict:
    """
    מריץ את התמונה על שלושת המודלים ומחזיר את ההכרעה הכוללת.
    0 = Meleiem (מלאים)
    1 = Passim (פסים)
    """
    # טען תמונה
    img = Image.open(image_path).convert("RGB")
    img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)

    # טען את כל המודלים
    model_names = ["mobilenet", "efficientnet", "vit"]
    results = {}
    total_weight = 0.0
    weighted_sum = 0.0

    for name in model_names:
        weight = MODEL_WEIGHTS[name]
        model = load_model(name)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output)[0].item()  # הסתברות בין 0–1
            pred_label = 1 if prob > 0.5 else 0
            results[name] = {
                "prob": prob,
                "pred": pred_label,
                "weight": weight
            }

            weighted_sum += prob * weight
            total_weight += weight

    # חישוב ממוצע משוקלל
    final_score = weighted_sum / total_weight
    final_pred = 1 if final_score > 0.5 else 0
    final_label = "striped" if final_pred == 1 else "solid"

    # ספירת הכרעות (אם 2 מתוך 3 אמרו פסים)
    votes = sum(r["pred"] for r in results.values())
    vote_based = 1 if votes >= 2 else 0
    vote_label = "striped" if vote_based == 1 else "solid"

    return {
        "image": os.path.basename(image_path),
        "models": results,
        "weighted_avg_score": final_score,
        "weighted_label": final_label,
        "majority_vote_label": vote_label
    }

# ===========================
# 🧪 DEMO
# ===========================
if __name__ == "__main__":
    test_image = "input/example.jpg"  # שנה לנתיב אמיתי
    if not os.path.exists(test_image):
        print(f"❌ Image not found: {test_image}")
    else:
        res = predict_ensemble(test_image)
        print("\n🔍 Prediction summary:")
        for name, info in res["models"].items():
            print(f" - {name:12s}: prob={info['prob']:.3f} | pred={info['pred']} | weight={info['weight']}")
        print(f"\n📊 Weighted label: {res['weighted_label']}")
        print(f"🗳️ Majority vote:  {res['majority_vote_label']}")
