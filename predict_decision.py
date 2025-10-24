import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ===========================
# ⚙️ CONFIG
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
DECIDE_DIR = os.path.join(BASE_DIR, "decide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = (224, 224)

print(f"📁 Decision images directory: {DECIDE_DIR}")

# ===========================
# 🧠 MODEL LOADING
# ===========================
print("🔄 Loading model...")
model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully")

# ===========================
# 🖼️ IMAGE LOADING
# ===========================
images = [f for f in os.listdir(DECIDE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if not images:
    print("❌ no pictures available in 'decide' folder. Please add an image and try again.")
    exit()

image_path = os.path.join(DECIDE_DIR, images[0])
print(f"📸 Analyzing {image_path}...")

# ===========================
# 🔄 PREPROCESSING
# ===========================
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open(image_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

# ===========================
# 🔮 PREDICTION
# ===========================
with torch.no_grad():
    output = model(img_tensor)
    prob = torch.sigmoid(output)[0].item()

# ===========================
# 🏁 RESULT
# ===========================
if prob < 0.5:
    result = "🟤 (Solid) -> MELEIEM"
else:
    result = "⚪ (Striped) -> PASSIM"

print(f"✅ Prediction result: {result} (confidence={prob:.3f})")
