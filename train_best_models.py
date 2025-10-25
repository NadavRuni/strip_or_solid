import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import tempfile, shutil

# ===========================
# ⚙️ CONFIG
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "out")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning rates chosen from top-1 summary
BEST_LRS = {
    "mobilenet": 1e-3,
    "efficientnet": 1e-4,
    "vit": 1e-4
}

# ===========================
# 🧩 TRANSFORMS
# ===========================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ===========================
# 🧠 MODEL FACTORY
# ===========================
def create_model(model_name):
    if model_name == "mobilenet":
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    elif model_name == "efficientnet":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    elif model_name == "vit":
        model = models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, 1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(DEVICE)

# ===========================
# 📦 DATASET LOADING (filtered)
# ===========================
valid_classes = {"Meleiem", "Passim"}
available = set(os.listdir(DATA_DIR))
missing = valid_classes - available
if missing:
    raise FileNotFoundError(f"Missing required class folders: {missing}")

# צור ספרייה זמנית עם רק שתי התיקיות הרלוונטיות
tmp_root = tempfile.mkdtemp(prefix="filtered_")
for cls in valid_classes:
    src = os.path.join(DATA_DIR, cls)
    dst = os.path.join(tmp_root, cls)
    shutil.copytree(src, dst)

dataset = datasets.ImageFolder(tmp_root, transform=train_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"✅ Loaded dataset with {len(dataset)} images across {len(valid_classes)} classes.")

# ===========================
# 🧩 TRAIN FUNCTION
# ===========================
def train_model(model_name, lr):
    print(f"\n🚀 Training {model_name.upper()} | LR={lr:.0e}")
    model = create_model(model_name)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(EPOCHS):
        model.train()
        correct, total, total_loss = 0, 0, 0.0

        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.unsqueeze(1).float().to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        avg_loss = total_loss / total
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss={avg_loss:.4f} | Train Acc={train_acc:.3f}")

    # שמור את המודל המאומן
    save_path = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved {model_name} model to: {save_path}")

# ===========================
# 🏁 MAIN
# ===========================
def main():
    for model_name, lr in BEST_LRS.items():
        train_model(model_name, lr)
    print("\n🏆 All models trained and saved to /models directory.")

if __name__ == "__main__":
    main()
