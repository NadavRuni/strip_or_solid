import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict
import tempfile, shutil

# ===========================
# ⚙️ CONFIG
# ===========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "out")
VAL_DIR_MELEIEM = os.path.join(BASE_DIR, "out", "val_Meleiem")
VAL_DIR_PASSIM = os.path.join(BASE_DIR, "out", "val_Passim")

BATCH_SIZE = 32
EPOCHS = 25
NUM_FOLDS = 5
LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
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
# 📦 DATASET LOADING (FILTERED)
# ===========================
valid_classes = {"Meleiem", "Passim"}
available = set(os.listdir(DATA_DIR))
missing = valid_classes - available
if missing:
    raise FileNotFoundError(f"Missing required class folders: {missing}")

# צור ספרייה זמנית המכילה רק את התיקיות הרלוונטיות
tmp_root = tempfile.mkdtemp(prefix="filtered_")
for cls in valid_classes:
    src = os.path.join(DATA_DIR, cls)
    dst = os.path.join(tmp_root, cls)
    shutil.copytree(src, dst)

dataset = datasets.ImageFolder(tmp_root, transform=train_transform)
print(f"✅ Loaded filtered dataset with {len(dataset)} images.")
for cls, idx in dataset.class_to_idx.items():
    count = sum(1 for _, label in dataset.samples if label == idx)
    print(f"   - {cls}: {count} images")

# ===========================
# 🧩 TRAINING & CROSS-VALIDATION
# ===========================
def train_one_fold(model, train_loader, val_loader, optimizer, criterion, lr, fold, model_name):
    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.unsqueeze(1).float().to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.unsqueeze(1).float().to(DEVICE)
                outputs = model(images)
                preds = torch.sigmoid(outputs) > 0.5
                val_correct += (preds == labels.bool()).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"[{model_name}] LR={lr:.0e} Fold {fold} | Epoch {epoch+1}/{EPOCHS} | Train={train_acc:.3f} | Val={val_acc:.3f}")
    return val_acc


def cross_validate(model_name):
    print(f"\n=== 🔁 Cross-validating {model_name} ===")
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    results = []
    for lr in LEARNING_RATES:
        fold_accs = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset))), start=1):
            model = create_model(model_name)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

            val_acc = train_one_fold(model, train_loader, val_loader, optimizer, criterion, lr, fold, model_name)
            fold_accs.append(val_acc)

        mean_acc = np.mean(fold_accs)
        results.append((model_name, lr, mean_acc))
        print(f"✅ {model_name} | LR={lr:.0e} | Mean Val Acc={mean_acc:.3f}")
    return sorted(results, key=lambda x: x[2], reverse=True)[:3]  # top 3

# ===========================
# 🧾 FINAL VALIDATION
# ===========================
def evaluate_on_external(model, val_dir_meleiem, val_dir_passim):
    val_images = []
    val_labels = []

    for cls_dir, label in [(val_dir_meleiem, 0), (val_dir_passim, 1)]:
        if not os.path.exists(cls_dir):
            print(f"⚠️ Warning: validation folder missing: {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                val_images.append(os.path.join(cls_dir, fname))
                val_labels.append(label)

    correct = 0
    with torch.no_grad():
        for path, label in zip(val_images, val_labels):
            img = datasets.folder.default_loader(path)
            img_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
            output = model(img_tensor)
            prob = torch.sigmoid(output)[0].item()
            pred = 1 if prob > 0.5 else 0
            correct += (pred == label)

    acc = correct / len(val_images) if val_images else 0
    return acc

# ===========================
# 🏁 MAIN
# ===========================
def main():
    all_results = defaultdict(list)
    model_list = ["mobilenet", "efficientnet", "vit"]

    for model_name in model_list:
        top3 = cross_validate(model_name)
        all_results[model_name] = top3

        for _, lr, acc in top3:
            print(f"\n🔥 Retraining {model_name} with LR={lr:.0e} on full dataset...")
            model = create_model(model_name)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            for epoch in range(EPOCHS):
                model.train()
                for images, labels in loader:
                    images, labels = images.to(DEVICE), labels.unsqueeze(1).float().to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()

            # Final external validation
            val_acc = evaluate_on_external(model, VAL_DIR_MELEIEM, VAL_DIR_PASSIM)
            print(f"✅ {model_name} (LR={lr:.0e}) | External validation acc={val_acc:.3f}")

    print("\n🏆 Final summary:")
    for model_name, results in all_results.items():
        print(f"\n=== {model_name.upper()} ===")
        for idx, (name, lr, acc) in enumerate(results, 1):
            print(f"Top {idx}: LR={lr:.0e} | Mean CV Acc={acc:.3f}")


if __name__ == "__main__":
    main()
