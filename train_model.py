# train_model.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
import numpy as np

# ===========================
#  CONFIGURATION
# ===========================
DATA_DIR = "out"
BATCH_SIZE = 32
EPOCHS = 25
NUM_FOLDS = 5
#LEARNING_RATES = [1e-4, 3e-4, 1e-3]
LEARNING_RATES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "best_model.pth"

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
# 🧩 DATASET LOADING (filtered)
# ===========================
import os
from torchvision import datasets, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "out")

# Filter only 'Meleiem' and 'Passim'
valid_classes = {"Meleiem", "Passim"}

# Validate existence
available = set(os.listdir(DATA_DIR))
missing = valid_classes - available
if missing:
    raise FileNotFoundError(f"Missing required class folders: {missing}")

# Build temporary dataset root
import tempfile, shutil
tmp_root = tempfile.mkdtemp(prefix="filtered_")

for cls in valid_classes:
    src = os.path.join(DATA_DIR, cls)
    dst = os.path.join(tmp_root, cls)
    shutil.copytree(src, dst)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(360),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(tmp_root, transform=train_transform)

print(f"✅ Loaded dataset with {len(dataset)} images.")
for cls, idx in dataset.class_to_idx.items():
    count = sum(1 for _, label in dataset.samples if label == idx)
    print(f"   - {cls}: {count} images")

# ===========================
# 🧩 K-FOLD CROSS VALIDATION
# ===========================
def train_one_fold(model, train_loader, val_loader, optimizer, criterion, lr, fold):
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.unsqueeze(1).float().to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
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

        if epoch % 1 == 0 or epoch == EPOCHS - 1:
            print(f"[LR={lr:.0e}] Fold {fold} | Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    return val_acc


def create_model():
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model.to(DEVICE)


def cross_validate(lr):
    print(f"\n=== 🔁 Cross-validating with LR={lr:.0e} ===")
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset))), start=1):
        model = create_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        val_acc = train_one_fold(model, train_loader, val_loader, optimizer, criterion, lr, fold)
        fold_accuracies.append(val_acc)

    mean_acc = np.mean(fold_accuracies)
    print(f"✅ LR={lr:.0e} | Mean Validation Accuracy: {mean_acc:.3f}\n")
    return mean_acc


# ===========================
# 🧩 MAIN PIPELINE
# ===========================
def main():
    print(f"Training on {DEVICE} using {len(LEARNING_RATES)} learning rates and {NUM_FOLDS}-fold cross-validation.\n")

    best_lr = None
    best_acc = 0.0
    best_model_state = None

    for lr in LEARNING_RATES:
        mean_acc = cross_validate(lr)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_lr = lr
            # Train again on full dataset with this LR
            model = create_model()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=best_lr)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            for epoch in range(EPOCHS):
                model.train()
                for images, labels in loader:
                    images, labels = images.to(DEVICE), labels.unsqueeze(1).float().to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            best_model_state = model.state_dict()

    # Save best model
    if best_model_state:
        torch.save(best_model_state, BEST_MODEL_PATH)
        print(f"\n🏁 Best model saved to {BEST_MODEL_PATH} with accuracy={best_acc:.3f} and learning rate={best_lr:.0e}")
    else:
        print("❌ No model was trained successfully.")


if __name__ == "__main__":
    main()
