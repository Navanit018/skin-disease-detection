import os
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 15

PROJECT_DIR = Path("C:/Users/ACER/OneDrive/Desktop/Skin Disease Detection")

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
CLASS_TO_INDEX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

DISEASE_INFO = {
    'akiec': {'name': 'Actinic Keratoses', 'severity': 'Medium'},
    'bcc': {'name': 'Basal Cell Carcinoma', 'severity': 'Low-Medium'},
    'bkl': {'name': 'Benign Keratosis', 'severity': 'Very Low'},
    'df': {'name': 'Dermatofibroma', 'severity': 'Very Low'},
    'nv': {'name': 'Melanocytic Nevi (Moles)', 'severity': 'Very Low'},
    'vasc': {'name': 'Vascular Lesions', 'severity': 'Very Low'},
    'mel': {'name': 'Melanoma', 'severity': 'HIGH'}
}

MEDICINES = {
    'akiec': ['5-Fluorouracil cream', 'Imiquimod cream', 'Cryotherapy'],
    'bcc': ['Surgery', 'Mohs surgery', 'Radiation therapy'],
    'bkl': ['No treatment needed', 'Cryotherapy if cosmetic'],
    'df': ['No treatment needed', 'Surgical removal if symptomatic'],
    'nv': ['No treatment needed', 'Monitor for changes'],
    'vasc': ['Laser therapy', 'Corticosteroids'],
    'mel': ['Surgery', 'Immunotherapy', 'Chemotherapy']
}

class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        self.labels_list = []
        for _, row in df.iterrows():
            image_id = row['image_id']
            dx = row['dx']
            if dx not in CLASS_NAMES:
                continue
            found = False
            for folder in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
                img_path = Path(img_dir) / folder / f"{image_id}.jpg"
                if img_path.exists():
                    self.samples.append((str(img_path), CLASS_TO_INDEX[dx]))
                    self.labels_list.append(CLASS_TO_INDEX[dx])
                    found = True
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), shear=10, scale=(0.85, 1.15)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

def train():
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Loading metadata...")
    df = pd.read_csv(PROJECT_DIR / "HAM10000_metadata.csv")
    df = df[df['dx'].isin(CLASS_NAMES)]
    print(f"Total: {len(df)}")

    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['dx'])
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_transform, val_transform = get_transforms()

    print("Building datasets...")
    train_dataset = SkinDataset(train_df, PROJECT_DIR, train_transform)
    val_dataset = SkinDataset(val_df, PROJECT_DIR, val_transform)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    labels_tensor = torch.tensor(train_dataset.labels_list)
    class_counts = torch.bincount(labels_tensor)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[labels_tensor]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    print("Building model...")
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, NUM_CLASSES)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    print("\n=== Phase 1: Training classifier head ===")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.classifier.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1)

    best_val_acc = 0.0
    patience = 7
    patience_counter = 0

    for epoch in range(EPOCHS_PHASE1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Phase1 Epoch {epoch+1}/{EPOCHS_PHASE1} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), PROJECT_DIR / 'best_model_pt.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Phase 1 Best Val Acc: {best_val_acc:.4f}")

    print("\n=== Phase 2: Fine-tuning ===")
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2)

    patience_counter = 0
    for epoch in range(EPOCHS_PHASE2):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Phase2 Epoch {epoch+1}/{EPOCHS_PHASE2} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), PROJECT_DIR / 'best_model_pt.pt')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest Val Acc: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    model.load_state_dict(torch.load(PROJECT_DIR / 'best_model_pt.pt', weights_only=True))
    torch.save(model.state_dict(), PROJECT_DIR / 'skin_disease_model_pt.pt')

    artifact = {
        'class_names': CLASS_NAMES,
        'disease_info': DISEASE_INFO,
        'medicines': MEDICINES,
        'img_size': IMG_SIZE,
        'best_val_acc': best_val_acc
    }
    with open(PROJECT_DIR / 'model_artifact_pt.pkl', 'wb') as f:
        pickle.dump(artifact, f)

    print(f"Saved: skin_disease_model_pt.pt")
    print(f"Validation Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    train()
