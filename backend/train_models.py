"""
Derin Öğrenme Modellerini Eğitme Betiği

CNN ve LSTM modellerini gerçek eğitim veri seti ile eğitir ve ağırlıkları kaydeder.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from backend.algorithms.deep_learning import ForgeryCNN, ForgeryLSTM


class ForgerydataSet(Dataset):
    """Sahtecilik tespiti için veri seti."""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: fake/ ve real/ alt klasörleri içeren klasör
            transform: Görüntüye uygulanacak dönüşümler
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif"]
        
        # Sahte görüntüler (label=1)
        fake_dir = self.data_dir / "fake"
        if fake_dir.exists():
            for ext in image_extensions:
                for img_path in fake_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(1)
        
        # Gerçek görüntüler (label=0)
        real_dir = self.data_dir / "real"
        if real_dir.exists():
            for ext in image_extensions:
                for img_path in real_dir.glob(ext):
                    self.images.append(str(img_path))
                    self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Görüntüyü yükle
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Görüntü yüklenemedi: {img_path}")
        
        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Dönüşümler uygula
        if self.transform:
            image = self.transform(image)
        else:
            # Varsayılan: normalize et
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image, torch.tensor(label, dtype=torch.long)


def train_cnn(train_loader, val_loader, epochs=10, lr=0.001, patience=10):
    """CNN modelini eğit."""
    print("\n=== CNN Eğitimi Başladı ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    model = ForgeryCNN(input_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    no_improve = 0
    
    for epoch in range(epochs):
        # Eğitim
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{train_loss / (train_total // len(images)):.4f}"})
        
        train_acc = train_correct / train_total
        print(f"Epoch {epoch+1} - Train Doğruluk: {train_acc:.4f}")
        
        # Doğrulama
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1} - Val Doğruluk: {val_acc:.4f}")
            
            # En iyi modeli kaydet
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                model_path = Path("backend/models/cnn_weights.pth")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"✓ Model kaydedildi: {model_path}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏱ Early stopping: {patience} epoch boyunca iyileşme olmadı.")
                    break
    
    # Eğitim sonunda, eğer doğrulama yoksa ya da en iyi model bulunamadıysa, son modeli kaydet
    model_path = Path("backend/models/cnn_weights.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not val_loader or best_val_acc == 0.0:
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model kaydedildi: {model_path}")
    else:
        print(f"✓ En iyi doğrulama modelinin kaydedilmiş olması beklenir: {model_path}")
    
    return model


def train_lstm(train_loader, val_loader, epochs=10, lr=0.001, patience=10):
    """LSTM modelini eğit."""
    print("\n=== LSTM Eğitimi Başladı ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")
    
    model = ForgeryLSTM(feature_dim=128, hidden_dim=64, sequence_length=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    no_improve = 0
    
    for epoch in range(epochs):
        # Eğitim
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{train_loss / (train_total // len(images)):.4f}"})
        
        train_acc = train_correct / train_total
        print(f"Epoch {epoch+1} - Train Doğruluk: {train_acc:.4f}")
        
        # Doğrulama
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            print(f"Epoch {epoch+1} - Val Doğruluk: {val_acc:.4f}")
            
            # En iyi modeli kaydet
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                model_path = Path("backend/models/lstm_weights.pth")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"✓ Model kaydedildi: {model_path}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"⏱ Early stopping: {patience} epoch boyunca iyileşme olmadı.")
                    break
    
    # Eğitim sonunda, eğer doğrulama yoksa ya da en iyi model bulunamadıysa, son modeli kaydet
    model_path = Path("backend/models/lstm_weights.pth")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not val_loader or best_val_acc == 0.0:
        torch.save(model.state_dict(), model_path)
        print(f"✓ Model kaydedildi: {model_path}")
    else:
        print(f"✓ En iyi doğrulama modelinin kaydedilmiş olması beklenir: {model_path}")
    
    return model


def create_data_loaders(dataset, batch_size=16, val_split=0.2, shuffle=True, num_workers=0):
    """Veri yükleyicilerini oluştur."""
    if val_split is None or val_split <= 0.0:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return train_loader, None

    total_size = len(dataset)
    if total_size < 2:
        raise ValueError("Veri seti çok küçük: en az 2 görüntü gerekli.")

    val_size = max(1, int(total_size * val_split))
    train_size = total_size - val_size
    if train_size < 1:
        val_size = total_size - 1
        train_size = 1

    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forgery detection model training script")
    parser.add_argument("--data-dir", type=str, default="data/train", help="Veri seti klasörü")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch boyutu")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch sayısı")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Öğrenme hızı")
    parser.add_argument("--val-split", type=float, default=0.2, help="Doğrulama için ayrılacak veri oranı")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader için işçi sayısı")
    parser.add_argument("--model", type=str, choices=["cnn", "lstm", "both"], default="both", help="Hangi modeli eğiteceğini seç")
    parser.add_argument("--patience", type=int, default=10, help="Doğrulama iyileşmesi olmazsa kaç epoch sonra duracağı")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    VAL_SPLIT = args.val_split
    NUM_WORKERS = args.num_workers

    if not DATA_DIR.exists():
        print(f"❌ Veri seti bulunamadı: {DATA_DIR}")
        print("Lütfen data/train/fake ve data/train/real klasörlerini oluşturun.")
        exit(1)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Veri seti yükleniyor...")
    dataset = ForgerydataSet(DATA_DIR, transform=transform)
    print(f"Toplam görüntü: {len(dataset)}")

    if len(dataset) == 0:
        print("❌ Veri setinde hiç görüntü bulunamadı.")
        exit(1)

    train_loader, val_loader = create_data_loaders(
        dataset,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    print(f"Eğitim görüntü sayısı: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Doğrulama görüntü sayısı: {len(val_loader.dataset)}")

    if args.model in ("cnn", "both"):
        train_cnn(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=args.patience)

    if args.model in ("lstm", "both"):
        train_lstm(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, patience=args.patience)

    print("\n✓ Eğitim tamamlandı!")
    print("Kaydedilen modeller:")
    print("  - backend/models/cnn_weights.pth")
    print("  - backend/models/lstm_weights.pth")
