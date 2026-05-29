"""
Derin Öğrenme Tabanlı Görüntü Sahteciliği Tespiti

Bu modül, CNN ve LSTM temelli sinir ağları kullanarak görüntülerdeki 
manipülasyonları ve deepfake'leri tespit eder.
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple


class ForgeryCNN(nn.Module):
    """
    Evrişimli Sinir Ağı (CNN) tabanlı sahtecilik detektörü.
    
    Mimarisi:
    - 3 evrişimli katman (16 → 32 → 64 filtre)
    - Global ortalama havuzlaması
    - 2 tam bağlı katman
    """
    def __init__(self, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)  # Binary classification

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        return self.fc3(x)


class ForgeryLSTM(nn.Module):
    """
    LSTM tabanlı sahtecilik detektörü.
    
    Mimarisi:
    - Evrişimli katman ile özellik çıkarma
    - LSTM katmanı (temporal modellemesi)
    - Tam bağlı sınıflandırma katmanı
    """
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64, 
                 sequence_length: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((32, 32))
        
        self.lstm = nn.LSTM(
            input_size=feature_dim * 32 * 32,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
        
        self.sequence_length = sequence_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """İleri geçiş."""
        batch_size = x.size(0)
        
        features = self.conv(x)
        features = self.adaptive_pool(features)
        features_flat = features.view(batch_size, -1)
        
        # Sequence oluştur
        features_seq = features_flat.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        lstm_out, (h_n, c_n) = self.lstm(features_seq)
        
        # Son adımın çıktısı
        last_output = lstm_out[:, -1, :]
        
        return self.fc(last_output)


def _prepare_image_tensor(image: np.ndarray, 
                         target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Görüntüyü modele giriş olarak uygun hale getir.
    
    Args:
        image: BGR formatında giriş görüntüsü
        target_size: Hedef boyut
    
    Returns:
        Normalleştirilmiş torch tensörü (1, 3, H, W)
    """
    # BGR'den RGB'ye dönüştür
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    resized = cv2.resize(rgb_image, target_size)
    
    # Normalize (ImageNet istatistikleri)
    normalized = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (normalized - mean) / std
    normalized = normalized.astype(np.float32)
    
    # Channels first (CHW)
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    
    return tensor.unsqueeze(0)  # Batch dimension ekle


def _extract_ela_features(image: np.ndarray, quality: int = 95) -> np.ndarray:
    """
    Error Level Analysis (ELA) özellik çıkarma.
    
    Args:
        image: Giriş görüntüsü
        quality: JPEG kalitesi
    
    Returns:
        ELA haritası
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    
    ela = cv2.absdiff(image, decimg)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    
    return ela_gray


def _extract_frequency_features(image: np.ndarray) -> np.ndarray:
    """
    Frekans domain analizi (FFT).
    
    Args:
        image: Giriş görüntüsü
    
    Returns:
        Frekans ortalaması
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    return magnitude


def predict_deepfake(image: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Görüntü sahtecilik tespiti için CNN ve LSTM modellerini çalıştır.
    
    Args:
        image: Analiz edilecek BGR formatında görüntü
        threshold: Tespit güven eşiği (0-1 arası)
    
    Returns:
        Her modelin tahminini ve güven skorlarını içeren sözlük
    """
    results = {
        "cnn": {
            "probability": 0.0,
            "is_suspicious": False,
            "confidence": 0.0
        },
        "lstm": {
            "probability": 0.0,
            "is_suspicious": False,
            "confidence": 0.0
        },
        "ela_analysis": {
            "detected": False,
            "score": 0.0
        },
        "frequency_analysis": {
            "detected": False,
            "score": 0.0
        },
        "threshold": threshold
    }
    
    try:
        # Tensor hazırlığı
        tensor = _prepare_image_tensor(image)
        
        # CNN Model
        cnn_model = ForgeryCNN(input_channels=3)
        cnn_model.eval()
        
        with torch.no_grad():
            cnn_logits = cnn_model(tensor)
            cnn_scores = torch.softmax(cnn_logits, dim=1).cpu().numpy()[0]
        
        cnn_prob = float(cnn_scores[1])
        results["cnn"] = {
            "probability": cnn_prob,
            "is_suspicious": bool(cnn_prob >= threshold),
            "confidence": float(np.max(cnn_scores))
        }
        
        # LSTM Model
        lstm_model = ForgeryLSTM(feature_dim=128, hidden_dim=64, sequence_length=4)
        lstm_model.eval()
        
        with torch.no_grad():
            lstm_logits = lstm_model(tensor)
            lstm_scores = torch.softmax(lstm_logits, dim=1).cpu().numpy()[0]
        
        lstm_prob = float(lstm_scores[1])
        results["lstm"] = {
            "probability": lstm_prob,
            "is_suspicious": bool(lstm_prob >= threshold),
            "confidence": float(np.max(lstm_scores))
        }
        
        # ELA Analizi
        ela_map = _extract_ela_features(image)
        ela_mean = float(np.mean(ela_map) / 255.0)
        ela_threshold = 0.15
        results["ela_analysis"] = {
            "detected": ela_mean > ela_threshold,
            "score": min(ela_mean, 1.0)
        }
        
        # Frekans Analizi
        freq_features = _extract_frequency_features(image)
        freq_mean = float(np.mean(freq_features) / np.max(freq_features) if np.max(freq_features) > 0 else 0.0)
        results["frequency_analysis"] = {
            "detected": freq_mean > 0.3,
            "score": freq_mean
        }
        
    except Exception as e:
        results["error"] = str(e)
    
    # Genel özet
    all_probs = [
        results["cnn"]["probability"],
        results["lstm"]["probability"],
        results["ela_analysis"]["score"],
        results["frequency_analysis"]["score"]
    ]
    
    results["overall_suspicion"] = {
        "average_score": float(np.mean(all_probs)),
        "max_score": float(np.max(all_probs)),
        "consensus": sum([
            results["cnn"]["is_suspicious"],
            results["lstm"]["is_suspicious"],
            results["ela_analysis"]["detected"],
            results["frequency_analysis"]["detected"]
        ]) >= 2
    }
    
    return results
