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
from pathlib import Path


def _load_model_weights(model, model_name):
    """Load pre-trained weights if they exist."""
    weights_path = Path(__file__).parent.parent / "models" / f"{model_name}_weights.pth"
    if weights_path.exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
            print(f"[INFO] Model ağırlıkları yüklendi: {weights_path}")
        except Exception as e:
            print(f"[WARN] Model ağırlıkları yüklenemiyor: {model_name}: {e}")
    return model


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


def _extract_ela_features(image: np.ndarray, quality: int = 95) -> Tuple[float, float, float]:
    """
    Error Level Analysis (ELA) analizini yapar ve istatistiklerini döner.
    
    Args:
        image: BGR formatında giriş görüntüsü
        quality: JPEG sıkıştırma kalitesi
        
    Returns:
        (ortalama_hata, standart_sapma, yapaylik_skoru)
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    
    ela = cv2.absdiff(image, decimg)
    ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    
    mean_val = float(np.mean(ela_gray))
    std_val = float(np.std(ela_gray))
    
    # AI görüntülerinde (tümüyle yapay üretilmişse) gürültü varyansı son derece homojendir (düşük std).
    # Orijinal resimlerde ise dokulu alanlar nedeniyle std_val daha yüksektir.
    # Yapaylık skoru: std_val çok düşük (homojen) veya aşırı dengesiz ise yüksek olur.
    if std_val < 1.2:
        score = float(np.clip((1.2 - std_val) / 1.2, 0.0, 1.0))
    elif std_val > 12.0:
        score = float(np.clip((std_val - 12.0) / 20.0, 0.0, 1.0))
    else:
        score = 0.0
        
    return mean_val, std_val, score


def _extract_frequency_features(image: np.ndarray) -> Tuple[float, float]:
    """
    Frekans domain analizi (FFT) ile periyodik ızgara hatalarını yakalar.
    
    Args:
        image: BGR formatında giriş görüntüsü
        
    Returns:
        (spektral_oran, yapaylik_skoru)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Boyutu standartlaştırarak hesaplama tutarlılığını sağla
    gray_resized = cv2.resize(gray, (256, 256))
    
    f_transform = np.fft.fft2(gray_resized)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Log genlik sıkıştırması
    magnitude_log = np.log(magnitude + 1.0)
    
    # Merkez (alçak frekans) filtreleme
    cy, cx = 128, 128
    r = 20
    mask = np.ones((256, 256), dtype=np.float32)
    cv2.circle(mask, (cx, cy), r, 0, -1)
    
    high_freq = magnitude_log * mask
    non_zero = high_freq[mask > 0]
    
    mean_val = np.mean(non_zero)
    std_val = np.std(non_zero)
    
    # AI resimlerindeki yapay ızgaralardan dolayı spektral dağılım anormaldir.
    spectral_ratio = float(std_val / (mean_val + 1e-6))
    
    # Doğal görüntülerde spectral_ratio genellikle 0.18 - 0.35 arasındadır.
    if spectral_ratio < 0.15:
        score = float(np.clip((0.15 - spectral_ratio) / 0.15, 0.0, 1.0))
    elif spectral_ratio > 0.36:
        score = float(np.clip((spectral_ratio - 0.36) / 0.5, 0.0, 1.0))
    else:
        score = 0.0
        
    return spectral_ratio, score


def _extract_noise_features(image: np.ndarray) -> Tuple[float, float]:
    """
    Görüntünün yerel sensör gürültüsü varyansını hesaplar.
    
    Args:
        image: BGR formatında giriş görüntüsü
        
    Returns:
        (gürültü_varyansı, yapaylik_skoru)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Görüntüyü hafifçe bulanıklaştırıp orijinalden çıkararak yüksek frekanslı gürültüyü izole et
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_map = cv2.absdiff(gray, blurred)
    
    noise_var = float(np.var(noise_map))
    
    # Gerçek fotoğraflarda sensör gürültüsü varyansı genellikle 1.5 - 18.0 arasındadır.
    # AI difüzyon resimleri aşırı pürüzsüz (denoised) olduğundan varyans 0.8'in altındadır.
    if noise_var < 0.8:
        score = float((0.8 - noise_var) / 0.8)
    elif noise_var > 22.0:
        score = float(np.clip((noise_var - 22.0) / 40.0, 0.0, 1.0))
    else:
        score = 0.0
        
    return noise_var, score


def predict_deepfake(image: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    Görüntü sahtecilik ve AI sentez tespiti için CNN/LSTM modelleri ile 
    güçlendirilmiş FFT, ELA ve Gürültü analizlerini çalıştırır.
    
    Args:
        image: Analiz edilecek BGR formatında görüntü
        threshold: Tespit güven eşiği (0-1 arası)
        
    Returns:
        Detaylı analiz sonuçlarını içeren sözlük
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
            "score": 0.0,
            "std_val": 0.0
        },
        "frequency_analysis": {
            "detected": False,
            "score": 0.0,
            "ratio": 0.0
        },
        "noise_analysis": {
            "detected": False,
            "score": 0.0,
            "variance": 0.0
        },
        "synthesis_analysis": {
            "detected": False,
            "score": 0.0
        },
        "threshold": threshold
    }
    
    try:
        # 1. Derin Öğrenme Tensör Hazırlığı ve Tahminler
        tensor = _prepare_image_tensor(image)
        
        # CNN tahmini
        cnn_model = ForgeryCNN(input_channels=3)
        cnn_model = _load_model_weights(cnn_model, "cnn")
        cnn_model.eval()
        with torch.no_grad():
            cnn_logits = cnn_model(tensor)
            cnn_scores = torch.softmax(cnn_logits, dim=1).cpu().numpy()[0]
        cnn_prob = float(cnn_scores[1])
        
        # LSTM tahmini
        lstm_model = ForgeryLSTM(feature_dim=128, hidden_dim=64, sequence_length=4)
        lstm_model = _load_model_weights(lstm_model, "lstm")
        lstm_model.eval()
        with torch.no_grad():
            lstm_logits = lstm_model(tensor)
            lstm_scores = torch.softmax(lstm_logits, dim=1).cpu().numpy()[0]
        lstm_prob = float(lstm_scores[1])
        
        # 2. İstatistiksel Sentez Analizleri
        # ELA Analizi
        ela_mean, ela_std, ela_score = _extract_ela_features(image)
        # Frekans (FFT) Analizi
        fft_ratio, fft_score = _extract_frequency_features(image)
        # Sensör Gürültü Analizi
        noise_var, noise_score = _extract_noise_features(image)
        
        # Model tahmin sınırları
        cnn_flag = cnn_prob >= (threshold * 0.9)
        lstm_flag = lstm_prob >= (threshold * 0.9)
        
        # Analiz bayrakları
        ela_detected = ela_score > 0.40
        fft_detected = fft_score > 0.40
        noise_detected = noise_score > 0.40
        
        results["cnn"] = {
            "probability": cnn_prob,
            "is_suspicious": bool(cnn_prob >= threshold),
            "soft_suspicious": bool(cnn_flag),
            "confidence": float(np.max(cnn_scores))
        }
        
        results["lstm"] = {
            "probability": lstm_prob,
            "is_suspicious": bool(lstm_prob >= threshold),
            "soft_suspicious": bool(lstm_flag),
            "confidence": float(np.max(lstm_scores))
        }
        
        results["ela_analysis"] = {
            "detected": bool(ela_detected),
            "score": ela_score,
            "std_val": ela_std,
            "mean_val": ela_mean
        }
        
        results["frequency_analysis"] = {
            "detected": bool(fft_detected),
            "score": fft_score,
            "ratio": fft_ratio
        }
        
        results["noise_analysis"] = {
            "detected": bool(noise_detected),
            "score": noise_score,
            "variance": noise_var
        }
        
        # 3. AI Üretim (Sentez) Konsensüs Skoru
        # ELA homojenliği, FFT grid bozukluğu ve düşük gürültü varyansı ortak kararda birleştirilir
        synth_score = min(
            ela_score * 0.35 + fft_score * 0.35 + noise_score * 0.30,
            1.0
        )
        # En az 2 analiz yöntemi yapaylığa işaret ediyor ve genel sentez skoru eşiğin üzerindeyse
        synth_detected = synth_score >= 0.45 and (sum([ela_detected, fft_detected, noise_detected]) >= 2)
        
        results["synthesis_analysis"] = {
            "detected": bool(synth_detected),
            "score": synth_score
        }
        
        # 4. Genel Karar Mutabakatı (Consensus)
        all_probs = [
            cnn_prob,
            lstm_prob,
            ela_score,
            fft_score,
            noise_score,
            synth_score
        ]
        
        # AI Sentez veya Derin Öğrenme Modelleri Alarm Veriyorsa
        consensus_count = sum([cnn_flag, lstm_flag, ela_detected, fft_detected, noise_detected, synth_detected])
        
        final_decision = (
            (cnn_flag and lstm_flag) or
            synth_detected or
            (consensus_count >= 3) or
            ((cnn_flag or lstm_flag) and synth_score > 0.40)
        )
        
        results["overall_suspicion"] = {
            "average_score": float(np.mean(all_probs)),
            "max_score": float(np.max(all_probs)),
            "consensus": consensus_count >= 2,
            "final_decision": bool(final_decision),
            "ai_flags": {
                "cnn": bool(cnn_flag),
                "lstm": bool(lstm_flag),
                "ela": bool(ela_detected),
                "frequency": bool(fft_detected),
                "noise": bool(noise_detected),
                "synthesis": bool(synth_detected)
            }
        }
        
    except Exception as e:
        results["error"] = str(e)
        
    return results

