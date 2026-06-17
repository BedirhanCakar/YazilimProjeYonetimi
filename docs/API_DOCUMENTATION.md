# API Belgelendirmesi - Görüntü Sahteciliği Tespiti

## Giriş

Bu belge, Görüntü Sahteciliği Tespiti sistemi için API endpoinleri, parametreleri ve dönüş değerlerini açıklamaktadır.

## API Endpointleri

### 1. Ana Sayfa

**Endpoint:** `GET /`

Kullanıcı arayüzünü HTML olarak döndürür.

**Cevap:**
- `200`: HTML arayüzü
- `404`: Arayüz dosyası bulunamadı

---

### 2. Sağlık Kontrolü

**Endpoint:** `GET /health`

API'nin çalışıp çalışmadığını kontrol eder.

**Cevap:**

```json
{
    "status": "ok",
    "message": "API çalışıyor",
    "service": "Görüntü Sahteciliği Tespiti"
}
```

---

### 3. Bilgi Endpointi

**Endpoint:** `GET /info`

Uygulama hakkında detaylı bilgi sağlar.

**Cevap:**

```json
{
    "name": "Görüntü Sahteciliği Tespiti",
    "version": "1.0.0",
    "description": "Klasik CV ve Derin Öğrenme tabanlı görüntü sahtecilik tespiti",
    "algorithms": {
        "classical": ["SIFT", "SURF", "AKAZE", "ORB"],
        "deep_learning": ["CNN", "LSTM"],
        "analysis": ["ELA", "FFT"]
    }
}
```

---

### 4. Görüntü Yükleme ve Analiz

**Endpoint:** `POST /upload/`

Görüntü dosyasını yükler ve sahtecilik analizi yapar.

**İstek Parametreleri:**

| Parameter | Tür | Gerekli | Açıklama |
|-----------|-----|---------|----------|
| `file` | File | Evet | Analiz edilecek görüntü dosyası (JPG, PNG, GIF) |
| `threshold` | float | Hayır | AI tespit eşiği (0.0-1.0, varsayılan: 0.5) |

**cURL Örneği:**

```bash
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@test.jpg" \
  -F "threshold=0.5"
```

**Başarılı Cevap (200):**

```json
{
  "filename": "test.jpg",
  "image_shape": {
    "height": 1080,
    "width": 1920,
    "channels": 3
  },
  "classical_algorithms": {
    "sift": {
      "available": true,
      "keypoints_count": 245,
      "matches_count": 18,
      "region_detection": {
        "detected": false,
        "confidence": 0.12,
        "region_count": 18,
        "average_distance": 45.3,
        "consistency_score": 0.65
      }
    },
    "surf": { ... },
    "akaze": { ... },
    "orb": { ... },
    "summary": {
      "available_methods": ["sift", "surf", "akaze", "orb"],
      "overall_confidence": 0.15,
      "method_count": 4,
      "consensus_detected": false
    }
  },
  "ai_algorithms": {
    "cnn": {
      "probability": 0.23,
      "is_suspicious": false,
      "confidence": 0.87
    },
    "lstm": {
      "probability": 0.19,
      "is_suspicious": false,
      "confidence": 0.91
    },
    "ela_analysis": {
      "detected": false,
      "score": 0.08
    },
    "frequency_analysis": {
      "detected": false,
      "score": 0.12
    },
    "threshold": 0.5,
    "overall_suspicion": {
      "average_score": 0.15,
      "max_score": 0.23,
      "consensus": false
    }
  },
  "timestamp": null,
  "message": "Görüntü analizi tamamlandı"
}
```

**Hata Cevapları:**

```json
{
  "detail": "Sadece JPG, JPEG, PNG ve GIF formatları desteklenir."
}
```

HTTP Status: `400`

---

## Algoritmaların Açıklaması

### Klasik Algoritmalar

#### SIFT (Scale-Invariant Feature Transform)
- Ölçek ve rotasyona karşı değişmez
- Copy-move tespitinde etkili
- Yüksek hesaplama maliyeti

#### SURF (Speeded-Up Robust Features)
- SIFT'in hızlanmış versiyonu
- Daha düşük hesaplama maliyeti
- Benzer doğruluk oranı

#### AKAZE (Accelerated-KAZE)
- Açık kaynak ve hızlı
- Mobil cihazlara uygun
- İyi özellik çıkarma

#### ORB (Oriented FAST and Rotated BRIEF)
- Serbest ve açık kaynak
- En düşük hesaplama maliyeti
- Gerçek zamanlı uygulamalar için ideal

### Derin Öğrenme Modelleri

#### CNN (Convolutional Neural Networks)
- Piksel düzeyinde analiz
- Yerel özellik çıkarma
- Görüntü sınıflandırması

#### LSTM (Long Short-Term Memory)
- Zamansal bağımlılıkları modelleme
- Sekansyel veriler için ideal
- Video manipülasyonlarında etkili

### Gelişmiş Analiz

#### ELA (Error Level Analysis)
- JPEG sıkıştırma hataları analizi
- Manipülasyonu ayırt etmede faydalı
- Yüksek değerler şüphe gösterir

#### Frekans Domain Analizi (FFT)
- Fourier Transform ile analiz
- Doğal vs yapay ayırımı
- Gürültü karakteristiği

---

## Hata Kodları

| Kod | Açıklama | Nedeni |
|-----|----------|--------|
| 200 | Başarı | İstek başarıyla işlendi |
| 400 | Kötü İstek | Desteklenmeyen dosya formatı veya geçersiz parametre |
| 404 | Bulunamadı | Endpoint bulunamadı |
| 500 | Sunucu Hatası | Analiz sırasında dahili hata |

---

## Örnek İstekler

### Python Örneği

```python
import requests

url = "http://localhost:8000/upload/"
files = {"file": open("test.jpg", "rb")}
params = {"threshold": 0.5}

response = requests.post(url, files=files, params=params)
data = response.json()

print(f"CNN Olasılığı: {data['ai_algorithms']['cnn']['probability']}")
print(f"Klasik Algoritma Güveni: {data['classical_algorithms']['summary']['overall_confidence']}")
```

### JavaScript Örneği

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('threshold', 0.5);

const response = await fetch('/upload/', {
    method: 'POST',
    body: formData
});

const data = await response.json();
console.log(data);
```

---

## Performans Notları

- **Ortalama Analiz Süresi**: 2-10 saniye (görüntü boyutuna göre)
- **Önerilen Görüntü Boyutu**: 512x512 - 4096x4096 piksel
- **Maksimum Dosya Boyutu**: 50 MB
- **CPU Kullanımı**: Orta
- **GPU Desteği**: PyTorch GPU hızlandırması (CUDA kurulu ise)

---

## Sürüm Tarihi

| Sürüm | Tarih | Değişiklikler |
|-------|-------|---------------|
| 1.0.0 | 2026 | İlk sürüm, tüm ana özellikler |

---

**Son Güncelleme**: 2026  
**Durum**: Kararlı (Stable)
