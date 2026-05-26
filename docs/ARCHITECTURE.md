# Proje Yapı Mimarisi

## Dizin Yapısı

```
YazılımProjeYönetimi/
├── backend/                          # Python FastAPI uygulaması
│   ├── __init__.py                  # Backend modülü
│   ├── main.py                      # FastAPI uygulama ve endpointleri
│   ├── requirements.txt             # Python bağımlılıkları
│   └── algorithms/                  # Algoritma modülleri
│       ├── __init__.py              # Algoritma paketi
│       ├── classical.py             # Klasik CV algoritmaları (SIFT/SURF/AKAZE/ORB)
│       └── deep_learning.py         # Derin Öğrenme modelleri (CNN/LSTM)
│
├── frontend/                         # Web arayüzü
│   ├── index.html                  # Ana HTML dosyası
│   ├── css/
│   │   └── style.css               # CSS stilleri (Glassmorphism)
│   └── js/
│       └── app.js                  # Frontend JavaScript kodu
│
├── docs/                            # Dokümantasyon
│   ├── Doxyfile                    # Doxygen yapılandırması
│   ├── fsm_estimation.md           # FSM efor hesabı raporu
│   ├── user_manual.md              # Kullanıcı kılavuzu
│   ├── API_DOCUMENTATION.md        # API belgelendirmesi
│   ├── ARCHITECTURE.md             # Bu dosya
│   └── doxygen_output/             # Doxygen çıkışı (otomatik oluşur)
│
├── task_board.md                   # Scrum görev tahtası
├── README.md                       # Proje ana sayfası
└── .git/                           # Git deposu
```

## Modül Mimarisi

### Backend Katmanları

```
┌─────────────────────────────────────┐
│     Frontend (HTML/CSS/JS)          │
│   (Glassmorphism UI, Responsive)    │
└──────────────┬──────────────────────┘
               │ HTTP REST API
┌──────────────▼──────────────────────┐
│      FastAPI Application            │
│  (backend/main.py)                  │
│  - Endpoint: POST /upload/          │
│  - Endpoint: GET /health            │
│  - Endpoint: GET /info              │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼─────┐  ┌──────▼──────┐
│  Klasik CV  │  │ Deep Learning │
│ Algorithms  │  │   Models     │
│ (classical) │  │ (deep_learning)│
└─────────────┘  └──────────────┘
       │               │
   ┌───┴────┐      ┌───┴───┐
   │         │      │       │
SIFT  SURF AKAZE ORB  CNN  LSTM
```

## Veri Akışı

```
1. Kullanıcı Arayüzü
   └─> Dosya yükleme (index.html)
       └─> AJAX isteği (app.js)

2. Backend API
   └─> Dosya doğrulama (main.py)
       ├─> Klasik Algoritma Analizi
       │   └─> detect_copy_move() (classical.py)
       │       ├─> SIFT analizi
       │       ├─> SURF analizi
       │       ├─> AKAZE analizi
       │       └─> ORB analizi
       │
       └─> Derin Öğrenme Analizi
           └─> predict_deepfake() (deep_learning.py)
               ├─> CNN tahmini
               ├─> LSTM tahmini
               ├─> ELA analizi
               └─> FFT analizi

3. Sonuç
   └─> JSON formatında dönüş
       └─> Frontend görselleştirmesi
```

## Kod Kalitesi Standartları

### Python Stil Kılavuzu (PEP 8)

- **İndent**: 4 boşluk
- **Maksimum Satır Uzunluğu**: 100 karaktere kadar
- **Docstring Format**: Google Style / Doxygen Uyumlu

### Belgelendirme

Tüm fonksiyonlar ve sınıflar aşağıdaki bilgileri içermelidir:

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Bir satırlık kısa açıklama.
    
    Daha detaylı açıklama paragrafları.
    
    Args:
        param1: Parametrenin açıklaması
        param2: Parametrenin açıklaması
    
    Returns:
        Dönüş değerinin açıklaması
    
    Raises:
        ValueError: Ne zaman hatası atılacağı
    
    Example:
        Kullanım örneği::
        
            result = function_name(value1, value2)
    """
    pass
```

### Hata İşleme

- Tüm hatalı durumlar için HTTPException kullanılmalı
- Anlamlı hata mesajları sağlanmalı
- Loglar yapılmalı (gelecek versiyon)

### Birim Testler

Tüm kritik fonksiyonlar için test yazılmalı:

```python
# tests/test_algorithms.py
def test_detect_copy_move_with_valid_image():
    pass

def test_detect_copy_move_with_invalid_image():
    pass

def test_predict_deepfake_threshold():
    pass
```

## Bağımlılıklar

### Temel Kütüphaneler

| Kütüphane | Sürüm | Amaç |
|-----------|-------|------|
| FastAPI | >=0.100.0 | Web framework |
| OpenCV | >=4.7.0 | Bilgisayar görüşü |
| PyTorch | >=2.0.0 | Derin öğrenme |
| NumPy | >=1.24.0 | Sayısal hesaplamalar |
| Pillow | >=9.5.0 | Görüntü işleme |

### İsteğe Bağlı

| Kütüphane | Amaç |
|-----------|------|
| CUDA | GPU hızlandırması |
| Doxygen | Belgelendirme |
| SonarQube | Kod analizi |

## Güvenlik Notları

1. **Dosya Yüklemesi**
   - Sadece belirtilen formatlar (.jpg, .png, .gif) kabul edilir
   - Dosya boyutu sınırlaması yapılmalı (gelecek)
   - Virüs tarama entegrasyonu (gelecek)

2. **API Güvenliği**
   - Rate limiting eklenmelidir (gelecek)
   - API key doğrulaması (gelecek)
   - HTTPS kullanılmalıdır (üretim ortamında)

3. **Veri Gizliliği**
   - Yüklenen dosyalar sunucuda saklanmaz
   - Analiz sonuçları üretildikten sonra silinir

## Performans Optimizasyonu

### Şu Anki Durum
- Ortalama analiz süresi: 2-10 saniye
- CPU kullanımı: Orta (~50%)
- Bellek kullanımı: ~500 MB

### Gelecek Iyileştirmeler
- GPU desteği (CUDA)
- Model quantization
- Batch işleme
- Caching mekanizması

## Test Stratejisi

### Birim Testler
- Algoritmaların bireysel doğruluğu
- Parametrelerin geçerlilik kontrolü

### Entegrasyon Testleri
- Endpoint'lerin işlevselliği
- Backend-Frontend veri akışı

### Sistem Testleri
- Çeşitli görüntü formatlarında
- Farklı boyutlarda
- Kötü niyetli girdilere karşı

## Dağıtım Senaryoları

### Lokal Geliştirme
```bash
uvicorn backend.main:app --reload
```

### Docker
```bash
docker build -t forgery-detection .
docker run -p 8000:8000 forgery-detection
```

### Üretim (Gelecek)
- Gunicorn/Uvicorn workers
- Nginx reverse proxy
- PostgreSQL database
- Redis caching

## Versiyon Kontrolü

- **Aktif Branch**: main
- **Geliştirme Branch**: develop
- **Semantik Versiyon**: MAJOR.MINOR.PATCH (1.0.0)

## İletişim

- **Yönetici**: Ar-Ge Ekibi
- **Proje Başlangıcı**: 2026
- **Lisans**: MIT

---

**Son Güncelleme**: 2026  
**Durum**: Kararlı (Stable v1.0.0)
