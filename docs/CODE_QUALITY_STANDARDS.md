# Kod Kalitesi Standartları

## 1. Genel Prensipler

- **Okunabilirlik**: Kod, başka bir geliştirici tarafından 5 dakikada anlaşılmalı
- **Yeniden Kullanılabilirlik**: DRY (Don't Repeat Yourself) prensibine uyulmalı
- **Bakımlanabilirlik**: Kod değişikliği zaruri olduğunda kolay değiştirilmelidir
- **Performans**: Kod verimli ve hızlı çalışmalıdır
- **Güvenlik**: Tüm giriş verileri doğrulanmalı, çıkışlar sağlamlaştırılmalıdır

## 2. Python Stil Kılavuzu (PEP 8)

### Genel Kurallar

```python
# ✓ Doğru
variable_name = 10
function_name()
ClassName

# ✗ Yanlış
variableName = 10
function-name()
classname
```

### İndent ve Satır Uzunluğu

- **İndent**: 4 boşluk (Tab değil)
- **Maksimum Satır Uzunluğu**: 100 karaktere kadar
- **Bir dosyadaki maksimum satır sayısı**: 500 satır (aşırı uzun dosyaları ayır)

### İçe Aktarmalar (Imports)

```python
# ✓ Doğru sırası
import os
import sys
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI

from backend.algorithms import detect_copy_move
```

### Docstring Formatı

```python
def analyze_image(image_path: str, threshold: float = 0.5) -> dict:
    """
    Bir satırlık kısa açıklama.
    
    Detaylı açıklama burada yer alır. Fonksiyonun ne yaptığını,
    niçin yaptığını ve özellikleri hakkında bilgi verir.
    
    Args:
        image_path: Analiz edilecek görüntünün yolu
        threshold: Tespit eşiği (0.0-1.0 arası)
    
    Returns:
        Analiz sonuçlarını içeren sözlük
    
    Raises:
        FileNotFoundError: Dosya bulunamadığında
        ValueError: Eşik değeri geçersiz olduğunda
    
    Example:
        Kullanım örneği::
        
            result = analyze_image('test.jpg', threshold=0.5)
            print(result['confidence'])
    """
    pass
```

## 3. Sınıf Tasarımı

### Sınıf Yapısı

```python
class ImageAnalyzer:
    """
    Görüntü analizi yapan sınıf.
    
    Klasik CV ve derin öğrenme algoritmaları kullanarak
    görüntü sahteciliğini tespit eder.
    """
    
    def __init__(self, model_path: str = None):
        """Sınıfı başlatır."""
        self.model_path = model_path
        self._private_var = None
    
    def public_method(self) -> bool:
        """Herkese açık metod."""
        return True
    
    def _private_method(self) -> None:
        """Sadece sınıf içinde kullanılan metod."""
        pass
```

## 4. Fonksiyon İyi Uygulamaları

### Uzunluk ve Karmaşıklık

- **Fonksiyon Uzunluğu**: 50 satırdan az olmalı
- **Parametreler**: 5'ten fazla parametre varsa yeniden tasarla
- **Cyclomatic Complexity**: En fazla 10 olmalı

```python
# ✗ Yanlış: Çok karmaşık
def process_image(img, algo, th, resize, norm, fmt, save_path, log_level):
    if algo == 'sift':
        if resize:
            img = cv2.resize(img, ...)
            if norm:
                img = img / 255.0
        # ... 50 satır daha
    elif algo == 'surf':
        # ... başka 50 satır
    return result

# ✓ Doğru: Basit ve anlaşılır
def process_image(image: np.ndarray, algorithm: str, config: Config) -> dict:
    """Görüntü işleme."""
    processor = AlgorithmFactory.create(algorithm)
    return processor.process(image, config)
```

### Hata İşleme

```python
# ✓ Doğru
def read_image(filepath: str) -> np.ndarray:
    """Görüntü oku."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dosya bulunamadı: {filepath}")
    
    try:
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError("Dosya görüntü olarak okunamadı")
        return image
    except Exception as e:
        logger.error(f"Görüntü okuma hatası: {e}")
        raise

# ✗ Yanlış
def read_image(filepath):
    image = cv2.imread(filepath)
    return image  # Hata yönetimi yok
```

## 5. Değişken ve Sabitler

### Adlandırma Kuralları

```python
# Değişkenler: snake_case
user_count = 0
image_path = "/path/to/image.jpg"

# Sabitler: UPPER_SNAKE_CASE
MAX_IMAGE_SIZE = 4096
DEFAULT_THRESHOLD = 0.5

# Özel değişkenler: _leading_underscore
_internal_cache = {}
_helper_function()
```

### Türü Belirtme (Type Hints)

```python
# ✓ Doğru
def calculate_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """İki görüntü arasındaki benzerliği hesapla."""
    pass

def process_batch(images: List[np.ndarray], 
                 threshold: Optional[float] = None) -> Dict[str, Any]:
    """Görüntü grubunu işle."""
    pass

# ✗ Yanlış
def calculate_similarity(img1, img2):
    pass  # Tür bilgisi yok
```

## 6. API ve REST Endpoint İyi Uygulamaları

### Endpoint Tasarımı

```python
# ✓ Doğru
@app.post("/api/v1/upload/")
async def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    """Görüntü yükle ve analiz et."""
    pass

@app.get("/api/v1/health")
def health_check() -> dict:
    """API sağlık durumunu kontrol et."""
    pass

# ✗ Yanlış
@app.post("/upload_and_analyze_image_with_all_algorithms")
def upload(file):  # Uzun isim, hata işlemesi yok
    pass
```

### Yanıt Formatı

```python
# ✓ Doğru: Tutarlı JSON formatı
{
    "status": "success",
    "data": {
        "filename": "test.jpg",
        "confidence": 0.85
    },
    "timestamp": "2026-05-26T10:30:00Z"
}

# ✓ Hata yanıtı
{
    "status": "error",
    "error": {
        "code": 400,
        "message": "Dosya formatı desteklenmiyor"
    }
}
```

## 7. Test Standartları

### Birim Test Yapısı

```python
# tests/test_algorithms.py
import pytest
from backend.algorithms import detect_copy_move

class TestDetectCopyMove:
    """Copy-move tespiti testleri."""
    
    def test_valid_image(self):
        """Geçerli görüntü ile test."""
        assert True
    
    def test_invalid_image(self):
        """Geçersiz görüntü ile test."""
        with pytest.raises(ValueError):
            pass
    
    def test_threshold_boundary(self):
        """Eşik değer sınırlarını test et."""
        pass
```

### Test Kapsamı Hedefleri

- **Kritik Fonksiyonlar**: %100
- **Diğer Fonksiyonlar**: %80+
- **Genel Proje**: %70+

## 8. Belgelendirme Standardı

### README Gereksinimleri

- Proje açıklaması
- Kurulum talimatları
- Kullanım örnekleri
- API dokümantasyonuna bağlantı

### Kod Yorum İlkeleri

```python
# ✓ Doğru: Açıklayıcı, gerekli yerde
def calculate_distance(p1: Tuple, p2: Tuple) -> float:
    """Öklid uzaklığını hesapla."""
    # İki nokta arasındaki değişimleri hesapla
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # Pisagor teoremini uygula
    return math.sqrt(dx**2 + dy**2)

# ✗ Yanlış: Aşırı veya gereksiz yorum
def add(a, b):  # a ve b'yi topla ve sonucu döndür
    # Toplama işlemi
    result = a + b  # result değişkenini oluştur
    # result'ı döndür
    return result
```

## 9. Performans Standartları

### Hedeflenen Metrikleri

| Metrik | Hedef | Kabul Edilebilir |
|--------|-------|------------------|
| Analiz Süresi | < 5 saniye | < 10 saniye |
| Bellek Kullanımı | < 500 MB | < 1 GB |
| CPU Kullanımı | < 50% | < 80% |
| Model Yükleme Süresi | < 2 saniye | < 5 saniye |

### Optimizasyon Kontrol Listesi

- [ ] Gereksiz döngüler ortadan kaldırıldı
- [ ] Büyük nesneler sadece gerektiğinde oluşturuluyor
- [ ] Cache mekanizması kullanılıyor (varsa)
- [ ] Profil analiz yapıldı ve optimizasyonlar uygulandı

## 10. Güvenlik Standartları

### Giriş Doğrulama

```python
# ✓ Doğru
def upload_image(file: UploadFile = File(...)) -> JSONResponse:
    """Görüntü yükle."""
    # Dosya türünü kontrol et
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Dosya türü geçersiz")
    
    # Dosya boyutunu kontrol et
    content = file.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Dosya çok büyük")
    
    # İşlem yap
    return analyze(content)
```

### Veri Gizliliği

- Hassas veriler şifreli saklanmalı
- Loglardan şifre/token çıkarılmalı
- Geçici dosyalar silinmeli

## 11. SonarQube Kalite Kapısı Kriterler

### Başarısız Olmak İçin

- **Kod Kapsamı**: < 60%
- **Yinelenen Kod**: > 10%
- **Code Smells**: > 10
- **Güvenlik Açıkları**: > 0
- **Kritik Hatalar**: > 0

### Uyarı Seviyesi

- **Kod Kapsamı**: 60-70%
- **Yinelenen Kod**: 5-10%
- **Code Smells**: 5-10
- **Küçük Hatalar**: 10+

## 12. Kontrol Listeleri

### Commit Öncesi

- [ ] Tüm testler geçiyor
- [ ] Kod PEP 8 ile uyumlu
- [ ] Docstring'ler yazılmış
- [ ] Yorum satırları açık
- [ ] Hata işlemesi yapılmış

### Pull Request Öncesi

- [ ] Kod İncelemesi yapıldı
- [ ] Test kapsamı yeterli
- [ ] SonarQube raporu kontrol edildi
- [ ] Dokümantasyon güncellendi
- [ ] Performans test edildi

## 13. Ek Kaynaklar

- [PEP 8 Python Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [SonarQube Documentation](https://docs.sonarqube.org/)
- [Doxygen Documentation](https://www.doxygen.nl/)

---

**Son Güncelleme**: 2026  
**Versiyon**: 1.0.0  
**Durum**: Aktif ve Uygulanıyor
