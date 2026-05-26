# Kullanıcı El Kitabı

## 1. Proje Hakkında
Bu proje, görüntü sahteciliğinin tespiti için geliştirilmiş bir başlangıç uygulamasıdır. Yazılım, klasik OpenCV tabanlı özellik çıkarma algoritmaları ve basit AI modelleri kullanarak görüntüleri analiz eder.

## 2. Desteklenen Dosya Formatları
- JPG
- JPEG
- PNG
- GIF

## 3. Kurulum
1. Python 3.11 veya daha yeni bir sürüm kurun.
2. Proje dizininde terminal açın.
3. Aşağıdaki komutu çalıştırın:

```bash
pip install -r backend/requirements.txt
```

## 4. Uygulamayı Çalıştırma
Aşağıdaki komutu kullanarak API'yi başlatın:

```bash
uvicorn backend.main:app --reload
```

Tarayıcıda `http://127.0.0.1:8000/` adresine giderek kullanıcı arayüzünü kullanabilirsiniz.

## 5. Kullanım
1. Ana ekranda "Görüntü seçin" alanına tıklayın.
2. Bilgisayarınızdan JPG, PNG veya GIF formatındaki bir görüntü dosyası seçin.
3. "Gönder" butonuna basın.
4. Sonuç bölümünde klasik algoritma ve AI tabanlı tespit sonuçlarını görüntüleyebilirsiniz.

## 6. Ek Notlar
- Klasik algoritma sonuçları SIFT, SURF, AKAZE ve ORB özellik çıkarmaya dayanmaktadır.
- AI sonuçları, CNN ve LSTM temelli basit model yapıları kullanılarak hesaplanmak üzere proje iskeletine eklenmiştir.
- Gelecekte eğitimli modeller eklemek ve model ağırlıkları yüklemek mümkündür.
