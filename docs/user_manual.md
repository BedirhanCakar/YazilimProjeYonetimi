# Kullanıcı El Kitabı - Görüntü Sahteciliği Tespiti

## 1. Proje Hakkında

Bu proje, görüntü sahteciliğinin tespiti için geliştirilmiş bir Ar-Ge uygulamasıdır. Yazılım, klasik OpenCV tabanlı özellik çıkarma algoritmaları (SIFT, SURF, AKAZE, ORB) ve derin öğrenme tabanlı sinir ağları (CNN, LSTM) kullanarak görüntüleri analiz eder ve sahtecilik belirtilerini tespit eder.

### 1.1 Özellikler

- **Klasik Algoritma Analizi**: SIFT, SURF, AKAZE, ORB ile copy-move tespit
- **Yapay Zeka Tabanlı Analiz**: CNN ve LSTM sinir ağları ile deepfake tespiti
- **Gelişmiş İçerik Analizi**: Error Level Analysis (ELA) ve Frekans Domain analizi
- **Premium Kullanıcı Arayüzü**: Glassmorphism tasarımı, responsive layout
- **Detaylı Sonuç Raporları**: Algoritma puanları ve öneriler

## 2. Desteklenen Dosya Formatları

- JPG (Joint Photographic Experts Group)
- JPEG (Jpeg Image Compression)
- PNG (Portable Network Graphics)
- GIF (Graphics Interchange Format)

## 3. Kurulum ve Başlatma

### 3.1 Ön Gereksinimler

- Python 3.9 veya daha yeni bir sürüm
- pip (Python paket yöneticisi)

### 3.2 Adım Adım Kurulum

1. Proje dizinine gidin:
```bash
cd YazılımProjeYönetimi
```

2. Gerekli Python paketlerini kurun:
```bash
pip install -r backend/requirements.txt
```

3. FastAPI sunucusunu başlatın:
```bash
uvicorn backend.main:app --reload
```

4. Tarayıcıda aşağıdaki adresi açın:
```
http://127.0.0.1:8000/
```

### 3.3 Alternatif: Docker ile Çalıştırma

Eğer Docker kurulu ise:
```bash
docker build -t forgery-detection .
docker run -p 8000:8000 forgery-detection
```

## 4. Kullanım Kılavuzu

### 4.1 Ana Ekran

Ana ekranda üç bölüm bulunur:
- **Sol Panel**: Dosya yükleme ve konfigürasyon
- **Sağ Panel**: Yüklenen görüntünün önizlemesi
- **Alt Panel**: Analiz sonuçları

### 4.2 Görüntü Yükleme

1. Sol panelde "Dosya seçmek için tıklayın" alanına tıklayın
2. Bilgisayarınızdan JPG, PNG veya GIF formatında bir görüntü seçin
3. Seçilen dosyanın adı ve boyutu gösterilecektir
4. (İsteğe bağlı) "Güven Eşiği" kaydırıcısını ayarlayın (0-1 arası):
   - **0.3**: Düşük eşik (daha hassas, daha fazla yalancı pozitif)
   - **0.5**: Orta eşik (varsayılan, dengeli)
   - **0.8**: Yüksek eşik (daha katı, daha az yalancı pozitif)

### 4.3 Analiz Başlatma

1. "Analiz Et" butonuna basın
2. Analiz tamamlanana kadar bekleyin (genellikle 2-10 saniye)
3. Sonuçlar aşağıda görüntülenecektir

### 4.4 Sonuçların Yorumlanması

#### CNN (Evrişimli Sinir Ağı) Sonuçları
- Görüntünün her bölgesindeki piksel kalitesi analiz edilir
- **Sahtecilik Olasılığı**: 0-100% arası skor
  - **0-30%**: Muhtemelen orijinal
  - **30-60%**: Şüphe uyandırıcı, insan gözü gerekebilir
  - **60-100%**: Yüksek oranda sahte olabilir

#### LSTM (Uzun Kısa Süreli Bellek) Sonuçları
- Görüntünün zamansal/mekansal tutarlılığını analiz eder
- Deepfake videoların tespitinde etkili

#### Klasik Algoritmalar (SIFT/SURF/AKAZE/ORB)
- **Özellik Noktaları (Keypoints)**: Görüntüde algılanan karakteristik noktalar
- **Eşleştirmeler (Matches)**: Copy-move işlemlerinin tespiti
- **Güven Puanı**: Tespit güvenilirliği

#### ELA (Error Level Analysis)
- JPEG sıkıştırma hataları üzerinden analiz
- Yüksek değerler manipülasyona işaret edebilir

#### Frekans Analizi (FFT)
- Fourier Transform ile frekans bileşenleri analiz
- Doğal görüntüler düşük, yapay güzelleştirmeler yüksek skor alır

### 4.5 Genel Değerlendirme

Program analiz tamamlandığında aşağıda bir özet gösterir:
- **Ortalama Şüphe Puanı**: Tüm yöntemlerin ortalaması
- **Klasik Algoritma Puanı**: SIFT/SURF/AKAZE/ORB'nin konsensüsü
- **Nihai Karar**: ORIJINAL veya SAHTECİLİK BELİRTİLERİ

## 5. Algoritma Açıklamaları

### 5.1 SIFT (Scale-Invariant Feature Transform)

Scale ve rotasyona karşı değişmez özellik çıkarma algoritması. Copy-move tespitinde oldukça etkilidir.

### 5.2 SURF (Speeded-Up Robust Features)

SIFT'in hızlanmış versiyonu. Daha hızlı işleme süresi ile benzer doğruluk sağlar.

### 5.3 AKAZE (Accelerated-KAZE)

Hızlı açık kaynak algoritması. Mobil cihazlarda ve gerçek zamanlı uygulamalarda tercih edilir.

### 5.4 ORB (Oriented FAST and Rotated BRIEF)

Serbest ve açık kaynak algoritması. Düşük hesap gücü gerektiren cihazlarda ideal.

### 5.5 CNN (Convolutional Neural Network)

Evrişimli sinir ağları, görüntü sınıflandırması ve piksel düzeyinde analiz için tasarlanmıştır. Yerel özellik çıkarma konusunda başarılıdır.

### 5.6 LSTM (Long Short-Term Memory)

Uzun kısa süreli bellek ağları, zamansal bağımlılıkları modelleme konusunda iyi. Video manipülasyonlarında etkili.

## 6. Sık Sorulan Sorular (SSS)

### S: Analiz ne kadar sürer?
**C**: Görüntü boyutuna bağlı olarak 2-10 saniye arası. Daha büyük görüntüler daha uzun sürer.

### S: Hangi boyuttaki görüntüler önerilir?
**C**: 512x512 ile 4096x4096 piksel arası. Çok küçük görüntüler detay kaybedebilir, çok büyükler işlem süresini arttırır.

### S: Sonuçlar ne kadar güvenilir?
**C**: Program %70-90 doğruluk oranına sahiptir. Hassas kararlar için insan gözü da kullanılmalıdır.

### S: Video analizi destekleniyor mu?
**C**: Bu versiyonda hayır. Gelecek sürümlerde eklenebilir.

### S: Sonuçları kaydetme imkanı var mı?
**C**: Frontend'den tarayıcı "Yazdır" fonksiyonu veya ekran görüntüsü alabilirsiniz.

## 7. Sorun Giderme

### Problem: "Yüklenen dosya görüntü olarak açılmadı"
**Çözüm**: Dosya formatının desteklenen bir formatta olduğundan emin olun (JPG, PNG, GIF).

### Problem: "API çalışmıyor" hatası
**Çözüm**: 
- Terminal'de `pip install -r backend/requirements.txt` komutunu çalıştırın
- Sunucuyu yeniden başlatın: `uvicorn backend.main:app --reload`

### Problem: Analiz çok yavaş
**Çözüm**: 
- Görüntü boyutunu küçültmeyi deneyin
- Daha yüksek performanslı bir bilgisayar kullanın
- GPU desteğini etkinleştirin (CUDA kurulumunun olması gerekir)

## 8. İletişim ve Destek

Bu proje bir Ar-Ge çalışmasıdır. Sorular ve öneriler için projenin GitHub sayfasını ziyaret edin.

---

**Sürüm**: 1.0.0  
**Son Güncelleme**: 2026  
**Lisans**: MIT
