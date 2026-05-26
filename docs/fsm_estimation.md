# Yazılım Emeği Hesaplama Raporu (FSM Yöntemi)

Bu belgede, **Görüntü Sahteciliği Tespiti (Ar-Ge Projesi)** yazılımının işlevsel boyutu ve geliştirilmesi için gereken emek (adam-saat) hesabı yapılmıştır. Efor kestiriminde **İşlevsel Boyut Ölçümü (Functional Size Measurement - FSM)** yöntemlerinden **İşlem Puanı (Function Point - FP)** analizi ve ardından **COCOMO II (Constructive Cost Model)** modeli kullanılmıştır.

---

## 1. İŞLEVSEL BOYUT ÖLÇÜMÜ: İŞLEM PUANI (FUNCTION POINT - FP) ANALİZİ

İşlem Puanı yöntemi, yazılımın teknik detaylarından bağımsız olarak, kullanıcıya sunduğu işlevsel bileşenleri temel alarak boyutu ölçer. Beş temel işlevsel bileşen üzerinden analiz yapılmıştır.

### 1.1. Bileşenlerin Belirlenmesi ve Sınıflandırılması

#### A. Harici Girdiler (External Inputs - EI)
Sistem veri tabanını veya çalışma durumunu değiştiren kullanıcı girdileridir.
1. **Resim Dosyası Yükleme (EI-1):** Kullanıcının GIF, JPEG, PNG vb. formatlarındaki dosyaları sisteme yüklemesi. (Düşük Karmaşıklık)
2. **Klasik Algoritma Parametre Ayarları (EI-2):** SIFT/SURF/AKAZE/ORB algoritmaları için eşik değerler, RANSAC toleransı ve kümeleme hassasiyet ayarları. (Düşük Karmaşıklık)
3. **Derin Öğrenme Model Seçim ve Parametre Ayarları (EI-3):** ELA kalitesi, model tipi (CNN, LSTM) seçimi. (Düşük Karmaşıklık)

#### B. Harici Çıktılar (External Outputs - EO)
Kullanıcıya sunulan ve hesaplama/işleme mantığı barındıran veri çıktılarıdır.
1. **Görsel Eşleşme ve Bölge İşaretleme (EO-1):** Copy-move tespiti sonrasında eşleşen noktaların çizilmesi ve manipüle edilen alanların kutu içine alınması. (Orta Karmaşıklık)
2. **AI Isı Haritası ve Güven Skoru (EO-2):** ELA ve gürültü analizine göre piksel bazlı manipülasyon olasılığını gösteren ısı haritasının (heatmap) ve olasılık skorunun dinamik olarak çizilmesi. (Orta Karmaşıklık)
3. **Analiz Sonuç Raporu (EO-3):** Analiz edilen resmin, kullanılan yöntemlerin, doğruluk oranlarının ve sürelerin PDF formatında dışa aktarılması. (Orta Karmaşıklık)

#### C. Harici Sorgular (External Inquiries - EQ)
Doğrudan veri tabanından veya bellekten çekilen, üzerinde hesaplama yapılmayan bilgi sorgularıdır.
1. **Algoritma Açıklamaları ve Yardım Ekranı (EQ-1):** Kullanıcının arayüzden her bir algoritmanın (SIFT, SURF, CNN vb.) çalışma prensibini sorgulaması/görüntülemesi. (Düşük Karmaşıklık)

#### D. Dahili Mantıksal Dosyalar (Internal Logical Files - ILF)
Yazılımın kendi bünyesinde tuttuğu ve güncellediği mantıksal veri gruplarıdır.
1. **Analiz Geçmişi ve Oturum Verileri (ILF-1):** Geçici bellek veya hafif veri tabanında (SQLite/JSON) tutulan analiz geçmişi, yüklenen resim yolları ve sonuç özetleri. (Düşük Karmaşıklık)

#### E. Harici Arayüz Dosyaları (External Interface Files - EIF)
Yazılımın dışarıdan referans aldığı veya entegre olduğu kütüphane/veri dosyalarıdır.
1. **OpenCV Kütüphanesi Entegrasyonu (EIF-1):** Görüntü okuma, önişleme ve klasik algoritmalar için kullanılan API entegrasyonu. (Düşük Karmaşıklık)
2. **PyTorch AI Ağırlık Dosyaları (EIF-2):** CNN ve LSTM modellerinin önceden eğitilmiş ağırlık parametre dosyaları. (Düşük Karmaşıklık)
3. **PDF Rapor Oluşturucu Arayüzü (EIF-3):** Raporlama kütüphanesi entegrasyonu. (Düşük Karmaşıklık)

---

### 1.2. Düzeltilmemiş İşlem Puanı (Unadjusted Function Point - UFP) Hesaplama

Fonksiyon puanı standart ağırlık tablosuna göre bileşen puanları çarpılarak toplanır:

| Bileşen Tipi | Adet | Düşük Ağırlık | Orta Ağırlık | Yüksek Ağırlık | Toplam FP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Harici Girdi (EI)** | 3 | 3 x 3 = 9 | - | - | **9** |
| **Harici Çıktı (EO)** | 3 | - | 3 x 4 = 12 | - | **12** |
| **Harici Sorgu (EQ)** | 1 | 1 x 3 = 3 | - | - | **3** |
| **Dahili Mantıksal Dosya (ILF)** | 1 | 1 x 7 = 7 | - | - | **7** |
| **Harici Arayüz Dosyası (EIF)** | 3 | 3 x 5 = 15 | - | - | **15** |
| **TOPLAM UFP** | | | | | **46** |

---

### 1.3. Değer Ayarlama Faktörü (Value Adjustment Factor - VAF) Hesaplama

Sistemin genel özelliklerini değerlendiren 14 Sistem Karakteristiği (GSC) 0-5 arası puanlanmıştır:

1. Veri İletişimi (Data Communications): 2 (API üzerinden veri aktarımı var)
2. Dağıtık Veri İşleme (Distributed Data Processing): 1 (Sunucu-İstemci mimarisi)
3. Performans Hedefleri (Performance): 4 (Görüntü işleme ve AI modellerinin hızlı çalışması kritik)
4. Ağır Kullanılan Donanım Yapılandırması (Heavily Used Configuration): 2 (AI çıkarımı için CPU/GPU kullanımı)
5. İşlem Oranı (Transaction Rate): 2 (Çok yüksek anlık istek beklenmiyor)
6. Çevrimiçi Veri Girişi (On-Line Data Entry): 4 (Arayüz üzerinden doğrudan dosya yükleme ve parametre ayarı)
7. Son Kullanıcı Etkinliği (End-User Efficiency): 5 (Premium UI/UX, sürükle-bırak ve anlık canvas görselleştirme)
8. Çevrimiçi Güncelleme (On-Line Update): 2 (Veri tabanı güncellemeleri sınırlı)
9. Karmaşık İşleme (Complex Processing): 4 (Matematiksel eşleştirmeler, ELA fark hesapları, AI çıkarımları)
10. Yeniden Kullanılabilirlik (Reusability): 3 (Algoritmaların modüler yapıda olması)
11. Kurulum Kolaylığı (Installation Ease): 3 (Gereksinimler belgesi ile standart pip kurulumu)
12. İşletim Kolaylığı (Operational Ease): 4 (Tek tıkla başlatma, otomatik rapor üretimi)
13. Çoklu Konumda Kullanım (Multiple Sites): 1 (Yerel makine veya tek sunucu odaklı)
14. Değişime Kolaylık (Facilitate Change): 3 (Açık kaynak mimari, yeni AI modelleri eklemeye uygun)

**GSC Toplam Puanı = 37**

$$VAF = 0.65 + (0.01 \times \text{GSC}) = 0.65 + (0.01 \times 37) = 1.02$$

### 1.4. Düzeltilmiş İşlem Puanı (Adjusted Function Point - AFP)

$$AFP = UFP \times VAF = 46 \times 1.02 = 46.92 \text{ FP}$$

---

## 2. COCOMO II İLE EMEK VE ZAMAN TAHMİNİ

İşlem Puanı (FP) değerini, kod satır sayısına (SLOC) dönüştürmek için Python dili için standart çarpan olan **53 SLOC/FP** oranı kabul edilmiştir (QSM standartları).

### 2.1. Kod Satırı (SLOC) Tahmini

$$\text{SLOC} = AFP \times 53 = 46.92 \times 53 \approx 2487 \text{ Satır}$$
$$\text{KLOC (Bin Satır)} = 2.49$$

### 2.2. COCOMO II Sabitleri (Uygulama Odaklı - Organic Proje Modeli)

Küçük, yerleşik ve deneyimli ekipler tarafından geliştirilen bu tür projeler "Organic" (Uyumlu) sınıfa girmektedir.
- $A$ (Katsayı) = 2.94
- $B$ (Ölçek Faktörü) = 1.05
- $C$ (Zaman Katsayısı) = 2.5
- $D$ (Zaman Üssü) = 0.38
- $EAF$ (Efor Düzeltme Faktörü) = 1.0 (Ar-Ge projesi olduğu için standart koşullar nominal kabul edilmiştir)

### 2.3. Efor Hesaplama (Person-Month / Adam-Ay)

$$\text{Efor (Adam-Ay)} = A \times (\text{KLOC})^B \times EAF$$
$$\text{Efor} = 2.94 \times (2.49)^{1.05} \times 1.0 = 2.94 \times 2.62 \approx 7.7 \text{ Adam-Ay}$$

### 2.4. Geliştirme Süresi Hesaplama (Time to Develop - TDEV)

$$\text{Süre (Ay)} = C \times (\text{Efor})^D$$
$$\text{Süre} = 2.5 \times (7.7)^{0.38} \approx 2.5 \times 2.17 \approx 5.43 \text{ Ay}$$

### 2.5. Adam-Saat Cinsinden Toplam Emek Hesabı

Yazılım sektöründe standart kabul edilen **1 Adam-Ay = 152 Çalışma Saati** (haftada 5 gün, günde 8 saat, resmi tatiller düşüldüğünde) formülü esas alınmıştır.

$$\text{Toplam Emek (Adam-Saat)} = \text{Efor (Adam-Ay)} \times 152$$
$$\text{Toplam Emek} = 7.7 \times 152 = 1170.4 \text{ Adam-Saat}$$

---

## 3. ÖZET SONUÇLAR VE YORUM

- **Toplam İşlevsel Boyut:** 46.92 Function Point (FP)
- **Tahmini Kod Satırı:** ~2,487 Satır Python/JS/HTML
- **Hesaplanan Efor:** 7.7 Adam-Ay
- **Proje Süresi:** ~5.4 Ay (Tek geliştirici için)
- **Toplam Emek (Adam-Saat):** **1170.4 Adam-Saat**

**Değerlendirme:** Ar-Ge projesinde yer alan derin öğrenme model mimarilerinin araştırılması, eğitilmesi, OpenCV entegrasyonu ve premium görsel arayüzün hazırlanması süreçleri göz önüne alındığında, projenin tek bir mühendis tarafından tam zamanlı çalışılarak yaklaşık **1170.4 adam-saatte** tamamlanabileceği öngörülmektedir.
