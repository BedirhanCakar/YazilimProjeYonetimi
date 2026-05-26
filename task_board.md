# Scrum Görev Tahtası (Task Board)

Bu dosya, projenin geliştirilme sürecindeki Scrum/Kanban tahtasını temsil eder. İşlerin durumunu takip etmek için güncellenecektir.

---

## 📋 YOL HARİTASI VE GÖREVLER

| Görev Kodu | Görev Tanımı | Öncelik | Durum | Atanan |
| :--- | :--- | :---: | :---: | :---: |
| **US-1** | Görüntü Yükleme Desteği (GIF, JPEG, PNG, vb.) | Yüksek | ✅ Done | Antigravity |
| **US-2** | SIFT, SURF, AKAZE, ORB ile Copy-Move Tespiti | Kritik | ✅ Done | Antigravity |
| **US-3** | Yapay Zeka ile Tespit (CNN ve CNN-LSTM) | Kritik | ✅ Done | Antigravity |
| **DOC-1** | FSM (Function Point + COCOMO) Efor Hesabı | Yüksek |  Done | Antigravity |
| **DOC-2** | Doxygen Graphviz Dokümantasyonu Hazırlığı | Orta | ⏳ To Do | Antigravity |
| **DOC-3** | Kullanıcı El Kitabı (User Manual) Yazılması | Orta | ✅ Done | Antigravity |
| **UI-1** | Premium Glassmorphism Arayüz Tasarımı & UX | Yüksek | ✅ Done | Antigravity |
| **QUAL-1** | SonarQube Kod Kalitesi Hazırlıkları | Orta | ⏳ To Do | Antigravity |

---

## 🗂️ SÜTUNLAR

### ⏳ YAPILACAKLAR (TO DO)
- [ ] **DOC-2:** Kaynak kodda Doxygen tarzı yorum satırlarının yazılması ve `docs/Doxyfile` yapılandırılması.
- [ ] **QUAL-1:** Kod standartlarının ve SonarQube analiz dosyasının hazırlanması.

### 🔄 YAPILIYOR (IN PROGRESS)
- [ ] **UI-1:** Premium CSS ve responsive HTML dashboard tasarımı (`frontend/index.html`, `frontend/css/style.css`) - tamamlandı.

### 🔍 DEĞERLENDİRME / TEST (IN REVIEW)
- *Henüz bu aşamada görev bulunmamaktadır.*

###  TAMAMLANANLAR (DONE)
- [x] **PLAN-1:** Proje mimarisinin ve `implementation_plan.md` planının oluşturulması.
- [x] **DOC-1:** FSM Efor Hesabı raporunun (`docs/fsm_estimation.md`) hazırlanması.
- [x] **US-1-PRE:** `backend/requirements.txt` dosyasının hazırlanması.
- [x] **US-1:** Proje ana dizininin, FastAPI backend'i ve görüntü yükleme altyapısının oluşturulması.
- [x] **US-2:** SIFT, SURF, AKAZE ve ORB algoritmaları ile copy-move tespit modülü (`backend/algorithms/classical.py`) geliştirilmesi.
- [x] **US-3:** ELA-CNN ve LSTM temelli yapay zeka modellerini barındıran `backend/algorithms/deep_learning.py` geliştirilmesi.
- [x] **DOC-3:** Program kullanımını ve algoritmaları anlatan `docs/user_manual.md` hazırlanması.
- [x] **UI-1:** Modern CSS (glassmorphism), responsive HTML ve interaktif JS arayüzü (`frontend/index.html`, `frontend/css/style.css`, `frontend/js/app.js`) tamamlanması.
