# Scrum Görev Tahtası (Task Board)

Bu dosya, projenin geliştirilme sürecindeki Scrum/Kanban tahtasını temsil eder. İşlerin durumunu takip etmek için güncellenecektir.

---

## 📋 YOL HARİTASI VE GÖREVLER

| Görev Kodu | Görev Tanımı | Öncelik | Durum | Atanan |
| :--- | :--- | :---: | :---: | :---: |
| **US-1** | Görüntü Yükleme Desteği (GIF, JPEG, PNG, vb.) | Yüksek | 🔄 In Progress | Antigravity |
| **US-2** | SIFT, SURF, AKAZE, ORB ile Copy-Move Tespiti | Kritik | ⏳ To Do | Antigravity |
| **US-3** | Yapay Zeka ile Tespit (CNN ve CNN-LSTM) | Kritik | ⏳ To Do | Antigravity |
| **DOC-1** | FSM (Function Point + COCOMO) Efor Hesabı | Yüksek |  Done | Antigravity |
| **DOC-2** | Doxygen Graphviz Dokümantasyonu Hazırlığı | Orta | ⏳ To Do | Antigravity |
| **DOC-3** | Kullanıcı El Kitabı (User Manual) Yazılması | Orta | ⏳ To Do | Antigravity |
| **UI-1** | Premium Glassmorphism Arayüz Tasarımı & UX | Yüksek | ⏳ To Do | Antigravity |
| **QUAL-1** | SonarQube Kod Kalitesi Hazırlıkları | Orta | ⏳ To Do | Antigravity |

---

## 🗂️ SÜTUNLAR

### ⏳ YAPILACAKLAR (TO DO)
- [ ] **US-2:** SIFT, SURF, AKAZE ve ORB algoritmalarını içeren `backend/algorithms/classical.py` geliştirilmesi.
- [ ] **US-3:** ELA-CNN ve SRM-CNN-LSTM modellerini barındıran `backend/algorithms/deep_learning.py` geliştirilmesi.
- [ ] **UI-1:** Modern CSS ve responsive HTML dashboard tasarımı (`frontend/index.html`, `frontend/css/style.css`).
- [ ] **DOC-2:** Kaynak kodda Doxygen tarzı yorum satırlarının yazılması ve `docs/Doxyfile` yapılandırılması.
- [ ] **DOC-3:** Program kullanımını ve algoritmaları anlatan `docs/user_manual.md` hazırlanması.
- [ ] **QUAL-1:** Kod standartlarının ve SonarQube analiz dosyasının hazırlanması.

### 🔄 YAPILIYOR (IN PROGRESS)
- [ ] **US-1:** Proje ana dizininin ve backend alt yapısının kurulması (`backend/main.py`, `backend/requirements.txt`).

### 🔍 DEĞERLENDİRME / TEST (IN REVIEW)
- *Henüz bu aşamada görev bulunmamaktadır.*

###  TAMAMLANANLAR (DONE)
- [x] **PLAN-1:** Proje mimarisinin ve `implementation_plan.md` planının oluşturulması.
- [x] **DOC-1:** FSM Efor Hesabı raporunun (`docs/fsm_estimation.md`) hazırlanması.
- [x] **US-1-PRE:** `backend/requirements.txt` dosyasının hazırlanması.
