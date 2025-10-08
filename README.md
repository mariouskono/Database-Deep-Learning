# 🔗 Deep Learning Database Integration (SQL, NoSQL, Cloud DB)

Proyek ini mengeksplorasi **tiga pendekatan integrasi database dengan model deep learning**, yaitu:

1. **SQL Database (SQLite3)** – untuk dataset terstruktur kecil.
2. **NoSQL Database (MongoDB)** – untuk data teks tidak terstruktur.
3. **Cloud Database** – untuk dataset besar dan kompleks pada layanan cloud.

Tujuannya adalah membandingkan efektivitas, kinerja, dan integrasi ketiganya dalam konteks pengembangan model deep learning.

---

## 📂 Struktur Proyek

```
📦 DeepLearning-Database-Integration
├── load_iris.py                          # Implementasi DL dengan SQLite3 (dataset Iris)
├── nosql.py                              # Implementasi DL dengan MongoDB (dataset IMDB)
├── Cloud DB - Deep Learning.ipynb        # Implementasi DL dengan Cloud DB (dataset CIFAR-10)
├── README.md                             # Dokumentasi proyek
```

---

## ⚙️ Deskripsi Setiap Pendekatan

### 🧮 1. SQL Database – *Iris Dataset*

* **Database:** SQLite3
* **Model:** Neural Network sederhana
* **Akurasi:**

  * Training: 96.9%
  * Validation: 100%
  * Testing: 96.7%
* **Analisis:**
  Model mampu melakukan generalisasi dengan baik. Confusion matrix menunjukkan prediksi hampir sempurna.
* **Kesimpulan:**
  SQL sangat efektif untuk dataset kecil dan terstruktur. Integrasinya ringan, efisien, dan mudah diimplementasikan.

---

### 💬 2. NoSQL Database – *IMDB Movie Review*

* **Database:** MongoDB
* **Model:** LSTM (Long Short-Term Memory)
* **Akurasi:**

  * Training: 98.6%
  * Testing: 98.68%
* **Analisis:**
  Precision, recall, dan F1-score mendekati 0.99 untuk semua kelas.
* **Kesimpulan:**
  MongoDB unggul untuk data teks tidak terstruktur. Integrasi via `pymongo` memudahkan pipeline, meskipun preprocessing teks cukup kompleks.

---

### ☁️ 3. Cloud Database – *CIFAR-10*

* **Database:** Cloud DB (terintegrasi dengan layanan cloud)
* **Model:** CNN (Convolutional Neural Network)
* **Akurasi:**

  * Training: ~89%
  * Validation: 50–61%
  * Testing: 59%
* **Analisis:**
  Terjadi **overfitting**, menunjukkan arsitektur CNN sederhana tidak cukup untuk dataset gambar besar.
* **Kesimpulan:**
  Cloud DB unggul untuk skalabilitas dan kolaborasi, tetapi membutuhkan arsitektur DL yang lebih kuat untuk performa maksimal.

---

## 📊 Perbandingan Tiga Pendekatan

| Aspek         | SQL (SQLite3)   | NoSQL (MongoDB)                | Cloud DB               |
| ------------- | --------------- | ------------------------------ | ---------------------- |
| Jenis Data    | Terstruktur     | Tidak terstruktur (teks)       | Kompleks (gambar)      |
| Skala Dataset | Kecil–menengah  | Besar                          | Sangat besar           |
| Model DL      | Neural Network  | LSTM                           | CNN                    |
| Akurasi       | 96.7%           | 98.68%                         | 59%                    |
| Keunggulan    | Ringan & cepat  | Fleksibel & efisien untuk teks | Scalable & kolaboratif |
| Kelemahan     | Tidak fleksibel | Butuh preprocessing berat      | Rentan overfitting     |

---

## 🧭 Kesimpulan Umum (5.1)

Setiap jenis database memiliki kekuatan dan kelemahan masing-masing:

* **SQL** → Cocok untuk dataset tabular kecil dan terstruktur.
* **NoSQL** → Ideal untuk teks besar dan tidak terstruktur.
* **Cloud DB** → Unggul untuk dataset besar dan kompleks, tetapi memerlukan model DL canggih.

Pemilihan database harus disesuaikan dengan **jenis data dan tujuan proyek deep learning.**

---

## 💡 Saran Pengembangan (5.2)

### Untuk SQL Database

* Gunakan **PostgreSQL/MySQL** untuk dataset yang lebih besar.
* Terapkan **indexing** agar query lebih cepat.

### Untuk NoSQL Database

* Gunakan **ElasticSearch** untuk mempercepat pencarian teks.
* Coba model **Transformer** (BERT, IndoBERT) untuk performa NLP yang lebih tinggi.

### Untuk Cloud Database

* Gunakan arsitektur **VGG, ResNet, atau EfficientNet**.
* Terapkan **data augmentation** dan **early stopping**.
* Integrasikan dengan layanan **MLOps Cloud** seperti Google AI Platform, AWS Sagemaker, atau Azure ML.

### Saran Umum

* Eksplorasi **hybrid database** (SQL + NoSQL).
* Fokus pada efisiensi biaya dan waktu dalam penggunaan layanan cloud.

---

## 🧑‍💻 Pengembang

**Nama:** Bertnardo Mario Uskono
**Universitas:** Universitas Bhayangkara Jakarta Raya
**Fokus:** Machine Learning & Data Science

---

## 📜 Lisensi

Proyek ini dibuat untuk tujuan akademik dan pembelajaran. Bebas digunakan dan dimodifikasi dengan mencantumkan sumber asli.

---
