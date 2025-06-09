# Dashboard Clustering Stunting Indonesia 🗺️

Aplikasi ini merupakan sebuah dashboard interaktif berbasis **Streamlit** yang dirancang untuk melakukan **analisis pengelompokan (clustering)** terhadap provinsi-provinsi di Indonesia berdasarkan prevalensi stunting dan sejumlah indikator sosial-ekonomi lainnya. Tujuan utama dari aplikasi ini adalah untuk membantu pengambil kebijakan dan pihak terkait dalam mengidentifikasi kelompok wilayah yang memiliki karakteristik serupa, sehingga intervensi dapat dilakukan secara lebih tepat sasaran dan efektif.

## ✨ Fitur Utama

### 🎯 Fitur Inti
- **Upload Data**: Mendukung upload data CSV terupdate atau menggunakan data default tahun 2023 
- **Pemilihan Variabel**: Fleksibilitas memilih variabel untuk analisis clustering
- **Metode Clustering**: Support untuk Hierarchical Clustering (Single, Complete, Average, dan Ward Linkage) dan K-Means
- **Jumlah Cluster**: Dapat disesuaikan antara 3-7 cluster
- **Filter Cluster**: Filter visualisasi berdasarkan cluster tertentu
- **Pencarian Provinsi**: Cari dan highlight provinsi spesifik dengan informasi detail
- Penanganan nilai yang hilang (missing value) melalui **imputasi berbasis centroid cluster terdekat**.
- Penentuan **prioritas intervensi** berdasarkan skor komposit dari seluruh indikator.
- Fitur ekspor hasil clustering dalam format CSV.



### 📊 Menu Dashboard

#### 1. **Main Dashboard**
- KPI nasional (rata-rata stunting, jumlah cluster, provinsi tertinggi/terendah)
- Peta hasil clustering (peta choropleth distribusi cluster per provinsi)
- Jumlah provinsi per cluster
- Bar chart rata-rata indikator per cluster
- Boxplot distribusi data per cluster
- Interpretasi dan rekomendasi pengabilan kebijakan untuk setiap cluster

#### 2. **Analisis Cluster**
- Preprocessing otomatis (duplikat, missing values, standardisasi)
- Pengecekan multikolinearitas dengan VIF
- Visualisasi dendrogram (untuk hierarchical clustering)
- Analisis silhouette score
- Plot perbandingan silhouette score untuk berbagai jumlah cluster

#### 3. **Validasi Cluster**
- Metrik Within-cluster variation (Sw) dan Between-cluster variation (Sb)
- Rasio Sw/Sb untuk evaluasi kualitas cluster
- Simulasi Multiscale Bootstrap analysis
- Informasi imputasi missing values
- Interpretasi hasil validasi

#### 4. **Hasil Clustering**
- Tabel lengkap hasil clustering dengan kategori risiko
- Download hasil dalam format CSV


## 🚀 Cara Menjalankan

### Instalasi Dependencies
```bash
pip install -r requirements.txt
```

### Menjalankan Aplikasi
```bash
streamlit run dashboard.py
```

### Atau dengan custom port
```bash
streamlit run dashboard.py --server.port 8501
```

## 📁 Struktur File

```
📦 StuntingMapID
├── 📄 dashboard_stunting.py      # File utama aplikasi
├── 📄 requirements.txt           # Dependencies Python
├── 📄 Stunting2023.csv          # Data sample
├── 📄 README.md                 # Dokumentasi
```

## 📋 Format Data

Data harus dalam format CSV dengan kolom berikut:

| Kolom | Deskripsi |
|-------|-----------|
| `Provinsi` | Nama provinsi |
| `Prevalensi Stunting` | Persentase stunting (%) |
| `Persentase Penduduk 15 Tahun Tidak Tamat SD` | Tingkat pendidikan rendah (%) |
| `Persentase Penduduk Miskin` | Tingkat kemiskinan (%) |
| `Ibu Hamil KEK` | Ibu hamil kurang energi kronis (%) |
| `Akses Sanitasi Layak` | Akses sanitasi yang memadai (%) |
| `Akses Air Minum Layak` | Akses air minum bersih (%) |
| `Akses Kesehatan Dasar` | Akses layanan kesehatan dasar (%) |

## 🎨 Panduan Penggunaan

### 1. **Setup Awal**
- Jalankan aplikasi dan tunggu hingga interface terbuka
- Upload data terupdate atau gunakan data default tahun 2023

### 2. **Konfigurasi Analisis**
- **Pemilihan Variabel**: Centang variabel yang ingin dianalisis
- **Metode Clustering**: Pilih dari dropdown (single, complete, average, ward, kmeans)
- **Jumlah Cluster**: Atur slider sesuai kebutuhan (3-7)

### 3. **Eksplorasi Data**
- **Filter Cluster**: Gunakan dropdown untuk fokus pada cluster tertentu
- **Pencarian Provinsi**: Ketik nama provinsi untuk melihat detail dan posisi cluster
- **Toggle Interpretasi**: Klik tombol 🔽/🔼 untuk membaca insight setiap cluster

### 4. **Analisis Mendalam**
- Kunjungi tab "Analisis Cluster" untuk melihat dendrogram dan silhouette analysis
- Periksa "Validasi Cluster" untuk memahami kualitas clustering
- Download hasil di tab "Hasil Clustering"


## 🔧 Troubleshooting

### Error Umum:
1. **File upload gagal**: Pastikan format CSV dan kolom sesuai template
2. **Visualisasi tidak muncul**: Refresh browser atau pilih ulang parameter
3. **Error clustering**: Pastikan minimal 2 variabel dipilih dan data numerik valid

### Tips Optimasi:
- Pilih variabel yang relevan dan tidak redundan
- Eksperimen dengan berbagai metode clustering untuk perbandingan
- Perhatikan interpretasi silhouette score untuk menentukan jumlah cluster optimal

## 📞 Support

Untuk pertanyaan atau masalah teknis, silakan:
1. Periksa log error di terminal
2. Pastikan semua dependencies terinstall
3. Verifikasi format data input

## 📝 Changelog

### Version 1.0
- ✅ Implementasi lengkap semua fitur yang diminta
- ✅ Support untuk 5 metode clustering
- ✅ Pencarian dan filter provinsi
- ✅ Download hasil dan template
- ✅ Validasi cluster comprehensive
- ✅ Interpretasi otomatis hasil clustering

---
