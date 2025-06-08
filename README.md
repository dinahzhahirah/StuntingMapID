# Dashboard Clustering Stunting Indonesia 🗺️

Dashboard interaktif untuk analisis clustering prevalensi stunting di seluruh provinsi Indonesia menggunakan berbagai metode clustering dan visualisasi yang komprehensif.

## ✨ Fitur Utama

### 🎯 Fitur Inti
- **Upload Data**: Mendukung upload data CSV terbaru atau menggunakan data default 2023
- **Pemilihan Variabel**: Fleksibilitas memilih variabel untuk analisis clustering
- **Metode Clustering**: Support untuk Single, Complete, Average, Ward Linkage dan K-Means
- **Jumlah Cluster**: Dapat disesuaikan antara 3-7 cluster
- **Filter Cluster**: Filter visualisasi berdasarkan cluster tertentu
- **Pencarian Provinsi**: Cari dan highlight provinsi spesifik dengan informasi detail

### 📊 Menu Dashboard

#### 1. **Main Dashboard**
- KPI nasional (rata-rata stunting, jumlah cluster, provinsi tertinggi/terendah)
- Peta hasil clustering (scatter plot dengan visualisasi cluster)
- Bar chart rata-rata indikator per cluster
- Boxplot distribusi data per cluster
- Jumlah provinsi per cluster
- Interpretasi dan rekomendasi untuk setiap cluster

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
- Ringkasan statistik per cluster
- Download hasil dalam format CSV
- Visualisasi distribusi cluster (pie chart)
- Parallel coordinates plot
- Profil detail setiap cluster dengan radar chart

## 🚀 Cara Menjalankan

### Instalasi Dependencies
```bash
pip install -r requirements.txt
```

### Menjalankan Aplikasi
```bash
streamlit run dashboard_stunting.py
```

### Atau dengan custom port
```bash
streamlit run dashboard_stunting.py --server.port 8501
```

## 📁 Struktur File

```
📦 Dashboard Stunting
├── 📄 dashboard_stunting.py      # File utama aplikasi
├── 📄 requirements.txt           # Dependencies Python
├── 📄 Stunting2023.csv          # Data sample
├── 📄 README.md                 # Dokumentasi
└── 📄 template_stunting.csv     # Template upload data
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
- Pilih tema (Dark/Light) menggunakan tombol 🌙/☀️
- Upload data baru atau gunakan data default 2023

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

## 🔍 Interpretasi Hasil

### Kategori Risiko Stunting:
- **Risiko Sangat Tinggi**: > 30%
- **Risiko Tinggi**: 25-30%
- **Risiko Sedang**: 20-25%
- **Risiko Rendah**: < 20%

### Metrik Validasi:
- **Silhouette Score**: Semakin tinggi (mendekati 1) semakin baik
- **Rasio Sw/Sb**: Semakin rendah semakin baik
- **AU Bootstrap**: > 95% menunjukkan stabilitas cluster yang baik

## 📊 Visualisasi yang Tersedia

1. **Scatter Plot Clustering**: Menampilkan hasil cluster dalam 2D
2. **Bar Chart Horizontal**: Perbandingan rata-rata indikator per cluster
3. **Box Plot**: Distribusi data setiap indikator per cluster
4. **Dendrogram**: Visualisasi hierarchical clustering
5. **Silhouette Plot**: Evaluasi kualitas cluster
6. **Pie Chart**: Distribusi provinsi per cluster
7. **Parallel Coordinates**: Profil multidimensional cluster
8. **Radar Chart**: Profil karakteristik setiap cluster

## 🔧 Troubleshooting

### Error Umum:
1. **File upload gagal**: Pastikan format CSV dan kolom sesuai template
2. **Visualisasi tidak muncul**: Refresh browser atau pilih ulang parameter
3. **Error clustering**: Pastikan minimal 2 variabel dipilih dan data numerik valid

### Tips Optimasi:
- Gunakan data yang sudah dibersihkan untuk hasil terbaik
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
- ✅ Dark/Light mode dengan toggle
- ✅ Pencarian dan filter provinsi
- ✅ Download hasil dan template
- ✅ Validasi cluster comprehensive
- ✅ Interpretasi otomatis hasil clustering

---
