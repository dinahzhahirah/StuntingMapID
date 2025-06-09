import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.utils import resample
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist, squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
import json
import base64
import io
from datetime import datetime
import requests
import geopandas as gpd

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Dashboard Clustering Stunting Indonesia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #666;
    }
    .cluster-interpretation {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}
.element-container {
    margin-bottom: 0rem !important;
}
iframe {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'cluster_results' not in st.session_state:
    st.session_state.cluster_results = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 5
if 'clustering_method' not in st.session_state:
    st.session_state.clustering_method = 'complete'

# Load default data function
@st.cache_data
def load_default_data():
    """Load default data with error handling"""
    # Create default data if file not found
    default_data = {
        'Provinsi': ['Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 'Sumatera Selatan', 
                        'Bengkulu', 'Lampung', 'Kepulauan Bangka Belitung', 'Kepulauan Riau', 'DKI Jakarta', 
                        'Jawa Barat', 'Jawa Tengah', 'DI Yogyakarta', 'Jawa Timur', 'Banten', 'Bali', 
                        'Nusa Tenggara Barat', 'Nusa Tenggara Timur', 'Kalimantan Barat', 'Kalimantan Tengah', 
                        'Kalimantan Selatan', 'Kalimantan Timur', 'Kalimantan Utara', 'Sulawesi Utara', 
                        'Sulawesi Tengah', 'Sulawesi Selatan', 'Sulawesi Tenggara', 'Gorontalo', 'Sulawesi Barat', 
                        'Maluku', 'Maluku Utara', 'Papua Barat', 'Papua Barat Daya', 'Papua', 'Papua Selatan', 
                        'Papua Tengah', 'Papua Pegunungan'],
            'Prevalensi Stunting': [20.3, 13.2, 17.1, 10.3, 9.5, 14.3, 14.4, 11.3, 15.1, 13.1, 13.1, 16.6, 
                                  16.0, 14.4, 13.8, 16.9, 5.5, 19.3, 26.2, 17.3, 15.9, 16.9, 17.1, 12.8, 
                                  14.8, 19.6, 20.7, 20.6, 18.4, 22.8, 18.3, 16.0, 16.8, 20.3, 18.2, 15.8, 
                                  18.6, 17.1],
            'Persentase Penduduk 15 Tahun Tidak Tamat SD': [8.13, 6.95, 11.75, 7.04, 7.23, 10.61, 10.7, 12.01, 
                                                           14.17, 5.23, 2.67, 6.55, 11.43, 10.82, 10.65, 6.31, 
                                                           6.37, 12.01, 15.69, 14.37, 8.6, 11.49, 5.4, 6.21, 
                                                           10.08, 7.15, 11.09, 8.14, 19.76, 13.36, 5.51, 9.24, 
                                                           6.1, np.nan, 3.36, np.nan, np.nan, np.nan],
            'Persentase Penduduk Miskin': [14.45, 8.15, 5.95, 6.68, 7.58, 11.78, 14.04, 11.11, 4.52, 5.69, 
                                         4.44, 7.62, 10.77, 11.04, 10.35, 6.17, 4.25, 13.85, 19.96, 6.71, 
                                         5.11, 4.29, 6.11, 6.45, 7.38, 12.41, 8.7, 11.43, 15.15, 11.49, 
                                         16.42, 6.46, 20.49, np.nan, 26.03, np.nan, np.nan, np.nan],
            'Ibu Hamil KEK': [10.4, 13.4, 16.5, 17.9, 12.7, 19.2, 10.1, 17.2, 18.4, 12.2, 11.7, 11.6, 24.6, 
                            21.4, 19.6, 5.4, 8.6, 15.7, 28.0, 11.1, 20.0, 15.9, 12.2, 5.2, 10.5, 21.3, 
                            19.7, 19.6, 19.7, 21.5, 21.2, 18.1, 23.8, 26.7, 10.0, 28.2, 7.6, 44.7],
            'Akses Sanitasi Layak': [75, 77, 79, 80, 78, 74.5, 86.43, 88.73, 85, 91.23, 94.01, 75.1, 85, 
                                   96.71, 86.7, 76, 90, 68, 65, 78, 77.54, 79, 80, 75, 83, 74, 82, 70, 
                                   72, 80.63, 70, 66, 62, 81.68, 60, 60.85, 41.44, 12.61],
            'Akses Air Minum Layak': [89.74, 92.19, 85.59, 90.47, 80.02, 87.19, 73.08, 82.78, 81.64, 92.1, 
                                    99.42, 93.86, 93.76, 96.69, 96.01, 92.95, 98.31, 96.03, 88.35, 82.08, 
                                    77.72, 76.29, 87.9, 90.19, 94.37, 86.85, 92.12, 94.8, 96, 79.86, 
                                    92.98, 89.01, 81.57, np.nan, 66.49, np.nan, np.nan, np.nan],
            'Akses Kesehatan Dasar': [72.59, 73.92, 88.77, 72.29, 73.41, 75.24, 81.41, 79.39, 88.17, 85.97, 
                                    76.83, 82.03, 86.15, 84.29, 83.37, 82.79, 90.54, 75.48, 43.5, 77.59, 
                                    75.67, 83.31, 78.87, 76.24, 86.48, 78.81, 85.52, 84.74, 82.44, 77.18, 
                                    74.11, 83.62, 65.38, np.nan, 31.78, np.nan, np.nan, np.nan]
        }
    return pd.DataFrame(default_data)

# Preprocessing function
def preprocess_data(df, selected_features):
    """Preprocess data with standardization"""
    # Remove duplicates
    df_clean = df.drop_duplicates()
    
    # Handle missing values
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    removed_rows = initial_rows - len(df_clean)
    
    # Prepare features for clustering
    feature_cols = [col for col in selected_features if col != 'Provinsi']
    X = df_clean[feature_cols].copy()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df_clean.index)
    
    return df_clean, X_scaled_df, scaler, removed_rows

# Clustering function
def perform_clustering(X_scaled, method, n_clusters):
    """Perform clustering using selected method"""
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(X_scaled)
    else:
        # Hierarchical clustering
        linkage_methods = {
            'single': 'single',
            'complete': 'complete', 
            'average': 'average',
            'ward': 'ward'
        }
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage=linkage_methods[method]
        )
        cluster_labels = clusterer.fit_predict(X_scaled)
    
    return cluster_labels

# VIF calculation
def calculate_vif(X):
    """Calculate VIF for multicollinearity check"""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Create dendrogram
def create_dendrogram(X_scaled, method):
    """Create dendrogram plot"""
    linkage_matrix = linkage(X_scaled, method=method)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linkage_matrix, ax=ax, leaf_rotation=90)
    ax.set_title(f'Dendrogram - {method.title()} Linkage')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    
    return fig

# Create Indonesia map
def create_indonesia_map(df_results, focus_province=None):
    """Create choropleth map of Indonesia using Folium"""
    center_lat, center_lon = -2.5, 118.0
    zoom = 5

    # Selalu load geojson_data di awal
    geojson_url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-province-simple.json"
    geojson_data = requests.get(geojson_url).json()

    # Jika ada pencarian provinsi, update center dan zoom
    if focus_province:
        for feature in geojson_data['features']:
            prov_name = feature['properties']['Propinsi']
            if focus_province.lower() in prov_name.lower():
                geom = feature['geometry']
                if geom['type'] == 'Polygon':
                    coords = np.array(geom['coordinates'][0])
                elif geom['type'] == 'MultiPolygon':
                    coords = np.array(geom['coordinates'][0][0])
                else:
                    continue
                center_lon, center_lat = coords.mean(axis=0)
                zoom = 6
                break

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles='OpenStreetMap')

    # Mapping nama provinsi baru ke nama di geojson lama
    provinsi_mapping = {
        "Aceh": "DI. ACEH",
        "Sumatera Utara": "SUMATERA UTARA",
        "Sumatera Barat": "SUMATERA BARAT",
        "Riau": "RIAU",
        "Jambi": "JAMBI",
        "Sumatera Selatan": "SUMATERA SELATAN",
        "Bengkulu": "BENGKULU",
        "Lampung": "LAMPUNG",
        "Kepulauan Bangka Belitung": "BANGKA BELITUNG",
        "Kepulauan Riau": "KEPULAUAN RIAU",
        "DKI Jakarta": "DKI JAKARTA",
        "Jawa Barat": "JAWA BARAT",
        "Jawa Tengah": "JAWA TENGAH",
        "DI Yogyakarta": "DAERAH ISTIMEWA YOGYAKARTA",
        "Jawa Timur": "JAWA TIMUR",
        "Banten": "PROBANTEN",
        "Bali": "BALI",
        "Nusa Tenggara Barat": "NUSATENGGARA BARAT",
        "Nusa Tenggara Timur": "NUSA TENGGARA TIMUR",
        "Kalimantan Barat": "KALIMANTAN BARAT",
        "Kalimantan Tengah": "KALIMANTAN TENGAH",
        "Kalimantan Selatan": "KALIMANTAN SELATAN",
        "Kalimantan Timur": "KALIMANTAN TIMUR",
        "Kalimantan Utara": None,  # Tidak ada di geojson lama
        "Sulawesi Utara": "SULAWESI UTARA",
        "Sulawesi Tengah": "SULAWESI TENGAH",
        "Sulawesi Selatan": "SULAWESI SELATAN",
        "Sulawesi Tenggara": "SULAWESI TENGGARA",
        "Gorontalo": "GORONTALO",
        "Sulawesi Barat": None,  # Tidak ada di geojson lama
        "Maluku": "MALUKU",
        "Maluku Utara": "MALUKU UTARA",
        "Papua Barat": "IRIAN JAYA BARAT",
        "Papua Barat Daya": "IRIAN JAYA BARAT",
        "Papua": "IRIAN JAYA TENGAH",
        "Papua Selatan": "IRIAN JAYA TIMUR",
        "Papua Tengah": "IRIAN JAYA TENGAH",
        "Papua Pegunungan": "IRIAN JAYA TENGAH",
    }

    # Terapkan mapping, drop provinsi yang tidak ada di geojson
    df_map = df_results.copy()
    df_map['Provinsi_Geo'] = df_map['Provinsi'].map(provinsi_mapping)
    df_map = df_map[df_map['Provinsi_Geo'].notnull()]

    # Choropleth coloring by cluster
    choropleth = folium.Choropleth(
        geo_data=geojson_data,
        name="Choropleth",
        data=df_map,
        columns=["Provinsi_Geo", "Cluster"],
        key_on="feature.properties.Propinsi",
        fill_color="Set1",
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_color="gray",
        highlight=True
    )
    choropleth.add_to(m)

    # Hapus legend/colorbar bawaan folium (termasuk kotak legenda dan colorbar di atas peta)
    for key in list(m._children):
        if "legend" in key.lower():
            del m._children[key]

    # Tambahkan popup info per provinsi (menggunakan centroid geojson)
    for feature in geojson_data['features']:
        prov_name = feature['properties']['Propinsi']
        prov_row = df_map[df_map['Provinsi_Geo'] == prov_name]
        if not prov_row.empty:
            cluster = int(prov_row['Cluster'].values[0])
            stunting = prov_row['Prevalensi Stunting'].values[0]
            popup_text = f"<b>{prov_name}</b><br><b>Cluster:</b> {cluster}<br><b>Prevalensi Stunting:</b> {stunting:.1f}%"
            # Ambil centroid polygon
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                coords = np.array(geom['coordinates'][0])
            elif geom['type'] == 'MultiPolygon':
                coords = np.array(geom['coordinates'][0][0])
            else:
                continue
            lon, lat = coords.mean(axis=0)
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)

    return m

# Calculate cluster validation metrics
def calculate_cluster_validation(X_scaled, cluster_labels):
    """Calculate within-cluster and between-cluster variations"""
    n_clusters = len(np.unique(cluster_labels))
    
    # Calculate centroids
    centroids = []
    for i in range(n_clusters):
        cluster_points = X_scaled[cluster_labels == i]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    
    # Within-cluster variation (Sw)
    sw = 0
    for i in range(n_clusters):
        cluster_points = X_scaled[cluster_labels == i]
        centroid = centroids[i]
        for point in cluster_points:
            sw += np.sum((point - centroid) ** 2)
    
    # Between-cluster variation (Sb)
    overall_centroid = np.mean(X_scaled, axis=0)
    sb = 0
    for i in range(n_clusters):
        cluster_size = np.sum(cluster_labels == i)
        centroid = centroids[i]
        sb += cluster_size * np.sum((centroid - overall_centroid) ** 2)
    
    ratio = sw / sb if sb != 0 else float('inf')
    
    return sw, sb, ratio

# Download template function
def get_template_download():
    """Create template CSV for download"""
    template_data = {
        'Provinsi': ['Aceh', 'Sumatera Utara', 'Sumatera Barat', 'Riau', 'Jambi', 'Sumatera Selatan', 
                        'Bengkulu', 'Lampung', 'Kepulauan Bangka Belitung', 'Kepulauan Riau', 'DKI Jakarta', 
                        'Jawa Barat', 'Jawa Tengah', 'DI Yogyakarta', 'Jawa Timur', 'Banten', 'Bali', 
                        'Nusa Tenggara Barat', 'Nusa Tenggara Timur', 'Kalimantan Barat', 'Kalimantan Tengah', 
                        'Kalimantan Selatan', 'Kalimantan Timur', 'Kalimantan Utara', 'Sulawesi Utara', 
                        'Sulawesi Tengah', 'Sulawesi Selatan', 'Sulawesi Tenggara', 'Gorontalo', 'Sulawesi Barat', 
                        'Maluku', 'Maluku Utara', 'Papua Barat', 'Papua Barat Daya', 'Papua', 'Papua Selatan', 
                        'Papua Tengah', 'Papua Pegunungan'],
        'Prevalensi Stunting': [0.0] * 38,
        'Persentase Penduduk 15 Tahun Tidak Tamat SD': [0.0] * 38,
        'Persentase Penduduk Miskin': [0.0] * 38,
        'Ibu Hamil KEK': [0.0] * 38,
        'Akses Sanitasi Layak': [0.0] * 38,
        'Akses Air Minum Layak': [0.0] * 38,
        'Akses Kesehatan Dasar': [0.0] * 38
    }
    template_df = pd.DataFrame(template_data)
    return template_df.to_csv(index=False)

# Main app
def main():
    st.markdown('<h1 class="main-header"> Dashboard Clustering Stunting Indonesia</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        
        # File upload section
        st.subheader("Data Input")
        uploaded_file = st.file_uploader(
            "Upload file CSV (opsional)",
            type=['csv'],
            help="Upload file CSV dengan struktur yang sama dengan template"
        )
        
        # Download template button
        template_csv = get_template_download()
        st.download_button(
            label="📥Download Template CSV",
            data=template_csv,
            file_name="template_stunting.csv",
            mime="text/csv"
        )
        
        # Load data
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=';')
                st.success("✅ Data berhasil diupload!")
                st.session_state.data = df
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                df = load_default_data()
                st.session_state.data = df
        else:
            df = load_default_data()
            st.session_state.data = df
            st.info("Menggunakan data default 2023")
        
        # Feature selection
        st.subheader("Pemilihan Variabel")
        available_features = [col for col in df.columns if col != 'Provinsi']
        selected_features = st.multiselect(
            "Pilih variabel untuk clustering:",
            available_features,
            default=available_features,
            help="Pilih minimal 2 variabel"
        )
        
        if len(selected_features) < 2:
            st.warning("Pilih minimal 2 variabel!")
            return

         # Simpan fitur numerik yang dipilih untuk clustering
        st.session_state.selected_features = ['Provinsi'] + selected_features
        # Simpan daftar lengkap jika perlu nanti
        st.session_state.selected_features_with_provinsi = ['Provinsi'] + selected_features

        # Clustering parameters
        st.subheader("Parameter Clustering")
        
        clustering_method = st.selectbox(
            "Metode Clustering:",
            ['single', 'complete', 'average', 'ward', 'kmeans'],
            index=1,
            help="Pilih metode clustering yang diinginkan"
        )
        st.session_state.clustering_method = clustering_method
        
        n_clusters = st.slider(
            "Jumlah Cluster:",
            min_value=3,
            max_value=7,
            value=5,
            help="Tentukan jumlah cluster yang diinginkan"
        )
        st.session_state.n_clusters = n_clusters
        
        # Process clustering
        if st.button("Jalankan Clustering", type="primary"):
            with st.spinner("Memproses clustering..."):
                # Preprocess data
                df_clean, X_scaled, scaler, removed_rows = preprocess_data(
                    df, st.session_state.selected_features
                )
                
                # Perform clustering
                cluster_labels = perform_clustering(
                    X_scaled.values, clustering_method, n_clusters
                )
                
                # Create results dataframe
                df_results = df_clean.copy()
                df_results['Cluster'] = cluster_labels + 1
                
                # Store results
                st.session_state.processed_data = {
                    'df_clean': df_clean,
                    'X_scaled': X_scaled,
                    'scaler': scaler,
                    'removed_rows': removed_rows
                }
                st.session_state.cluster_results = df_results

                # Salin data awal sebelum preprocessing
                original_data = st.session_state.data.copy()

                # Pisahkan data yang memiliki missing value
                data_with_missing = original_data[original_data.isnull().any(axis=1)].copy()
                data_cleaned = original_data.dropna().copy()

                # Gunakan data_cleaned untuk clustering (kode clustering kamu sebelumnya)

                # Asumsikan hasil clustering disimpan di:
                # st.session_state.cluster_results -> Data dengan hasil clustering
                # Kolom 'Cluster' harus sudah ada di situ

                # Ambil hasil clustering dan centroid
                clustered_data = st.session_state.cluster_results.copy()
                feature_cols = [col for col in st.session_state.selected_features if col != 'Provinsi' and col in clustered_data.columns]
                cluster_centroids = clustered_data.groupby('Cluster')[feature_cols].mean()

                # Imputasi kembali data_with_missing
                imputasi_rows = []
                imputasi_info = []

                for idx, row in data_with_missing.iterrows():
                    prov_name = row['Provinsi']
                    missing_cols = row.index[row.isnull()].tolist()
                    available_feats = [col for col in feature_cols if col not in missing_cols]

                    if not available_feats:
                        imputasi_info.append(f"{prov_name} tidak memiliki fitur tersedia untuk imputasi.")
                        continue

                    # Normalisasi fitur yang tersedia
                    scaler_temp = StandardScaler().fit(clustered_data[available_feats])
                    row_scaled = scaler_temp.transform(row[available_feats].values.reshape(1, -1))
                    centroid_scaled = scaler_temp.transform(cluster_centroids[available_feats])

                    # Hitung jarak ke setiap centroid
                    distances = np.linalg.norm(centroid_scaled - row_scaled, axis=1)
                    closest_cluster_idx = np.argmin(distances)
                    closest_cluster_label = cluster_centroids.index[closest_cluster_idx]

                    # Buat salinan row dan imputasi nilai hilang
                    imputasi_dict = row.to_dict()
                    imputasi_dict['Cluster'] = closest_cluster_label
                    for col in missing_cols:
                        if col in cluster_centroids.columns:
                            imputasi_dict[col] = cluster_centroids.loc[closest_cluster_label, col]
                        else:
                            imputasi_dict[col] = np.nan  # atau nilai default lain

                    imputasi_rows.append(imputasi_dict)
                    imputasi_info.append(f"{prov_name} diimputasi menggunakan centroid cluster {closest_cluster_label}")

                # Gabungkan kembali data yang sudah diimputasi
                imputasi_df = pd.DataFrame(imputasi_rows)
                hasil_clustering = pd.concat([clustered_data, imputasi_df], ignore_index=True)

                # Simpan ke session_state
                st.session_state.cluster_results = hasil_clustering
                st.session_state.imputasi_info = imputasi_info

                st.success("Clustering berhasil!")
         
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Main", "📊 Analisis Cluster", "✅ Validasi Cluster", "📋 Hasil Clustering"])
    
    with tab1:
        if st.session_state.cluster_results is not None:
            df_results = st.session_state.cluster_results
            
            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_stunting = df_results['Prevalensi Stunting'].mean()
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-value">{avg_stunting:.1f}%</div>
                    <div class="kpi-label">Rata-rata Nasional Prevalensi Stunting</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                n_clusters_formed = df_results['Cluster'].nunique()
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-value">{n_clusters_formed}</div>
                    <div class="kpi-label">Jumlah Cluster Terbentuk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                highest_stunting = df_results.loc[df_results['Prevalensi Stunting'].idxmax()]
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-value" style="font-size: 1.2rem;">{highest_stunting['Provinsi']}</div>
                    <div class="kpi-label">Stunting Tertinggi ({highest_stunting['Prevalensi Stunting']:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                lowest_stunting = df_results.loc[df_results['Prevalensi Stunting'].idxmin()]
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-value" style="font-size: 1.2rem;">{lowest_stunting['Provinsi']}</div>
                    <div class="kpi-label">Stunting Terendah ({lowest_stunting['Prevalensi Stunting']:.1f}%)</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Cari Provinsi
            st.markdown("🔍 Cari Provinsi")
            province_search = st.text_input(
                "Nama Provinsi:",
                placeholder="Ketik nama provinsi...",
                key="province_search_main"
            )

            # Handle province search
            province_data = None
            focus_province = None
            if province_search:
                matching_provinces = df_results[
                    df_results['Provinsi'].str.contains(province_search, case=False, na=False)
                ]
                if not matching_provinces.empty:
                    province_data = matching_provinces.iloc[0]
                    focus_province = province_data['Provinsi']
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>Provinsi Ditemukan: {province_data['Provinsi']}</strong><br>
                        <strong>Cluster:</strong> {province_data['Cluster']}<br>
                        <strong>Prevalensi Stunting:</strong> {province_data['Prevalensi Stunting']:.1f}%
                    </div>
                    """, unsafe_allow_html=True)

            # SUBHEADER PETA
            st.subheader("Peta Cluster Wilayah")

            # Create and display map
            map_obj = create_indonesia_map(df_results, focus_province=focus_province)
            st_folium(map_obj, height=500, use_container_width=True)
            
            # Charts (atas-bawah, bukan kolom)
            st.subheader("Jumlah Provinsi per Cluster")
            cluster_counts = df_results['Cluster'].value_counts().sort_index()
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="Distribusi Provinsi per Cluster",
                labels={'x': 'Cluster', 'y': 'Jumlah Provinsi'},
                text=cluster_counts.values
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(height=400, margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

            # Rata-rata Indikator per Cluster
            st.subheader("Rata-rata Indikator per Cluster")

            # Filter Cluster
            cluster_options = sorted(df_results['Cluster'].unique())
            cluster_filter = st.selectbox(
                "Pilih Cluster untuk ditampilkan:",
                options=["Semua"] + [f"Cluster {i}" for i in cluster_options],
                index=0
            )

            # Filter data untuk grafik rata-rata indikator
            if cluster_filter == "Semua":
                cluster_means = df_results.groupby('Cluster')[selected_features].mean()
                clusters_to_plot = cluster_means.index
            else:
                selected_cluster = int(cluster_filter.split()[-1])
                cluster_means = df_results[df_results['Cluster'] == selected_cluster].groupby('Cluster')[selected_features].mean()
                clusters_to_plot = [selected_cluster]

            fig = go.Figure()
            for feature in selected_features:
                fig.add_trace(go.Bar(
                    name=feature,
                    x=clusters_to_plot,
                    y=cluster_means[feature],
                    text=cluster_means[feature].round(1),
                    textposition='auto'
                ))
            fig.update_layout(
                title="Rata-rata Indikator per Cluster",
                xaxis_title="Cluster",
                yaxis_title="Nilai",
                barmode='group',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Boxplot
            st.subheader("Boxplot per Cluster")

            n_features = len(selected_features)
            cols = 2  # 2 kolom
            rows = (n_features + cols - 1) // cols

            fig = make_subplots(
                rows=rows, cols=cols,
                subplot_titles=selected_features,
                vertical_spacing=0.15,
                horizontal_spacing=0.08
            )

            cluster_list = sorted(df_results['Cluster'].unique())
            cluster_colors = px.colors.qualitative.Set1
            color_map = {cluster: cluster_colors[i % len(cluster_colors)] for i, cluster in enumerate(cluster_list)}

            for i, feature in enumerate(selected_features):
                row = i // cols + 1
                col = i % cols + 1

                for j, cluster in enumerate(cluster_list):
                    cluster_data = df_results[df_results['Cluster'] == cluster][feature]
                    fig.add_trace(
                        go.Box(
                            y=cluster_data,
                            name=f"Cluster {cluster}",
                            marker_color=color_map[cluster],
                            boxpoints='outliers',
                            showlegend=(i == 0),
                            legendgroup=f"Cluster {cluster}",
                            line_width=2,
                            width=0.5  # Perbesar box
                        ),
                        row=row, col=col
                    )

                # Sumbu x hanya angka cluster
                fig.update_xaxes(
                    tickmode='array',
                    tickvals=cluster_list,
                    ticktext=[str(c) for c in cluster_list],
                    title_text="Cluster",
                    row=row, col=col
                )

            fig.update_layout(
                height=350*rows,
                showlegend=True,
                boxmode='group',
                title_text=None
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Penentuan Prioritas Cluster Otomatis
            cluster_means_all = df_results.groupby('Cluster')[selected_features].mean()

            # Variabel: semakin tinggi semakin prioritas
            high_is_bad = [
                'Prevalensi Stunting',
                'Persentase Penduduk 15 Tahun Tidak Tamat SD',
                'Persentase Penduduk Miskin',
                'Ibu Hamil KEK'
            ]
            # Variabel: semakin rendah semakin prioritas
            low_is_bad = [
                'Akses Sanitasi Layak',
                'Akses Air Minum Layak',
                'Akses Kesehatan Dasar'
            ]

            # Skor prioritas per cluster
            priority_scores = pd.DataFrame(index=cluster_means_all.index)
            for var in high_is_bad:
                if var in cluster_means_all.columns:
                    # Ranking descending: nilai besar = skor besar
                    priority_scores[var] = cluster_means_all[var].rank(ascending=False, method='min')
            for var in low_is_bad:
                if var in cluster_means_all.columns:
                    # Ranking ascending: nilai kecil = skor besar
                    priority_scores[var] = cluster_means_all[var].rank(ascending=True, method='min')

            priority_scores['Total'] = priority_scores.sum(axis=1)
            priority_scores['Prioritas'] = priority_scores['Total'].rank(ascending=True, method='min').astype(int)
            priority_scores = priority_scores.sort_values('Prioritas')

            # Mapping cluster ke prioritas dan total skor
            cluster_to_priority = priority_scores['Prioritas'].to_dict()
            cluster_to_total_score = priority_scores['Total'].to_dict()

            # Cluster interpretation
            st.subheader("Interpretasi Cluster")
            st.markdown("""
            <div class="warning-box">
            <b>Catatan:</b><br>
            Penentuan <b>Prioritas Intervensi</b> pada dashboard ini dihitung dari <b>akumulasi ranking rata-rata seluruh indikator utama</b> (stunting, pendidikan, kemiskinan, ibu KEK, akses sanitasi, air minum, dan kesehatan dasar) dengan bobot sama.<br>
            <b>Prioritas #1</b> berarti cluster dengan masalah terakumulasi paling berat dan membutuhkan penanganan segera.<br>
            Semakin besar masalah pada indikator-indikator tersebut, semakin tinggi prioritas intervensi yang diberikan.
            </div>
            """, unsafe_allow_html=True)
            # Rata-rata nasional untuk threshold
            national_means = df_results[selected_features].mean()

            # Mapping indikator ke label masalah & rekomendasi intervensi
            indikator_labels = {
                'Prevalensi Stunting': ('prevalensi stunting tinggi', 'perbaikan gizi'),
                'Persentase Penduduk 15 Tahun Tidak Tamat SD': ('tingkat pendidikan rendah', 'peningkatan pendidikan'),
                'Persentase Penduduk Miskin': ('kemiskinan tinggi', 'pengentasan kemiskinan'),
                'Ibu Hamil KEK': ('ibu KEK tinggi', 'intervensi gizi ibu hamil'),
                'Akses Sanitasi Layak': ('akses sanitasi rendah', 'peningkatan sanitasi'),
                'Akses Air Minum Layak': ('akses air minum rendah', 'peningkatan air minum layak'),
                'Akses Kesehatan Dasar': ('akses kesehatan dasar rendah', 'peningkatan akses kesehatan dasar')
            }

            for cluster in sorted(df_results['Cluster'].unique()):
                cluster_data = df_results[df_results['Cluster'] == cluster]
                cluster_means = cluster_data[selected_features].mean()
                prioritas = cluster_to_priority.get(cluster, "-")
                total_score = cluster_to_total_score.get(cluster, None)
                q1 = priority_scores['Total'].quantile(0.33)
                q2 = priority_scores['Total'].quantile(0.66)
                if total_score <= q1:
                    risk_level = "Risiko Tinggi"
                elif total_score <= q2:
                    risk_level = "Risiko Sedang"
                else:
                    risk_level = "Risiko Rendah"

                # --- Cari masalah berat di cluster ini ---
                masalah_berat = []
                rekomendasi = []
                for var in selected_features:
                    if var not in indikator_labels:
                        continue
                    label_masalah, label_rekom = indikator_labels[var]
                    # Untuk high_is_bad: jika cluster > nasional
                    if var in high_is_bad and cluster_means[var] > national_means[var]:
                        masalah_berat.append(f"{label_masalah} ({cluster_means[var]:.1f}%)")
                        rekomendasi.append(label_rekom)
                    # Untuk low_is_bad: jika cluster < nasional
                    if var in low_is_bad and cluster_means[var] < national_means[var]:
                        masalah_berat.append(f"{label_masalah} ({cluster_means[var]:.1f}%)")
                        rekomendasi.append(label_rekom)

                masalah_str = ", ".join(masalah_berat) if masalah_berat else "tidak ada masalah berat menonjol"
                rekom_str = ", ".join(sorted(set(rekomendasi)))

                interpretation = (
                    f"Cluster {cluster} mencerminkan wilayah dengan {risk_level.lower()} stunting. "
                    f"Karakteristik utama meliputi {masalah_str}. "
                    f"Wilayah ini membutuhkan intervensi prioritas dalam program {rekom_str}."
                )

                with st.expander(f"Cluster {cluster} - {risk_level} ({len(cluster_data)} provinsi) | Prioritas #{prioritas}"):
                    st.markdown(f"""
                    <div class="cluster-interpretation">
                        <strong>Prioritas Intervensi: #{prioritas}</strong><br>
                        {interpretation}
                        <br><br>
                        <strong>Provinsi dalam cluster ini:</strong><br>
                        {', '.join(cluster_data['Provinsi'].tolist())}
                    </div>
                    """, unsafe_allow_html=True)
            
        else:
            st.info("👆 Silakan atur parameter dan jalankan clustering terlebih dahulu.")
    
    with tab2:
      
        if st.session_state.processed_data is not None:
            processed_data = st.session_state.processed_data
            df_clean = processed_data['df_clean']
            X_scaled = processed_data['X_scaled']
            
            # Preprocessing info
            st.subheader("Preprocessing Data")
            st.markdown("""
            <div class="success-box">
                Data yang digunakan telah melalui preprocessing berupa pengecekan duplikat, missing value, dan standarisasi. 
                Apabila ditemukan missing value, provinsi tersebut akan dihapus dan akan dilakukan imputasi berdasarkan cluster terdekat.
            </div>
            """, unsafe_allow_html=True)
            
            if processed_data['removed_rows'] > 0:
                st.warning(f"⚠️ {processed_data['removed_rows']} baris data dihapus karena missing value.")
            
            # Multicollinearity check
            st.subheader("Pengecekan Multikolinearitas")
            vif_data = calculate_vif(X_scaled)
            
            # Color code VIF values
            def color_vif(val):
                if val > 10:
                    return 'background-color: #ffcccc'  # Red for high VIF
                elif val > 5:
                    return 'background-color: #ffffcc'  # Yellow for moderate VIF
                else:
                    return 'background-color: #ccffcc'  # Green for low VIF
            
            styled_vif = vif_data.style.applymap(color_vif, subset=['VIF'])
            st.dataframe(styled_vif, use_container_width=True)
            
            st.markdown("""
            **Interpretasi VIF:**
            - VIF < 10: Tidak ada multikolinearitas
            - VIF > 10: Multikolinearitas tinggi
            """)
            
            # Dendrogram
            st.subheader("Dendrogram")
            if st.session_state.clustering_method != 'kmeans':
                fig_dendro = create_dendrogram(X_scaled.values, st.session_state.clustering_method)
                st.pyplot(fig_dendro)
            else:
                st.info("Dendrogram tidak tersedia untuk metode K-Means.")
            
            # Silhouette Analysis (Line Chart)
            st.subheader("Analisis Silhouette")
            if st.session_state.processed_data is not None:
                X_scaled = st.session_state.processed_data['X_scaled'].values
                method = st.session_state.clustering_method

                n_clusters_range = range(2, 8)
                silhouette_scores = []

                for n in n_clusters_range:
                    if method == 'kmeans':
                        clusterer = KMeans(n_clusters=n, random_state=42)
                    else:
                        linkage_methods = {
                            'single': 'single',
                            'complete': 'complete',
                            'average': 'average',
                            'ward': 'ward'
                        }
                        clusterer = AgglomerativeClustering(
                            n_clusters=n,
                            linkage=linkage_methods[method]
                        )
                    labels = clusterer.fit_predict(X_scaled)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X_scaled, labels)
                    else:
                        score = np.nan
                    silhouette_scores.append(score)

                # Plot line chart dengan nilai score di setiap titik
                fig = px.line(
                    x=list(n_clusters_range),
                    y=silhouette_scores,
                    markers=True,
                    text=[f"{s:.3f}" if not np.isnan(s) else "" for s in silhouette_scores],
                    labels={'x': 'Jumlah Cluster', 'y': 'Silhouette Score'},
                    title=f'Silhouette Score vs Jumlah Cluster ({method.title()})'
                )
                fig.update_traces(textposition="top center")
                st.plotly_chart(fig, use_container_width=True)

                # Info interpretasi berdasarkan selisih terbesar
                if np.any(~np.isnan(silhouette_scores)):
                    # Hitung selisih antar score (pakai selisih absolut)
                    diffs = np.diff(silhouette_scores)
                    if len(diffs) > 0:
                        max_diff_idx = np.nanargmax(np.abs(diffs))
                        best_n = max_diff_idx + 3  # +2 karena range(2,8), +1 lagi karena diff offset
                        best_score = silhouette_scores[max_diff_idx + 1]
                        st.markdown(f"""
                        **Silhouette Score terbaik (berdasarkan lonjakan terbesar)** pada jumlah cluster {best_n}
                        """)
                    else:
                        # fallback ke score maksimum jika hanya 1 cluster
                        best_score = np.nanmax(silhouette_scores)
                        best_n = np.nanargmax(silhouette_scores) + 2
                        st.markdown(f"""
                        **Silhouette Score terbaik:** {best_score:.4f} pada jumlah cluster {best_n}
                        """)

    with tab3:
        if st.session_state.cluster_results is not None and st.session_state.processed_data is not None:
            df_clean = st.session_state.processed_data['df_clean']
            X_scaled = st.session_state.processed_data['X_scaled']

            # Ambil hasil cluster dari cluster_results, filter hanya provinsi yang ada di df_clean
            df_results = st.session_state.cluster_results
            cluster_labels = df_results.set_index('Provinsi').loc[df_clean['Provinsi'], 'Cluster'].values - 1  # -1 untuk indeks 0 di Python
            
            if len(np.unique(cluster_labels)) >= 2:      
                # Cluster validation metrics
                st.subheader("Metrik Validasi Cluster")
                sw, sb, ratio = calculate_cluster_validation(X_scaled.values, cluster_labels)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-value">{sw:.2f}</div>
                        <div class="kpi-label">Within-cluster Variation (Sw)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-value">{sb:.2f}</div>
                        <div class="kpi-label">Between-cluster Variation (Sb)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-value">{ratio:.3f}</div>
                        <div class="kpi-label">Rasio Sw/Sb</div>
                    </div>
                    """, unsafe_allow_html=True)

                with st.expander("🔽 Interpretasi Metrik Validasi"):
                    st.markdown(
                        """
                <div class="cluster-interpretation">
                <strong>Rasio Sw/Sb lebih kecil menunjukkan kualitas cluster yang lebih baik.</strong><br><br>
                - <strong>Within-cluster variation (Sw):</strong> Mengukur variasi dalam cluster. Nilai lebih kecil menunjukkan anggota cluster lebih homogen.<br>
                - <strong>Between-cluster variation (Sb):</strong> Mengukur variasi antar cluster. Nilai lebih besar menunjukkan cluster lebih terpisah.<br>
                - <strong>Ratio Sw/Sb:</strong> Rasio yang lebih kecil menunjukkan cluster yang lebih compact dan terpisah dengan baik.
                </div>
                        """, unsafe_allow_html=True
                    )
            else:
                st.info("Tidak cukup cluster untuk validasi (minimal 2 cluster berbeda).")
            
            # Multiscale Bootstrap (berbasis data dan metode aktual)
            st.subheader("Analisis Multiscale Bootstrap")
            
            if st.session_state.clustering_method != 'kmeans':
                X_scaled = st.session_state.processed_data['X_scaled'].values
                method = st.session_state.clustering_method
                n_clusters = st.session_state.n_clusters
            
                def bootstrap_clustering(X, method, n_clusters, n_bootstrap=100):
                    """Simulasi bootstrap berdasarkan stabilitas keanggotaan cluster"""
                    n_samples = X.shape[0]
                    base_linkage = linkage(X, method=method)
                    base_labels = fcluster(base_linkage, n_clusters, criterion='maxclust')
                    stability_counts = defaultdict(int)
            
                    for _ in range(n_bootstrap):
                        idx_bootstrap = np.random.choice(n_samples, size=n_samples, replace=True)
                        X_bootstrap = X[idx_bootstrap]
                        boot_linkage = linkage(X_bootstrap, method=method)
                        boot_labels = fcluster(boot_linkage, n_clusters, criterion='maxclust')
                        for i, original_idx in enumerate(idx_bootstrap):
                            if original_idx < len(base_labels):
                                if base_labels[original_idx] == boot_labels[i]:
                                    stability_counts[original_idx] += 1
            
                    au_values_per_point = np.array([
                        stability_counts[i] / n_bootstrap * 100 for i in range(n_samples)
                    ])
            
                    clusterwise_au = []
                    clusterwise_bp = []
            
                    for c in range(1, n_clusters + 1):
                        members = (base_labels == c)
                        au_vals = au_values_per_point[members]
                        clusterwise_au.append(np.mean(au_vals))
                        clusterwise_bp.append(np.median(au_vals))  # Proxy
            
                    return np.array(clusterwise_au), np.array(clusterwise_bp)
            
                # Jalankan bootstrap
                with st.spinner("⏳ Menjalankan simulasi bootstrap..."):
                    au_vals, bp_vals = bootstrap_clustering(
                        X=X_scaled,
                        method=method,
                        n_clusters=n_clusters,
                        n_bootstrap=100
                    )
            
                # Tampilkan hasil
                bootstrap_df = pd.DataFrame({
                    'Cluster': [f'Cluster {i+1}' for i in range(len(au_vals))],
                    'AU (%)': au_vals,
                    'BP (%)': bp_vals
                })
            
                avg_au = np.mean(au_vals)
            
                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-value">{avg_au:.1f}%</div>
                    <div class="kpi-label">Rata-rata AU (Approximately Unbiased)</div>
                </div>
                """, unsafe_allow_html=True)
            
                st.dataframe(bootstrap_df, use_container_width=True)
            
                with st.expander("🔽 Interpretasi Multiscale Bootstrap"):
                    st.markdown(
                        """
                        <div class="cluster-interpretation">
                        <strong>Interpretasi AU dan BP:</strong><br><br>
                        - <strong>AU (Approximately Unbiased):</strong> Mengukur stabilitas cluster berdasarkan hasil bootstrap. Jika AU > 95%, cluster dianggap sangat stabil dan tidak terbentuk secara acak.<br>
                        - <strong>BP (Bootstrap Probability):</strong> Probabilitas kemunculan cluster dalam sampling ulang. BP > 90% menunjukkan cluster cukup kuat.<br><br>
                        Nilai AU & BP yang tinggi menunjukkan bahwa struktur pengelompokan cukup konsisten terhadap variasi data. Hasil ini memperkuat kepercayaan terhadap segmentasi yang dihasilkan dan bisa digunakan untuk rekomendasi kebijakan berbasis data.
                        </div>
                        """, unsafe_allow_html=True
                    )
            else:
                st.info("Bootstrap hanya tersedia untuk metode hierarkis (single, complete, average, ward).")
            
            # Missing value imputation simulation
            st.subheader("Imputasi Missing Value")
            
            # Check for missing values in original data
            original_data = st.session_state.data
            missing_info = original_data.isnull().sum()
            
            if missing_info.sum() > 0:
                st.markdown("**Tabel Provinsi dengan Missing Value:**")
                missing_provinces = original_data[original_data.isnull().any(axis=1)]
                show_cols = ['Provinsi'] + [col for col in selected_features if col in original_data.columns]
                st.dataframe(missing_provinces[show_cols], use_container_width=True)

                # Tambahkan info provinsi yang diimputasi dan centroid cluster terdekat
                imputasi_info = []
                df_results = st.session_state.cluster_results
                feature_cols = [col for col in selected_features if col != 'Provinsi']
                cluster_centroids = df_results.groupby('Cluster')[feature_cols].mean()

                for idx, row in missing_provinces.iterrows():
                    prov_name = row['Provinsi']
                    available_features = [col for col in feature_cols if not pd.isna(row[col])]
                    if not available_features:
                        continue  # skip jika semua fitur NaN
                    prov_values = row[available_features].values.astype(float)
                    dists = cluster_centroids[available_features].apply(
                        lambda centroid: np.linalg.norm(prov_values - centroid.values), axis=1
                    )
                    closest_cluster = dists.idxmin()
                    imputasi_info.append(f"- {prov_name}: centroid cluster {closest_cluster}")

                st.markdown(
                    f"""
                    <div class="success-box">
                    <strong>Missing value telah diimputasi menggunakan centroid cluster terdekat:</strong><br>
                    {'<br>'.join(imputasi_info)}
                    </div>
                    """, unsafe_allow_html=True
                )
            else:
                st.success("✅ Tidak ditemukan missing value dalam data.")
        else:
            st.info("👆 Jalankan clustering terlebih dahulu untuk melihat validasi.")
    
    with tab4:
        
        if st.session_state.cluster_results is not None:
            df_results = st.session_state.cluster_results
                        
            # Display results table
            st.subheader("Tabel Hasil Clustering")
            
            # Add styling to highlight clusters
            def highlight_cluster(row):
                colors = ['background-color: #e1f5fe', 'background-color: #f3e5f5', 
                        'background-color: #e8f5e8', 'background-color: #fff3e0',
                        'background-color: #fce4ec', 'background-color: #f1f8e9',
                        'background-color: #e0f2f1']
                cluster = int(row['Cluster']) - 1
                color = colors[cluster % len(colors)]
                return [color] * len(row)

            styled_results = df_results.style.apply(highlight_cluster, axis=1)
            st.dataframe(styled_results, use_container_width=True)
            
            # Download button
            st.subheader("Download Hasil")
            
            # Prepare download data
            download_data = df_results.copy()
            csv_data = download_data.to_csv(index=False)
            
            st.download_button(
                label="📥Download Hasil Clustering (CSV)",
                data=csv_data,
                file_name=f"hasil_clustering_stunting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download hasil clustering dalam format CSV"
            )
            
        else:
            st.info("👆 Jalankan clustering terlebih dahulu untuk melihat hasil.")
    
    # Sidebar information
    with st.sidebar:
        st.markdown("---")
        st.subheader("ℹ️ Informasi Aplikasi")
        st.write("**Versi:** 1.0")
        st.write("**Dibuat:** 2024")
        st.write("**Tujuan:** Analisis clustering stunting provinsi Indonesia")
        
        with st.expander("📖 Panduan Penggunaan"):
            st.write("""
            **Langkah-langkah:**
            1. Upload data atau gunakan data default 2023
            2. Pilih variabel untuk clustering
            3. Tentukan metode dan jumlah cluster
            4. Eksplorasi hasil di berbagai tab
            5. Download hasil analisis
            
            **Tips:**
            - Gunakan filter cluster untuk fokus analisis
            - Manfaatkan pencarian provinsi untuk detail
            - Perhatikan interpretasi cluster untuk insight
            """)

if __name__ == "__main__":
    main()
