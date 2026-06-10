import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Konfiguracja strony
st.set_page_config(page_title="Analizator Krzywych Pro AI", layout="wide")

# =================================================================
# CACHOWANIE I PRZETWARZANIE DANYCH
# =================================================================
@st.cache_data
def load_and_preprocess(file):
    df_raw = pd.read_excel(file, header=None)
    df = inteligentne_pobranie_tabeli(df_raw)
    x = df.iloc[:, 0]
    krzywe = df.iloc[:, 1:]
    dane = StandardScaler().fit_transform(krzywe.T)
    return df, x, krzywe, dane

def inteligentne_pobranie_tabeli(df_raw):
    df_raw = df_raw.dropna(how='all', axis=0).dropna(how='all', axis=1).reset_index(drop=True)
    indeks_startu = 0
    for idx, row in df_raw.iterrows():
        if row.notna().sum() > 1 and idx + 1 < len(df_raw):
            if pd.to_numeric(df_raw.iloc[idx + 1], errors='coerce').notna().sum() > 1:
                indeks_startu = idx; break
    df_czysty = df_raw.iloc[indeks_startu + 1:].copy()
    df_czysty.columns = df_raw.iloc[indeks_startu]
    return df_czysty.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1).reset_index(drop=True)

# =================================================================
# SILNIK KLASTROWANIA
# =================================================================
def uruchom_silnik_klastrowania(nazwa_metody, dane, k_grup, df_sygnaly_raw=None):
    try:
        if nazwa_metody == "K-means": return KMeans(n_clusters=k_grup, random_state=42, n_init=5).fit_predict(dane) + 1
        elif "Filtrowanie szumów" in nazwa_metody:
            dane_ward = StandardScaler().fit_transform(df_sygnaly_raw.rolling(window=5, center=True, min_periods=1).mean().T)
            return fcluster(linkage(dane_ward, method='ward'), t=k_grup, criterion='maxclust')
        elif "UMAP + HDBSCAN" in nazwa_metody:
            import umap
            przestrzen_2d = umap.UMAP(n_neighbors=15, random_state=42).fit_transform(dane)
            raw = HDBSCAN(min_cluster_size=k_grup).fit_predict(przestrzen_2d)
            return np.array([n + 1 if n >= 0 else 0 for n in raw])
        elif "GMM" in nazwa_metody: return GaussianMixture(n_components=k_grup, random_state=42).fit_predict(dane) + 1
        else: return fcluster(linkage(dane, method='ward'), t=k_grup, criterion='maxclust')
    except: return np.ones(dane.shape[0])

# =================================================================
# UI GŁÓWNE
# =================================================================
st.title("📊 Analizator Krzywych Pro AI")
uploaded_file = st.file_uploader("Wgraj plik Excel", type=["xlsx"])

if uploaded_file:
    df, x, krzywe, dane = load_and_preprocess(uploaded_file)
    nazwy_krzywych = krzywe.columns.tolist()
    
    lista_metod = ["Hierarchiczna (Warda)", "K-means", "Filtrowanie szumów + Hierarchiczna", "UMAP + HDBSCAN", "GMM"]
    metoda = st.sidebar.selectbox("Algorytm:", lista_metod)
    k = st.sidebar.slider("Liczba grup:", 2, 10, 5)
    
    numery_grup = uruchom_silnik_klastrowania(metoda, dane, k, df_sygnaly_raw=krzywe)
    
    # WYKRES 1: Plotly Interactive
    st.subheader("Wykres 1: Wszystkie krzywe (Plotly)")
    fig = go.Figure()
    for i, col in enumerate(krzywe.columns):
        fig.add_trace(go.Scatter(x=x, y=krzywe.iloc[:, i], name=str(col), mode='lines', opacity=0.5))
    fig.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # WYKRES 2: Profile modelowe + MSE Anomaly Detection
    st.subheader("Wykres 2: Profile Modelowe i Diagnostyka MSE")
    fig_s = go.Figure()
    mse_vals = np.zeros(len(krzywe.columns))
    
    for k_id in np.unique(numery_grup):
        maska = (numery_grup == k_id)
        if any(maska):
            k_dane = krzywe.iloc[:, maska]
            srednia = k_dane.mean(axis=1)
            std = k_dane.std(axis=1)
            fig_s.add_trace(go.Scatter(x=x, y=srednia, name=f"Klaster {k_id}", line=dict(width=3)))
            # MSE dla każdej krzywej w klastrze
            k_dane_scaled = dane[maska]
            mse_vals[maska] = np.mean((k_dane_scaled - k_dane_scaled.mean(axis=0))**2, axis=1)
            
    st.plotly_chart(fig_s, use_container_width=True)

    # TABELA ANOMALII
    df['MSE'] = mse_vals
    st.markdown("##### 🚨 Top 5 Anomalii (Wysokie MSE):")
    st.dataframe(df[['MSE']].sort_values('MSE', ascending=False).head(5), width=400)

    # BRUTE-FORCE AI
    st.write("---")
    st.subheader("🤖 Brute-Force AI (Turniej Silhouette)")
    if st.button("Uruchom Turniej"):
        bf_results = []
        for m in lista_metod:
            for i in range(2, 6):
                p = uruchom_silnik_klastrowania(m, dane, i, df_sygnaly_raw=krzywe)
                score = silhouette_score(dane, p) if len(np.unique(p)) > 1 else 0
                bf_results.append({"Metoda": m, "K": i, "Silhouette": round(score, 4)})
        st.table(pd.DataFrame(bf_results).sort_values("Silhouette", ascending=False))
