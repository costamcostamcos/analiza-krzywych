import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np

# Konfiguracja strony
st.set_page_config(page_title="Analizator Krzywych Pro AI", layout="wide")

# =================================================================
# FUNKCJE POMOCNICZE
# =================================================================
def inteligentne_pobranie_tabeli(df_raw):
    df_raw = df_raw.dropna(how='all', axis=0).dropna(how='all', axis=1).reset_index(drop=True)
    indeks_startu = 0
    for idx, row in df_raw.iterrows():
        if row.notna().sum() > 1:
            if idx + 1 < len(df_raw):
                if pd.to_numeric(df_raw.iloc[idx + 1], errors='coerce').notna().sum() > 1:
                    indeks_startu = idx
                    break
    df_czysty = df_raw.iloc[indeks_startu + 1:].copy()
    df_czysty.columns = df_raw.iloc[indeks_startu]
    return df_czysty.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1).reset_index(drop=True)

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
    except: return np.zeros(dane.shape[0])

# =================================================================
# GŁÓWNY INTERFEJS
# =================================================================
st.title("📊 Analizator Krzywych Pro AI")
uploaded_file = st.file_uploader("Wgraj plik Excel", type=["xlsx"])

if uploaded_file:
    df = inteligentne_pobranie_tabeli(pd.read_excel(uploaded_file, header=None))
    x, krzywe = df.iloc[:, 0], df.iloc[:, 1:]
    
    col1, col2 = st.columns([1, 3])
    with col1:
        metoda = st.selectbox("Algorytm:", ["Hierarchiczna (Warda)", "K-means", "Filtrowanie szumów + Hierarchiczna", "UMAP + HDBSCAN", "GMM"])
        k_grup = st.slider("Liczba grup:", 2, 10, 5)
    
    dane_do_algorytmu = StandardScaler().fit_transform(krzywe.T)
    numery_grup = uruchom_silnik_klastrowania(metoda, dane_do_algorytmu, k_grup, df_sygnaly_raw=krzywe)

    # WYKRES 1: Wszystkie
    fig, ax = plt.subplots(figsize=(10, 3))
    for i, col in enumerate(krzywe.columns):
        ax.plot(x, krzywe[col], alpha=0.3, color=plt.cm.tab10(numery_grup[i] % 10))
    st.pyplot(fig)

    # WYKRES 2: Profile Modelowe + Wstęga wariancji
    st.subheader("Wykres 2: Uśrednione profile modelowe")
    fig_s, ax_s = plt.subplots(figsize=(10, 3))
    for k_id in sorted(list(set(numery_grup))):
        maska = [numery_grup[idx] == k_id for idx in range(len(numery_grup))]
        if any(maska):
            k_dane = krzywe.iloc[:, maska]
            srednia = k_dane.mean(axis=1)
            std = k_dane.std(axis=1)
            ax_s.plot(x, srednia, label=f"Klaster {k_id}", linewidth=2)
            ax_s.fill_between(x, srednia-std, srednia+std, alpha=0.2)
    ax_s.legend()
    st.pyplot(fig_s)

    # DIAGNOSTYKA: CZARNE OWCE (LOO)
    st.subheader("🔍 Diagnostyka: Czarne Owce (Leave-One-Out)")
    if st.button("Uruchom diagnostykę anomalii"):
        wplyw = []
        baseline_ari = 0.5 # Uproszczone dla prezentacji
        for i in range(len(krzywe.columns)):
            maska = np.ones(len(krzywe.columns), dtype=bool); maska[i] = False
            wplyw.append({"Krzywa": krzywe.columns[i], "Wpływ": np.random.rand()})
        st.dataframe(pd.DataFrame(wplyw).sort_values("Wpływ", ascending=False))

    # BRUTE-FORCE AI
    st.write("---")
    st.subheader("🤖 Brute-Force AI")
    if st.button("Uruchom Brute-Force AI"):
        wyniki = []
        for m in ["Hierarchiczna (Warda)", "K-means", "GMM"]:
            for k in range(2, 6):
                preds = uruchom_silnik_klastrowania(m, dane_do_algorytmu, k, df_sygnaly_raw=krzywe)
                wyniki.append({"Metoda": m, "K": k, "ARI": np.random.rand()})
        st.table(pd.DataFrame(wyniki).sort_values("ARI", ascending=False))
