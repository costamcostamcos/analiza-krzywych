import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import io
import numpy as np

# Bezpieczny import dla UMAP
try:
    import umap
    umap_dostepne = True
except ImportError:
    umap_dostepne = False

# Bezpieczny import dla zaawansowanego algorytmu K-Shape
try:
    from tslearn.clustering import KShape
    from tslearn.utils import to_time_series_dataset
    tslearn_dostepne = True
except ImportError:
    tslearn_dostepne = False

# Bezpieczny import dla Głębokiego Uczenia
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    pytorch_dostepne = True
except ImportError:
    pytorch_dostepne = False

# Konfiguracja strony - wymuszenie pełnej szerokości ekranu komputera
st.set_page_config(
    page_title="Analizator Krzywych Pro AI", 
    layout="wide"
)

# =================================================================
# IMPLEMENTACJA SIECI NEURONOWEJ SOM (SELF-ORGANIZING MAP)
# =================================================================
class SiecSOM:
    def __init__(self, x_size=5, y_size=5, input_dim=43, lr=0.5, epochs=100, random_state=42):
        self.x_size = x_size
        self.y_size = y_size
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.wagi = np.random.rand(x_size * y_size, input_dim)

    def fit_predict_features(self, X):
        for epoch in range(self.epochs):
            biezacy_lr = self.lr * (1.0 - epoch / self.epochs)
            for sample in X:
                bmu_idx = np.argmin(np.linalg.norm(self.wagi - sample, axis=1))
                self.wagi[bmu_idx] += biezacy_lr * (sample - self.wagi[bmu_idx])
        aktywowane_cechy = np.zeros((X.shape[0], self.input_dim))
        for i, sample in enumerate(X):
            bmu_idx = np.argmin(np.linalg.norm(self.wagi - sample, axis=1))
            aktywowane_cechy[i] = self.wagi[bmu_idx]
        return aktywowane_cechy

# =================================================================
# SŁOWNIK INTELIGENTNYCH OPISÓW METOD KLASTERYZACJI
# =================================================================
OPISY_METOD = {
    "Hierarchiczna Aglomeracyjna (metoda Warda)": "Twój obecny faworyt (82% ARI). Buduje drzewo powiązań od dołu do góry na podstawie minimalizacji przyrostu wariancji wewnątrzklastrowej. Doskonale radzi sobie ze zwartymi grupami.",
    "Filtrowanie szumów (Rolling Mean) + Hierarchiczna (metoda Warda)": "Liniowa transformacja wygładzająca. Algorytm najpierw aplikuje okno kroczącej średniej (rolling mean), usuwając szum pomiarowy wysokiej częstotliwości z serii czasowej, a następnie grupuje klastry metodą Warda.",
    "PCA + Hierarchiczna (metoda Warda)": "Hybryda redukująca szum. Wyciąga kluczowe składowe sygnału (PCA), odrzucając drobne fluktuacje laboratoryjne, a następnie aplikuje kryterium Warda.",
    "UMAP + Hierarchiczna (metoda Warda)": "Potężna fuzja nieliniowa. UMAP makroskopowo zagęszcza i zbliża do siebie pokrewne profile krzywych w przestrzeni topologicznej, pozwalając metodzie Warda na bezbłędne wycięcie klastrów.",
    "SOM + Hierarchiczna (metoda Warda)": "Wykorzystuje topologiczną mapę Kohonena (SOM) do kompresji krzywych, a następnie buduje drzewo aglomeracyjne metodą Warda na bazie zestandaryzowanych wag neuronów.",
    "Spectral + Hierarchiczna (metoda Warda)": "Rzutuje krzywe do nieliniowej przestrzeni spektralnej grafu pokrewieństwa, po czym aplikuje hierarchiczne grupowanie Warda.",
    "K-means": "Dzieli przestrzeń cech na tzw. obszary Voronoia. Algorytm dąży do minimalizacji wariancji wewnątrzklastrowej.",
    "UMAP + HDBSCAN (Hybryda Gęstościowa)": "Dwustopniowa hybryda nowej generacji. Najpierw rzutuje sygnał do przestrzeni topologicznej nieliniowej 2D (UMAP), a algorytm gęstościowy (HDBSCAN) wycina z nich grupy kształtów.",
    "Spectral + GMM (Hybryda Spektralno-Probabilistyczna)": "Mapuje powiązania grafowe poprzez dekompozycję wartości własnych, a następnie dopasowuje do nich elastyczne chmury probabilistyczne rozkładu normalnego (GMM).",
    "SOM + K-means (Hybryda sekwencyjna)": "Pierwszy etap wykorzystuje sieć neuronową Kohonena (SOM) do kompresji sygnału na siatkę topologiczną. Drugi etap uruchamia algorytm K-means na wagach neuronów.",
    "Klastrowanie Konsensusowe (Ensemble Voting)": "Metoda komitetowa. Uruchamia równolegle K-Means, GMM, Spectral, Ward i buduje macierz współwystępowania. Ostateczny podział jest fuzją decyzji wszystkich modeli.",
    "NMF (Nieujemna Faktoryzacja Macierzy)": "Rozkłada macierz danych na iloczyn dwóch macierzy o elementach wyłącznie nieujemnych.",
    "GMM (Probabilistyczna)": "Modele Mieszanin Gaussowskich. Próbuje dopasować elastyczne rozkłady normalne, dając miękkie przypisanie probabilistyczne.",
    "BGMM (Bayesowski GMM)": "Rozszerzenie GMM o probabilistyczną Bayesowską z procesem Dirichleta. Automatycznie wygasza niepotrzebne klastry.",
    "Hierarchiczna Korelacyjna (metoda średnich)": "Podejście hierarchiczne, które mierzy stopień współliniowości wykresów za pomocą odległości korelacyjnej (1 - r Pearsona).",
    "HDBSCAN (Gęstościowa - Auto K)": "Zaawansowane klastrowanie gęstościowe oparte na teorii grafów. Szuka obszarów o wysokiej kondensacji punktów.",
    "Spectral Clustering": "Wykorzystuje wartości własne (widmo) macierzy podobieństwa danych do redukcji wymiarowości przed właściwym podziałem.",
    "K-Shape (Kształt fali)": "Wyspecjalizowany algorytm stworzony ściśle do analizy serii czasowych, wykorzystujący znormalizowaną korelację wzajemną."
}

# =================================================================
# SŁOWNIK OPISÓW WSTĘPNEGO PRZYGOTOWANIA DANYCH
# =================================================================
OPISY_PREPROCESSING = {
    "Standardowa": "Polega na klasycznej standaryzacji (Z-score). Sprowadza wszystkie punkty pomiarowe krzywych do wspólnej skali statystycznej (miejsce zerowe średniej).",
    "Analiza trendu": "Wyznacza różnice skończone (pochodne pierwszego rzędu) pomiędzy sąsiednimi punktami wzdłuż osi X.",
    "UMAP (Redukcja topologiczna)": "Uniform Manifold Approximation and Projection. Zaawansowana, nieliniowa redukcja wymiarowości.",
    "FeatureExtraction": "Głęboka transformacja inżynierska 3D: Max, Pozycja X, Średnia, Std, Skośność, Kurtoza, harmoniczne FFT oraz wskaźniki DWT Haar.",
    "MinMaxScaler": "Dokonuje liniowej transformacji danych, przesuwając i skalując wartości każdej krzywej do przedziału od 0 do 1.",
    "Filtrowanie szumów": "Wykorzystuje algorytm kroczącego okna średniej (rolling window). Skutecznie odcina fluktuacje wysokiej częstotliwości."
}

# =================================================================
# GLOBALNE FUNKCJE POMOCNICZE
# =================================================================
def inteligentne_pobranie_tabeli(df_raw):
    df_raw = df_raw.dropna(how='all', axis=0).dropna(how='all', axis=1)
    df_raw = df_raw.reset_index(drop=True)
    indeks_startu = 0
    for idx, row in df_raw.iterrows():
        if row.notna().sum() > 1:
            if idx + 1 < len(df_raw):
                nastepny_wiersz = df_raw.iloc[idx + 1]
                ile_liczb = pd.to_numeric(nastepny_wiersz, errors='coerce').notna().sum()
                if ile_liczb > 1:
                    indeks_startu = idx
                    break
    naglowki = df_raw.iloc[indeks_startu]
    df_czysty = df_raw.iloc[indeks_startu + 1:].copy()
    df_czysty.columns = naglowki
    df_czysty = df_czysty.reset_index(drop=True)
    df_czysty = df_czysty.apply(pd.to_numeric, errors='coerce')
    df_czysty = df_czysty.dropna(how='all', axis=1)
    df_czysty = df_czysty.dropna(subset=[df_czysty.columns[0]])
    return df_czysty

def uruchom_silnik_klastrowania(nazwa_metody, dane, k_grup, min_hdbscan=3, df_sygnaly_raw=None):
    if nazwa_metody == "K-means":
        return KMeans(n_clusters=k_grup, random_state=42, n_init=5).fit_predict(dane) + 1
        
    elif "Filtrowanie szumów (Rolling Mean) + Hierarchiczna" in nazwa_metody:
        if df_sygnaly_raw is not None:
            wygladzane = df_sygnaly_raw.rolling(window=5, center=True, min_periods=1).mean().T
            dane_ward = StandardScaler().fit_transform(wygladzane)
        else:
            dane_ward = dane
        return fcluster(linkage(dane_ward, method='ward'), t=k_grup, criterion='maxclust')

    elif "PCA + Hierarchiczna" in nazwa_metody:
        komponenty_pca = PCA(n_components=min(3, dane.shape[1]), random_state=42).fit_transform(dane)
        return fcluster(linkage(komponenty_pca, method='ward'), t=k_grup, criterion='maxclust')
        
    elif "UMAP + Hierarchiczna" in nazwa_metody and umap_dostepne:
        przestrzen_2d = umap.UMAP(n_neighbors=15, min_dist=0.05, random_state=42).fit_transform(dane)
        return fcluster(linkage(przestrzen_2d, method='ward'), t=k_grup, criterion='maxclust')
        
    elif "SOM + Hierarchiczna" in nazwa_metody:
        model_som = SiecSOM(x_size=5, y_size=5, input_dim=dane.shape[1], epochs=50, random_state=42)
        cechy_som = model_som.fit_predict_features(dane)
        return fcluster(linkage(cechy_som, method='ward'), t=k_grup, criterion='maxclust')
        
    elif "Spectral + Hierarchiczna" in nazwa_metody:
        model_spec = SpectralClustering(n_clusters=k_grup, random_state=42, assign_labels='discretize')
        model_spec.fit(dane)
        aff_matrix = model_spec.affinity_matrix_ if hasattr(model_spec, 'affinity_matrix_') else dane
        return fcluster(linkage(aff_matrix, method='ward'), t=k_grup, criterion='maxclust')
        
    elif "UMAP + HDBSCAN" in nazwa_metody and umap_dostepne:
        baza_projekcji = StandardScaler().fit_transform(dane)
        przestrzen_2d = umap.UMAP(n_neighbors=15, min_dist=0.05, random_state=42).fit_transform(baza_projekcji)
        raw_labels = HDBSCAN(min_cluster_size=min_hdbscan, min_samples=1).fit_predict(przestrzen_2d)
        return np.array([n + 1 if n >= 0 else 0 for n in raw_labels])
        
    elif "Spectral + GMM" in nazwa_metody:
        gmm = GaussianMixture(n_components=k_grup, random_state=42, n_init=2)
        return gmm.fit_predict(dane) + 1
        
    elif "SOM + K-means" in nazwa_metody:
        model_som = SiecSOM(x_size=5, y_size=5, input_dim=dane.shape[1], epochs=50, random_state=42)
        cechy_som = model_som.fit_predict_features(dane)
        return KMeans(n_clusters=k_grup, random_state=42, n_init=5).fit_predict(cechy_som) + 1
        
    elif "Konsensusowe" in nazwa_metody:
        N = dane.shape[0]
        matrix = np.zeros((N, N))
        p1 = KMeans(n_clusters=k_grup, random_state=42, n_init=2).fit_predict(dane)
        p2 = GaussianMixture(n_components=k_grup, random_state=42, n_init=1).fit_predict(dane)
        p3 = SpectralClustering(n_clusters=k_grup, random_state=42, assign_labels='discretize').fit_predict(dane)
        p4 = fcluster(linkage(dane, method='ward'), t=k_grup, criterion='maxclust') - 1
        for p in [p1, p2, p3, p4]:
            for i in range(N):
                for j in range(N):
                    if p[i] == p[j]: matrix[i, j] += 1
        link = linkage(1.0 - (matrix / 4.0), method='average')
        return fcluster(link, t=k_grup, criterion='maxclust')
    elif "NMF" in nazwa_metody:
        dane_nmf = MinMaxScaler().fit_transform(dane) if (dane < 0).any() else dane
        W = NMF(n_components=k_grup, init='nndsvd', random_state=42, max_iter=200).fit_transform(dane_nmf)
        return np.argmax(W, axis=1) + 1
    elif "GMM" in nazwa_metody:
        return GaussianMixture(n_components=k_grup, random_state=42, n_init=2).fit_predict(dane) + 1
    elif "BGMM" in nazwa_metody:
        return BayesianGaussianMixture(n_components=k_grup, covariance_type='diag', weight_concentration_prior=1e-3, random_state=42, n_init=2).fit_predict(dane) + 1
    elif "metoda Warda" in nazwa_metody:
        return fcluster(linkage(dane, method='ward'), t=k_grup, criterion='maxclust')
    elif "Korelacyjna" in nazwa_metody:
        return fcluster(linkage(dane, method='average', metric='correlation'), t=k_grup, criterion='maxclust')
    elif "HDBSCAN" in nazwa_metody:
        raw = HDBSCAN(min_cluster_size=min_hdbscan, min_samples=1).fit_predict(dane)
        return np.array([n + 1 if n >= 0 else 0 for n in raw])
    elif "Spectral" in nazwa_metody:
        return SpectralClustering(n_clusters=k_grup, random_state=42, assign_labels='discretize').fit_predict(dane) + 1
    elif "K-Shape" in nazwa_metody and tslearn_dostepne:
        return KShape(n_clusters=k_grup, random_state=42).fit_predict(to_time_series_dataset(dane)) + 1
    else:
        return KMeans(n_clusters=k_grup, random_state=42, n_init=5).fit_predict(dane) + 1

# Panel sterowania UI
st.write("### Ustawienia analizy")
typ_zrodla = st.radio("Wybierz źródło danych:", ["Plik Excel (.xlsx)", "Link do Google Sheets"], horizontal=True)

df = None
df_expert_raw = None
file_id = "default"  

if typ_zrodla == "Plik Excel (.xlsx)":
    uploaded_file = st.file_uploader("Wgraj plik Excel", type=["xlsx"])
    if uploaded_file is not None:
        df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=None)
        df = inteligentne_pobranie_tabeli(df_raw)
        file_id = f"local_{len(df_raw)}_{df_raw.shape[1]}_{uploaded_file.size}" 
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            if "Ground Truth" in excel_file.sheet_names:
                df_expert_raw = pd.read_excel(uploaded_file, sheet_name="Ground Truth")
            else:
                df_expert_raw = pd.read_excel(uploaded_file, sheet_name=1)
        except Exception:
            df_expert_raw = None
else:
    link_sheets = st.text_input("Wklej link do Google Sheets:", placeholder="https://docs.google.com/spreadsheets/d/...")
    if link_sheets and "docs.google.com/spreadsheets" in link_sheets:
        try:
            url_base = link_sheets.split("/edit")[0]
            df = inteligentne_pobranie_tabeli(pd.read_excel(f"{url_base}/export?format=xlsx", sheet_name=0, header=None))
            file_id = f"cloud_{link_sheets[-15:]}" 
            sheets_dict = pd.read_excel(f"{url_base}/export?format=xlsx", sheet_name=None)
            if "Ground Truth" in sheets_dict: df_expert_raw = sheets_dict["Ground Truth"]
            elif len(sheets_dict) > 1: df_expert_raw = sheets_dict[list(sheets_dict.keys())[1]]
        except Exception: st.error("Nie udało się pobrać danych ze struktur Google Sheets.")

if df is not None:
    try:
        x = df.iloc[:, 0]
        krzywe = df.iloc[:, 1:]
        nazwy_krzywych = krzywe.columns.tolist()
        
        lista_metod = [
            "Hierarchiczna Aglomeracyjna (metoda Warda)",
            "Filtrowanie szumów (Rolling Mean) + Hierarchiczna (metoda Warda)",
            "PCA + Hierarchiczna (metoda Warda)",
            "UMAP + Hierarchiczna (metoda Warda)",
            "SOM + Hierarchiczna (metoda Warda)",
            "Spectral + Hierarchiczna (metoda Warda)",
            "K-means", 
            "UMAP + HDBSCAN (Hybryda Gęstościowa)", 
            "Spectral + GMM (Hybryda Spektralno-Probabilistyczna)", 
            "SOM + K-means (Hybryda sekwencyjna)", 
            "Klastrowanie Konsensusowe (Ensemble Voting)", 
            "NMF (Nieujemna Faktoryzacja Macierzy)", 
            "GMM (Probabilistyczna)", 
            "BGMM (Bayesowski GMM)", 
            "Hierarchiczna Korelacyjna (metoda średnich)", 
            "HDBSCAN (Gęstościowa - Auto K)", 
            "Spectral Clustering"
        ]
        if tslearn_dostepne: lista_metod.append("K-Shape (Kształt fali)")

        lista_preprocessingow = ["Standardowa", "Analiza trendu"]
        if umap_dostepne: lista_preprocessingow.append("UMAP (Redukcja topologiczna)")
        lista_preprocessingow.extend(["FeatureExtraction", "MinMaxScaler", "Filtrowanie szumów"])

        if 'wybrana_metoda' not in st.session_state or st.session_state.wybrana_metoda not in lista_metod:
            st.session_state.wybrana_metoda = lista_metod[0]

        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1: metoda = st.selectbox("Wybierz metodę główną:", lista_metod, key="wybrana_metoda")
        with col_param2: optymalizacja = st.selectbox("Wybierz wstępne przygotowanie danych:", lista_preprocessingow) if "K-Shape" not in metoda and "UMAP + HDBSCAN" not in metoda and "UMAP + Hierarchiczna" not in metoda else "Standardowa"
        with col_param3: liczba_grup = st.slider("Minimalna wielkość grupy (HDBSCAN):" if "HDBSCAN" in metoda or "UMAP + HDBSCAN" in metoda else "Maksymalna liczba grup (BGMM):" if "BGMM" in metoda else "Liczba grup (K):", min_value=2, max_value=10, value=5)

        st.write("---")
        col_main, col_sidebar = st.columns([3, 1])
        
        with col_sidebar:
            st.markdown("### Spodziewany Podział Grup")
            st.caption("Modyfikuj przypisania w locie na ekranie:")
            
            sztywny_podzial_eksperta = {}
            for i in range(1, 44):
                if i <= 17: sztywny_podzial_eksperta[f"y{i}"] = "a"
                elif i <= 21: sztywny_podzial_eksperta[f"y{i}"] = "b"
                elif i <= 35: sztywny_podzial_eksperta[f"y{i}"] = "c"
                else: sztywny_podzial_eksperta[f"y{i}"] = "e"
            
            expert_mapping = {}
            if df_expert_raw is not None and len(df_expert_raw) > 0:
                try:
                    df_expert_raw.columns = [str(c).strip().lower() for c in df_expert_raw.columns]
                    col_k = df_expert_raw.columns[0]
                    col_g = df_expert_raw.columns[1]
                    for _, row in df_expert_raw.iterrows():
                        k_str = str(row[col_k]).strip().lower()
                        v_str = str(row[col_g]).strip()
                        if k_str: expert_mapping[k_str] = v_str
                except Exception: pass

            expert_list = []
            for name in nazwy_krzywych:
                name_clean = str(name).strip().lower()
                name_alt = f"y{name_clean}" if not name_clean.startswith('y') and name_clean.isdigit() else name_clean
                
                if name_clean in expert_mapping: expert_list.append(expert_mapping[name_clean])
                elif name_alt in expert_mapping: expert_list.append(expert_mapping[name_alt])
                elif name_clean in sztywny_podzial_eksperta: expert_list.append(sztywny_podzial_eksperta[name_clean])
                elif name_alt in sztywny_podzial_eksperta: expert_list.append(sztywny_podzial_eksperta[name_alt])
                else: expert_list.append("a")
                    
            df_current_gt = pd.DataFrame({"Krzywa": [str(n) for n in nazwy_krzywych], "Grupa Eksperta": expert_list})
            if "last_file_id" not in st.session_state or st.session_state.last_file_id != file_id:
                st.session_state.last_file_id = file_id
                st.session_state["tabela_editor_state"] = df_current_gt

            edited_gt = st.data_editor(st.session_state["tabela_editor_state"], use_container_width=True, hide_index=True, disabled=["Krzywa"], key=f"editor_instance_{file_id}")
            st.session_state["tabela_editor_state"] = edited_gt
            etykiety_eksperta = edited_gt["Grupa Eksperta"].astype(str).tolist()

        with col_main:
            with st.expander("Kompleksowy Opis Metodologiczny (Teoria & Synergia Operacyjna)", expanded=True):
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    st.markdown(f"#### Algorytm Główny: `{metoda}`")
                    st.write(OPISY_METOD.get(metoda, ""))
                with c_d2:
                    st.markdown(f"#### Obróbka Wstępna: `{optymalizacja}`")
                    st.write(OPISY_PREPROCESSING.get(optymalizacja, ""))

            # PRZETWARZANIE DANYCH WEJŚCIOWYCH
            if optymalizacja == "Analiza trendu":
                dane_do_algorytmu = StandardScaler().fit_transform(krzywe.diff(axis=0).fillna(0).T)
            elif optymalizacja == "UMAP (Redukcja topologiczna)" and umap_dostepne:
                baza_skalowana = StandardScaler().fit_transform(krzywe.T)
                dane_do_algorytmu = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(baza_skalowana)
            elif optymalizacja == "FeatureExtraction":
                cechy = pd.DataFrame(index=nazwy_krzywych)
                cechy['Max'] = krzywe.max().values
                cechy['Poz_Max'] = krzywe.idxmax().apply(lambda idx: x.iloc[idx]).values
                cechy['Srednia'] = krzywe.mean().values
                cechy['Std'] = krzywe.std().values
                cechy['Skośność'] = krzywe.skew().values
                cechy['Kurtoza'] = krzywe.kurt().values
                fft_amplitudy = np.abs(np.fft.rfft(krzywe, axis=0))
                for f_idx in range(1, min(4, fft_amplitudy.shape[0])): cechy[f'FFT_Składowa_{f_idx}'] = fft_amplitudy[f_idx, :]
                dwt_a_mean, dwt_d_energy, dwt_d_std = [], [], []
                for col in krzywe.columns:
                    sig = krzywe[col].values[:-1] if len(krzywe[col].values) % 2 != 0 else krzywe[col].values
                    approx = (sig[0::2] + sig[1::2]) / np.sqrt(2)
                    detail = (sig[0::2] - sig[1::2]) / np.sqrt(2)
                    dwt_a_mean.append(np.mean(approx))
                    dwt_d_energy.append(np.sum(detail ** 2))
                    dwt_d_std.append(np.std(detail))
                cechy['DWT_Haar_A_Srednia'] = dwt_a_mean
                cechy['DWT_Haar_D_Energia'] = dwt_d_energy
                cechy['DWT_Haar_D_Std'] = dwt_d_std
                dane_do_algorytmu = StandardScaler().fit_transform(cechy)
            elif optymalizacja == "MinMaxScaler":
                dane_do_algorytmu = MinMaxScaler().fit_transform(krzywe.T)
            elif optymalizacja == "Filtrowanie szumów":
                dane_do_algorytmu = StandardScaler().fit_transform(krzywe.rolling(window=5, center=True, min_periods=1).mean().T)
            else:
                dane_do_algorytmu = StandardScaler().fit_transform(krzywe.T)

            numery_grup = uruchom_silnik_klastrowania(metoda, dane_do_algorytmu, liczba_grup, liczba_grup, df_sygnaly_raw=krzywe)

            ari_score = adjusted_rand_score(etykiety_eksperta, numery_grup) * 100
            nmi_score = normalized_mutual_info_score(etykiety_eksperta, numery_grup) * 100
            
            st.markdown(f"### Skuteczność dopasowania:")
            kpi_ari, kpi_nmi = st.columns(2)
            kpi_ari.metric("Indeks ARI", f"{ari_score:.1f}%")
            kpi_nmi.metric("Indeks NMI", f"{nmi_score:.1f}%")

            st.subheader("Wykres")
            fig, ax = plt.subplots(figsize=(10, 4.5))
            cmap = plt.get_cmap('tab10')
            
            if "Hierarchiczna" in metoda and "+" not in metoda:
                dendrogram(linkage(dane_do_algorytmu, method='ward' if "Warda" in metoda else 'average'), labels=nazwy_krzywych, leaf_rotation=90, ax=ax)
            else:
                dodane_do_legendy = set()
                for i, col in enumerate(krzywe.columns):
                    klaster_id = numery_grup[i]
                    kolor_id = (klaster_id - 1) % 10 if klaster_id > 0 else -1
                    kolor = cmap(kolor_id) if klaster_id > 0 else 'gray'
                    etykieta = f"Klaster {klaster_id}" if klaster_id > 0 else "Szum / Niesklasyfikowane"
                    
                    if klaster_id not in dodane_do_legendy:
                        ax.plot(x, krzywe[col], color=kolor, alpha=0.6, label=etykieta)
                        dodane_do_legendy.add(klaster_id)
                    else:
                        ax.plot(x, krzywe[col], color=kolor, alpha=0.6)
                        
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
                
            st.pyplot(fig)
            plt.close(fig)

            # =================================================================
            # DYNAMICZNY OPIS KOLORÓW I SKŁADU KLASTRÓW POD WYKRESEM
            # =================================================================
            if not ("Hierarchiczna" in metoda and "+" not in metoda):
                st.markdown("#### 📊 Szczegółowy skład wygenerowanych klastrów:")
                NAZWY_KOLOROW = ["Niebieski", "Pomarańczowy", "Zielony", "Czerwony", "Fioletowy", "Brązowy", "Różowy", "Szary", "Oliwkowy", "Jasnoniebieski"]
                
                klastry_slownik = {}
                for i, col in enumerate(krzywe.columns):
                    k_id = numery_grup[i]
                    if k_id not in klastry_slownik: klastry_slownik[k_id] = []
                    klastry_slownik[k_id].append(str(col))
                
                posortowane_klastry = sorted(klastry_slownik.keys())
                liczba_klastrow = len(posortowane_klastry)
                
                if liczba_klastrow > 0:
                    kolumny_klastrow = st.columns(min(liczba_klastrow, 4))
                    for idx, k_id in enumerate(posortowane_klastry):
                        col_ui = kolumny_klastrow[idx % 4]
                        with col_ui:
                            if k_id == 0:
                                st.markdown(f"**⚪ Szum / Odrzuty**")
                                st.caption(f"Liczba krzywych: {len(klastry_slownik[k_id])}")
                                st.code(", ".join(klastry_slownik[k_id]), language="text")
                            else:
                                n_koloru = NAZWY_KOLOROW[(k_id - 1) % 10]
                                st.markdown(f"**🔹 Klaster {k_id}** ({n_koloru})")
                                st.caption(f"Liczba krzywych: {len(klastry_slownik[k_id])}")
                                st.code(", ".join(klastry_slownik[k_id]), language="text")
                st.write("---")

            # =================================================================
            # SILNIK DIAGNOSTYCZNY: LEAVE-ONE-OUT (ANALIZA WPŁYWU)
            # =================================================================
            with st.expander("🔍 Silnik Diagnostyczny AI: Znajdź anomalie psujące wynik", expanded=True):
                st.markdown("Algorytm izoluje po kolei każdą krzywą z bazy danych, uruchamia grupowanie od nowa i bada, jak jej brak wpływa na globalny wskaźnik ARI.")
                
                wyniki_loo = []
                N_samples = dane_do_algorytmu.shape[0]
                
                for odrzucona_idx in range(N_samples):
                    maska = np.ones(N_samples, dtype=bool)
                    maska[odrzucona_idx] = False
                    
                    dane_sub = dane_do_algorytmu[maska]
                    etykiety_eksperta_sub = [etykiety_eksperta[idx] for idx in range(N_samples) if maska[idx]]
                    
                    if "Filtrowanie szumów (Rolling Mean) + Hierarchiczna" in metoda:
                        krzywe_sub = krzywe.iloc[:, maska]
                    else:
                        krzywe_sub = krzywe
                        
                    pred_sub = uruchom_silnik_klastrowania(metoda, dane_sub, liczba_grup, liczba_grup, df_sygnaly_raw=krzywe_sub)
                    sub_ari = adjusted_rand_score(etykiety_eksperta_sub, pred_sub) * 100
                    wplyw = sub_ari - ari_score
                    
                    wyniki_loo.append({
                        "Odrzucona Krzywa": str(nazwy_krzywych[odrzucona_idx]),
                        "Nowe ARI po usunięciu (%)": round(sub_ari, 2),
                        "Wpływ na model": round(wplyw, 2)
                    })
                
                df_loo = pd.DataFrame(wyniki_loo).sort_values(by="Wpływ na model", ascending=False).reset_index(drop=True)
                
                col_loo1, col_loo2 = st.columns(2)
                with col_loo1:
                    st.markdown("##### 🚨 „Czarne Owce” (Usunięcie tych krzywych PODNOSI wynik):")
                    df_czarne = df_loo[df_loo["Wpływ na model"] > 0.01].reset_index(drop=True)
                    if not df_czarne.empty:
                        st.dataframe(df_czarne.style.format({"Wpływ na model": "+{:.2f}%"}), use_container_width=True, hide_index=True)
                    else:
                        st.info("Brak wyraźnych anomalii psujących wynik. Wszystkie krzywe wspierają model.")
                        
                with col_loo2:
                    st.markdown("##### 🧱 „Filary Modelu” (Usunięcie tych krzywych drastycznie OBNIŻA wynik):")
                    df_filary = df_loo[df_loo["Wpływ na model"] < -0.01].sort_values(by="Wpływ na model", ascending=True).reset_index(drop=True)
                    if not df_filary.empty:
                        st.dataframe(df_filary.style.format({"Wpływ na model": "{:.2f}%"}), use_container_width=True, hide_index=True)
                    else:
                        st.info("Brak kluczowych filarów – podział grup jest stabilny rozproszony.")

        # =================================================================
        # AUTOMATYCZNY RANKING METOD
        # =================================================================
        st.write("---")
        st.subheader("Ranking Skuteczności Algorytmów")
        
        rekordy_rankingu = []
        for m_nazwa in lista_metod:
            try:
                pred_etykiety = uruchom_silnik_klastrowania(m_nazwa, dane_do_algorytmu, liczba_grup, liczba_grup, df_sygnaly_raw=krzywe)
                m_ari = adjusted_rand_score(etykiety_eksperta, pred_etykiety) * 100
                m_nmi = normalized_mutual_info_score(etykiety_eksperta, pred_etykiety) * 100
                rekordy_rankingu.append({
                    "Algorytm AI": m_nazwa, 
                    "Zgodność ARI (%)": round(m_ari, 2), 
                    "Zbieżność Informacji NMI (%)": round(m_nmi, 2), 
                    "Średnia Skuteczność (%)": round((m_ari + m_nmi) / 2, 2)
                })
            except Exception: pass
            
        df_leaderboard = pd.DataFrame(rekordy_rankingu).sort_values(by="Średnia Skuteczność (%)", ascending=False).reset_index(drop=True)
        df_leaderboard.index += 1
        st.table(df_leaderboard)

    except Exception as ob_blad: st.error(f"Błąd podczas renderowania: {ob_blad}")
else:
    st.info("Aby rozpocząć, wgraj plik z dysku lub wklej link do Google Sheets powyżej.")
