import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
    "Hierarchiczna Aglomeracyjna (metoda Warda)": "Buduje drzewo powiązań od dołu do góry na podstawie minimalizacji przyrostu wariancji wewnątrzklastrowej. Doskonale radzi sobie ze zwartymi grupami.",
    "Filtrowanie szumów (Rolling Mean) + Hierarchiczna (metoda Warda)": "Liniowa transformacja wygładzająca. Algorytm najpierw aplikuje okno kroczącej średniej (rolling mean), usuwając szum pomiarowy wysokiej częstotliwości z serii czasowej, a następnie grupuje klastry metodą Warda.",
    "PCA + Hierarchiczna (metoda Warda)": "Hybryda redukująca szum. Wyciąga kluczowe składowe sygnału (PCA), odrzucając drobne fluktuacje laboratoryjne, a następnie aplikuje kryterium Warda.",
    "UMAP + Hierarchiczna (metoda Warda)": "Potężna fuzja nieliniowa. UMAP makroskopowo zagęszcza i zbliża do siebie pokrewne profile krzywych w przestrzeni topologicznej, pozwalając metodzie Warda na bezbłędne wycięcie klastrów.",
    "SOM + Hierarchiczna (metoda Warda)": "Wykorzystuje topologiczną mapę Kohonena (SOM) do kompresji krzywych, a następnie buduje drzewo aglomeracyjne metodą Warda na bazie zestandaryzowanych wag neuronów.",
    "Spectral + Hierarchiczna (metoda Warda)": "Rzutuje krzywe do nieliniowej przestrzeni spektralnej grafu pokrewieństwa, po czym aplikuje hierarchiczne grupowanie Warda.",
    "K-means": "Dzieli przestrzeń cech na tzw. obszary Voronoia. Algorytm dąży do minimalizacji wariancji wewnątrzklastrowej.",
    "UMAP + HDBSCAN (Hybryda Gęstościowa)": "Dwustopniowa hybryda nowej generacji. Najpierw rzutuje sygnał do przestrzeni topologicznej nieliniowej 2D (UMAP), a algorytm gęstościowy (HDBSCAN) wycina z nich grupy kształtów.",
    "Spectral + GMM (Hybryda Spektralno-Probabilistyczna)": "Mapuje powiązania grafowe poprzez dekompozycję wartości własnych, a następnie dopasowuje do nich elastyczne chmury probabilistyczne rozkładu normalnego (GMM).",
    "SOM + K-means (Hybryda sekwencyjna)": "Pierwszy etap wykorzystuje sieć neuronową Kohonena (SOM) do kompresji sygnału na siatkę topologiczna. Drugi etap uruchamia algorytm K-means na wagach neuronów.",
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
    "Standardowa": "Polega na klasycznej standaryzacji (Z-score). Sprowadza wszystkie punkty pomiarowe krzywych do wspólnej skali statystycznej.",
    "Analiza trendu": "Wyznacza różnice skończone (pochodne pierwszego rzędu) pomiędzy sąsiednimi punktami wzdłuż osi X.",
    "UMAP (Redukcja topologiczna)": "Uniform Manifold Approximation and Projection. Zaawansowana, nieliniowa redukcja wymiarowości.",
    "FeatureExtraction": "Głęboka transformacja inżynierska 3D: Max, Pozycja X, Średnia, Std, Skośność, Kurtoza, harmoniczne FFT oraz wskaźniki DWT Haar.",
    "MinMaxScaler": "Dokonuje liniowej transformacji danych, przesuwając i skalując wartości każdej krzywej do przedziału od 0 do 1.",
    "Filtrowanie szumów": "Wykorzystuje algorytm kroczącego okna średniej (rolling window). Skutecznie odcina fluktuacje wysokiej częstotliwości.",
    "Augmentacja sygnału": "Data Augmentation dla sieci neuronowych. Sztucznie rozbudowuje zbiór danych przez generowanie zaszumionych wariantów każdej krzywej. Dostępne techniki: Jitter (szum Gaussowski), Time Warping (deformacja osi czasu), Amplitude Scaling (losowe skalowanie amplitudy), Window Slicing (losowe przycięcie okna) oraz Permutation (przestawienie segmentów)."
}

OPISY_AUGMENTACJI = {
    "Jitter": "Dodaje do każdej krzywej losowy szum Gaussowski. Symuluje szum pomiarowy — najprostsza i najszybsza technika augmentacji.",
    "Time Warping": "Losowo rozciąga i ściska oś czasu przez interpolację na odkształconej siatce punktów. Symuluje zmienną prędkość procesu.",
    "Amplitude Scaling": "Mnoży amplitudę każdej krzywej przez losowy współczynnik bliski 1.0. Symuluje zmienność wzmocnienia sygnału.",
    "Window Slicing": "Wycina losowy fragment krzywej i rozciąga go z powrotem do oryginalnej długości. Uczy modelu rozpoznawania lokalnych wzorców.",
    "Permutation": "Dzieli krzywą na segmenty i losowo je przestawia. Testuje odporność modelu na zmiany kolejności fragmentów sygnału."
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


# =================================================================
# FUNKCJA AUGMENTACJI SYGNAŁU (DATA AUGMENTATION)
# =================================================================

def augmentuj_sygnal(krzywe_df, technika, sila, n_kopii, random_state=42):
    """
    Generuje n_kopii augmentowanych wariantów każdej krzywej i zwraca
    rozszerzony DataFrame wraz z etykietami źródłowymi (nazwa oryginału).
    sila: float 0.0–1.0 — intensywność przekształcenia
    """
    rng = np.random.default_rng(random_state)
    wyniki = {}
    etykiety_zrodlowe = {}  # nazwa_aug -> nazwa_oryginalna

    # Zachowaj oryginały
    for col in krzywe_df.columns:
        wyniki[str(col)] = krzywe_df[col].values.copy()
        etykiety_zrodlowe[str(col)] = str(col)

    n_punktow = len(krzywe_df)

    for col in krzywe_df.columns:
        syg = krzywe_df[col].values.astype(float)

        for k in range(1, n_kopii + 1):
            nazwa_aug = f"{col}_aug{k}"

            if technika == "Jitter":
                # Szum Gaussowski skalowany siłą * std sygnału
                szum = rng.normal(0, sila * np.std(syg), size=n_punktow)
                aug = syg + szum

            elif technika == "Time Warping":
                # Odkształcenie osi czasu: losowa krzywa warpingu przez interpolację
                t_orig = np.linspace(0, 1, n_punktow)
                # Generuj losowe węzły warpingu
                n_wezlow = max(4, int(n_punktow * 0.1))
                wezly = np.sort(rng.uniform(0, 1, n_wezlow))
                wezly = np.concatenate([[0], wezly, [1]])
                # Zaburz węzły o sila
                zaburzenie = rng.uniform(-sila * 0.3, sila * 0.3, len(wezly))
                zaburzenie[0] = 0
                zaburzenie[-1] = 0
                t_warp = np.clip(wezly + zaburzenie, 0, 1)
                t_warp = np.sort(t_warp)
                t_nowe = np.interp(t_orig, t_warp, t_orig)
                aug = np.interp(t_nowe, t_orig, syg)

            elif technika == "Amplitude Scaling":
                # Losowy współczynnik skalowania amplitudy
                wspolczynnik = rng.uniform(1.0 - sila * 0.5, 1.0 + sila * 0.5)
                aug = syg * wspolczynnik

            elif technika == "Window Slicing":
                # Wycina fragment (1-sila)..1.0 długości i rozciąga do pełnej
                min_dlugosc = max(int(n_punktow * (1.0 - sila * 0.4)), 3)
                dlugosc_okna = rng.integers(min_dlugosc, n_punktow)
                start = rng.integers(0, n_punktow - dlugosc_okna + 1)
                wycinek = syg[start: start + dlugosc_okna]
                aug = np.interp(
                    np.linspace(0, 1, n_punktow),
                    np.linspace(0, 1, dlugosc_okna),
                    wycinek
                )

            elif technika == "Permutation":
                # Dzieli na segmenty i losowo je przestawia
                n_segmentow = max(2, int(2 + sila * 8))
                granice = np.array_split(np.arange(n_punktow), n_segmentow)
                kolejnosc = rng.permutation(len(granice))
                aug = np.concatenate([syg[granice[i]] for i in kolejnosc])

            else:
                aug = syg.copy()

            wyniki[nazwa_aug] = aug
            etykiety_zrodlowe[nazwa_aug] = str(col)

    df_aug = pd.DataFrame(wyniki, index=krzywe_df.index)
    return df_aug, etykiety_zrodlowe


@st.cache_data(show_spinner=False)
def uruchom_silnik_klastrowania(nazwa_metody, dane, k_grup, min_hdbscan=3, _df_sygnaly_raw=None):
    df_sygnaly_raw = _df_sygnaly_raw
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
                    if p[i] == p[j]:
                        matrix[i, j] += 1
        link = linkage(1.0 - (matrix / 4.0), method='average')
        return fcluster(link, t=k_grup, criterion='maxclust')

    elif "NMF" in nazwa_metody:
        dane_nmf = MinMaxScaler().fit_transform(dane) if (dane < 0).any() else dane
        W = NMF(n_components=k_grup, init='nndsvd', random_state=42, max_iter=200).fit_transform(dane_nmf)
        return np.argmax(W, axis=1) + 1

    elif nazwa_metody == "GMM (Probabilistyczna)":
        return GaussianMixture(n_components=k_grup, random_state=42, n_init=2).fit_predict(dane) + 1

    elif "BGMM" in nazwa_metody:
        return BayesianGaussianMixture(
            n_components=k_grup, covariance_type='diag',
            weight_concentration_prior=1e-3, random_state=42, n_init=2
        ).fit_predict(dane) + 1

    elif "metoda Warda" in nazwa_metody:
        return fcluster(linkage(dane, method='ward'), t=k_grup, criterion='maxclust')

    elif "Korelacyjna" in nazwa_metody:
        return fcluster(linkage(dane, method='average', metric='correlation'), t=k_grup, criterion='maxclust')

    elif nazwa_metody == "HDBSCAN (Gęstościowa - Auto K)":
        raw = HDBSCAN(min_cluster_size=min_hdbscan, min_samples=1).fit_predict(dane)
        return np.array([n + 1 if n >= 0 else 0 for n in raw])

    elif "Spectral Clustering" in nazwa_metody:
        return SpectralClustering(n_clusters=k_grup, random_state=42, assign_labels='discretize').fit_predict(dane) + 1

    elif "K-Shape" in nazwa_metody and tslearn_dostepne:
        return KShape(n_clusters=k_grup, random_state=42).fit_predict(to_time_series_dataset(dane)) + 1

    else:
        return fcluster(linkage(dane, method='ward'), t=k_grup, criterion='maxclust')


# =================================================================
# GŁÓWNY EKRAN APLIKACJI
# =================================================================

st.title("📊 Interaktywny Analizator Krzywych AI Pro")
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
            if "Ground Truth" in sheets_dict:
                df_expert_raw = sheets_dict["Ground Truth"]
            elif len(sheets_dict) > 1:
                df_expert_raw = sheets_dict[list(sheets_dict.keys())[1]]
        except Exception:
            st.error("Nie udało się pobrać danych ze struktur Google Sheets.")

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
        if tslearn_dostepne:
            lista_metod.append("K-Shape (Kształt fali)")

        lista_preprocessingow = ["Standardowa", "Analiza trendu"]
        if umap_dostepne:
            lista_preprocessingow.append("UMAP (Redukcja topologiczna)")
        lista_preprocessingow.extend(["FeatureExtraction", "MinMaxScaler", "Filtrowanie szumów", "Augmentacja sygnału"])

        if 'wybrana_metoda' not in st.session_state or st.session_state.wybrana_metoda not in lista_metod:
            st.session_state.wybrana_metoda = lista_metod[0]

        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            metoda = st.selectbox("Wybierz metodę główną:", lista_metod, key="wybrana_metoda")
        with col_param2:
            if "K-Shape" in metoda or "UMAP + HDBSCAN" in metoda or "UMAP + Hierarchiczna" in metoda:
                optymalizacja = "Standardowa"
            else:
                optymalizacja = st.selectbox("Wybierz wstępne przygotowanie danych:", lista_preprocessingow)
        with col_param3:
            if "HDBSCAN" in metoda or "UMAP + HDBSCAN" in metoda:
                slider_label = "Minimalna wielkość grupy (HDBSCAN):"
            elif "BGMM" in metoda:
                slider_label = "Maksymalna liczba grup (BGMM):"
            else:
                slider_label = "Liczba grup (K):"
            liczba_grup = st.slider(slider_label, min_value=2, max_value=10, value=5)

        st.write("---")

        # -----------------------------------------------------------------
        # PANEL AUGMENTACJI — widoczny tylko gdy wybrano tę opcję
        # -----------------------------------------------------------------
        aug_technika = "Jitter"
        aug_sila = 0.1
        aug_kopie = 2

        if optymalizacja == "Augmentacja sygnału":
            with st.expander("⚙️ Ustawienia Augmentacji Sygnału", expanded=True):
                col_aug1, col_aug2, col_aug3 = st.columns(3)
                with col_aug1:
                    aug_technika = st.selectbox(
                        "Technika augmentacji:",
                        ["Jitter", "Time Warping", "Amplitude Scaling", "Window Slicing", "Permutation"],
                        help="Wybierz metodę przekształcania krzywych"
                    )
                    st.caption(OPISY_AUGMENTACJI.get(aug_technika, ""))
                with col_aug2:
                    aug_sila = st.slider(
                        "Siła augmentacji:",
                        min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                        help="Im wyższa wartość, tym większe zniekształcenie sygnału"
                    )
                with col_aug3:
                    aug_kopie = st.slider(
                        "Liczba kopii na krzywą:",
                        min_value=1, max_value=10, value=2,
                        help="Ile augmentowanych wariantów wygenerować dla każdej krzywej"
                    )
                st.info(
                    f"Zbiór zostanie rozszerzony z **{len(krzywe.columns)}** do "
                    f"**{len(krzywe.columns) * (1 + aug_kopie)}** krzywych "
                    f"({aug_kopie} kopii × {len(krzywe.columns)} oryginałów + oryginały). "
                    f"Klasteryzacja działa na pełnym zbiorze, ARI/NMI liczone tylko dla oryginałów."
                )

        # =================================================================
        # WYSUWANY PANEL Z LEWEJ KRAWĘDZI — CSS/JS przez st.markdown
        # =================================================================

        # Buduj tabelkę eksperta
        sztywny_podzial_eksperta = {}
        for i in range(1, 44):
            if i <= 17:
                sztywny_podzial_eksperta[f"y{i}"] = "a"
            elif i <= 21:
                sztywny_podzial_eksperta[f"y{i}"] = "b"
            elif i <= 35:
                sztywny_podzial_eksperta[f"y{i}"] = "c"
            else:
                sztywny_podzial_eksperta[f"y{i}"] = "e"

        expert_mapping = {}
        if df_expert_raw is not None and len(df_expert_raw) > 0:
            try:
                df_expert_raw.columns = [str(c).strip().lower() for c in df_expert_raw.columns]
                col_k = df_expert_raw.columns[0]
                col_g = df_expert_raw.columns[1]
                for _, row in df_expert_raw.iterrows():
                    k_str = str(row[col_k]).strip().lower()
                    v_str = str(row[col_g]).strip()
                    if k_str:
                        expert_mapping[k_str] = v_str
            except Exception:
                pass

        expert_list = []
        for name in nazwy_krzywych:
            name_clean = str(name).strip().lower()
            name_alt = f"y{name_clean}" if not name_clean.startswith('y') and name_clean.isdigit() else name_clean
            if name_clean in expert_mapping:
                expert_list.append(expert_mapping[name_clean])
            elif name_alt in expert_mapping:
                expert_list.append(expert_mapping[name_alt])
            elif name_clean in sztywny_podzial_eksperta:
                expert_list.append(sztywny_podzial_eksperta[name_clean])
            elif name_alt in sztywny_podzial_eksperta:
                expert_list.append(sztywny_podzial_eksperta[name_alt])
            else:
                expert_list.append("a")

        df_current_gt = pd.DataFrame({"Krzywa": [str(n) for n in nazwy_krzywych], "Grupa Eksperta": expert_list})

        if "last_file_id" not in st.session_state or st.session_state.last_file_id != file_id:
            st.session_state.last_file_id = file_id
            st.session_state["tabela_editor_state"] = df_current_gt

        # Buduj HTML tabelki do wstrzyknięcia w panel
        rows_html = ""
        for _, row in st.session_state["tabela_editor_state"].iterrows():
            rows_html += f"<tr><td>{row['Krzywa']}</td><td>{row['Grupa Eksperta']}</td></tr>"

        # Dane eksportu CSV (base64) do przycisku w panelu
        df_eksport_panel = st.session_state.get("df_eksport_klastry", pd.DataFrame())
        eksport_dostepny = not df_eksport_panel.empty
        if eksport_dostepny:
            csv_str = df_eksport_panel.to_csv(index=False, encoding="utf-8-sig")
            import base64
            csv_b64 = base64.b64encode(csv_str.encode("utf-8-sig")).decode()
            nazwa_pliku_csv = f"sklady_klastrow_{metoda[:20].replace(' ', '_')}.csv"
        else:
            csv_b64 = ""
            nazwa_pliku_csv = "sklady_klastrow.csv"

        drawer_html = f"""
        <style>
        #drawer-toggle {{
            position: fixed;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            z-index: 9999;
            background: #1f77b4;
            color: white;
            border: none;
            border-radius: 0 8px 8px 0;
            padding: 14px 8px;
            cursor: pointer;
            font-size: 18px;
            writing-mode: vertical-rl;
            letter-spacing: 2px;
            box-shadow: 2px 0 8px rgba(0,0,0,0.18);
            transition: background 0.2s;
        }}
        #drawer-toggle:hover {{ background: #1557a0; }}

        #side-drawer {{
            position: fixed;
            left: -320px;
            top: 0;
            width: 320px;
            height: 100vh;
            background: #ffffff;
            border-right: 2px solid #1f77b4;
            box-shadow: 4px 0 18px rgba(0,0,0,0.15);
            z-index: 9998;
            transition: left 0.3s ease;
            display: flex;
            flex-direction: column;
            padding: 0;
            overflow: hidden;
        }}
        #side-drawer.open {{ left: 0; }}

        #drawer-header {{
            background: #1f77b4;
            color: white;
            padding: 14px 18px;
            font-weight: bold;
            font-size: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-shrink: 0;
        }}
        #drawer-close {{
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            line-height: 1;
        }}

        #drawer-body {{
            flex: 1;
            overflow-y: auto;
            padding: 14px 16px;
        }}

        #drawer-body h4 {{
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #444;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        #gt-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-bottom: 18px;
        }}
        #gt-table th {{
            background: #f0f4fa;
            padding: 6px 10px;
            text-align: left;
            border-bottom: 2px solid #1f77b4;
            position: sticky;
            top: 0;
        }}
        #gt-table td {{
            padding: 5px 10px;
            border-bottom: 1px solid #eee;
        }}
        #gt-table tr:hover td {{ background: #f5f9ff; }}

        .drawer-btn {{
            display: block;
            width: 100%;
            padding: 10px;
            margin-bottom: 8px;
            background: #1f77b4;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            text-align: center;
            text-decoration: none;
        }}
        .drawer-btn:hover {{ background: #1557a0; }}
        .drawer-btn.disabled {{
            background: #aaa;
            cursor: not-allowed;
        }}
        </style>

        <button id="drawer-toggle" onclick="toggleDrawer()">&#x276F;&#x276F; Panel</button>

        <div id="side-drawer">
            <div id="drawer-header">
                📋 Podział Grup &amp; Eksport
                <button id="drawer-close" onclick="toggleDrawer()">✕</button>
            </div>
            <div id="drawer-body">
                <h4>Spodziewany Podział Grup</h4>
                <table id="gt-table">
                    <thead><tr><th>Krzywa</th><th>Grupa</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>

                <h4>Pobierz skład klastrów</h4>
                {'<a class="drawer-btn" href="data:text/csv;base64,' + csv_b64 + '" download="' + nazwa_pliku_csv + '">⬇️ Pobierz CSV</a>' if eksport_dostepny else '<button class="drawer-btn disabled" disabled>⬇️ CSV — uruchom analizę</button>'}
            </div>
        </div>

        <script>
        function toggleDrawer() {{
            var d = document.getElementById('side-drawer');
            var t = document.getElementById('drawer-toggle');
            d.classList.toggle('open');
            t.innerHTML = d.classList.contains('open') ? '&#x276E;&#x276E; Panel' : '&#x276F;&#x276F; Panel';
        }}
        </script>
        """

        st.markdown(drawer_html, unsafe_allow_html=True)

        # Edytor tabelki w głównym obszarze (ukryty wizualnie przez CSS — dane z niego idą do session_state)
        with st.expander("📋 Edytuj Spodziewany Podział Grup", expanded=False):
            edited_gt = st.data_editor(
                st.session_state["tabela_editor_state"],
                width="stretch",
                hide_index=True,
                disabled=["Krzywa"],
                key=f"editor_instance_{file_id}"
            )
            st.session_state["tabela_editor_state"] = edited_gt
            st.caption("Zmiany są natychmiast widoczne w panelu bocznym po odświeżeniu.")

        etykiety_eksperta = edited_gt["Grupa Eksperta"].astype(str).tolist()

        with st.expander("Kompleksowy Opis Metodologiczny", expanded=True):
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    st.markdown(f"#### Algorytm Główny: `{metoda}`")
                    st.write(OPISY_METOD.get(metoda, ""))
                with c_d2:
                    st.markdown(f"#### Obróbka Wstępna: `{optymalizacja}`")
                    st.write(OPISY_PREPROCESSING.get(optymalizacja, ""))

        # -----------------------------------------------------------------
        # PRZETWARZANIE DANYCH WEJŚCIOWYCH
        # -----------------------------------------------------------------
        if optymalizacja == "Augmentacja sygnału":
            # Generuj rozszerzony zbiór krzywych
            krzywe_aug, etykiety_zrodlowe = augmentuj_sygnal(
                krzywe, aug_technika, aug_sila, aug_kopie, random_state=42
            )
            dane_do_algorytmu = StandardScaler().fit_transform(krzywe_aug.T)
            # Indeksy oryginalnych krzywych w rozszerzonym zbiorze (zawsze pierwsze N)
            indeksy_oryginalow = list(range(len(krzywe.columns)))
        elif optymalizacja == "Analiza trendu":
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
            cechy['Skosnosc'] = krzywe.skew().values
            cechy['Kurtoza'] = krzywe.kurt().values
            fft_amplitudy = np.abs(np.fft.rfft(krzywe, axis=0))
            for f_idx in range(1, min(4, fft_amplitudy.shape[0])):
                cechy[f'FFT_Skladowa_{f_idx}'] = fft_amplitudy[f_idx, :]
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
            dane_do_algorytmu = StandardScaler().fit_transform(
                krzywe.rolling(window=5, center=True, min_periods=1).mean().T
            )
        else:
            dane_do_algorytmu = StandardScaler().fit_transform(krzywe.T)

        if optymalizacja == "Augmentacja sygnału":
            numery_grup_aug = uruchom_silnik_klastrowania(
                metoda, dane_do_algorytmu, liczba_grup, liczba_grup,
                _df_sygnaly_raw=krzywe_aug
            )
            # Wyniki dla całego zbioru (do wykresów)
            numery_grup = numery_grup_aug
            nazwy_krzywych_aug = list(krzywe_aug.columns)
            # ARI/NMI tylko dla oryginalnych N krzywych
            numery_grup_oryg = numery_grup_aug[indeksy_oryginalow]
            ari_score = adjusted_rand_score(etykiety_eksperta, numery_grup_oryg) * 100
            nmi_score = normalized_mutual_info_score(etykiety_eksperta, numery_grup_oryg) * 100
            # Do wykresów i sekcji składu klastrów użyj oryginalnych krzywych
            krzywe_do_wykresu = krzywe
            nazwy_do_wykresu = nazwy_krzywych
            numery_grup_do_wykresu = numery_grup_oryg
        else:
            numery_grup = uruchom_silnik_klastrowania(metoda, dane_do_algorytmu, liczba_grup, liczba_grup, _df_sygnaly_raw=krzywe)
            ari_score = adjusted_rand_score(etykiety_eksperta, numery_grup) * 100
            nmi_score = normalized_mutual_info_score(etykiety_eksperta, numery_grup) * 100
            krzywe_do_wykresu = krzywe
            nazwy_do_wykresu = nazwy_krzywych
            numery_grup_do_wykresu = numery_grup

        st.markdown("### Skuteczność dopasowania:")
        kpi_ari, kpi_nmi = st.columns(2)
        kpi_ari.metric("Indeks ARI", f"{ari_score:.1f}%")
        kpi_nmi.metric("Indeks NMI", f"{nmi_score:.1f}%")

        # =================================================================
        # WYKRES 1: WSZYSTKIE KRZYWE (Z OKREŚLENIEM LEGENDY)
        # =================================================================
        PLOTLY_KOLORY = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]

        st.subheader("Wykres 1: Wszystkie sklasterowane krzywe")

        if "Hierarchiczna" in metoda and "+" not in metoda:
            # Dendrogram — liczba etykiet musi odpowiadać liczbie wierszy macierzy linkage
            cmap = plt.get_cmap('tab10')
            fig_dend, ax_dend = plt.subplots(figsize=(10, 4.2))
            if optymalizacja == "Augmentacja sygnału":
                # dane_do_algorytmu zawiera oryginały + kopie — użyj ich nazw
                etykiety_dendro = list(krzywe_aug.columns)
            else:
                etykiety_dendro = nazwy_do_wykresu
            dendrogram(
                linkage(dane_do_algorytmu, method='ward' if "Warda" in metoda else 'average'),
                labels=etykiety_dendro, leaf_rotation=90, ax=ax_dend
            )
            st.pyplot(fig_dend)
            plt.close(fig_dend)
        else:
            fig1 = go.Figure()
            dodane_do_legendy = set()
            for i, col in enumerate(krzywe_do_wykresu.columns):
                klaster_id = int(numery_grup_do_wykresu[i])
                if klaster_id > 0:
                    kolor = PLOTLY_KOLORY[(klaster_id - 1) % 10]
                    etykieta_grupy = f"Klaster {klaster_id}"
                else:
                    kolor = "#aaaaaa"
                    etykieta_grupy = "Szum / Odrzuty"
                fig1.add_trace(go.Scatter(
                    x=x,
                    y=krzywe_do_wykresu[col],
                    mode="lines",
                    name=etykieta_grupy,
                    legendgroup=etykieta_grupy,
                    showlegend=(klaster_id not in dodane_do_legendy),
                    line=dict(color=kolor, width=1.2),
                    opacity=0.6,
                    hovertemplate=f"<b>{col}</b><br>Klaster: {klaster_id if klaster_id > 0 else 'Szum'}<br>X: %{{x}}<br>Y: %{{y:.4f}}<extra></extra>"
                ))
                dodane_do_legendy.add(klaster_id)
            fig1.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(groupclick="toggleitem", bgcolor="rgba(255,255,255,0.8)", borderwidth=1),
                xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
                yaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
                hovermode="closest"
            )
            st.plotly_chart(fig1, use_container_width=True)

        # =================================================================
        # WYKRES 2: PROFILE MODELOWE (ŚREDNIE + CIEŃ WARIANCJI)
        # =================================================================
        if not ("Hierarchiczna" in metoda and "+" not in metoda):
            st.subheader("Wykres 2: Uśrednione profile modelowe (Wzorce kształtu fali)")

            fig2 = go.Figure()
            unikalne_klastry = sorted(list(set(numery_grup_do_wykresu)))

            for k_id in unikalne_klastry:
                maska_klastra = [numery_grup_do_wykresu[idx] == k_id for idx in range(len(numery_grup_do_wykresu))]
                krzywe_klastra = krzywe_do_wykresu.iloc[:, maska_klastra]
                if krzywe_klastra.shape[1] == 0:
                    continue

                profil_sredni = krzywe_klastra.mean(axis=1)
                profil_std = krzywe_klastra.std(axis=1).fillna(0)
                górna = profil_sredni + profil_std
                dolna = profil_sredni - profil_std

                if k_id > 0:
                    kolor = PLOTLY_KOLORY[(k_id - 1) % 10]
                    label_sredni = f"Wzorzec Klastra {k_id}"
                    liczba_krzywych = krzywe_klastra.shape[1]
                else:
                    kolor = "#aaaaaa"
                    label_sredni = "Średnia Szumu"
                    liczba_krzywych = krzywe_klastra.shape[1]

                # Wstęga ±1 sigma (fill_between)
                fig2.add_trace(go.Scatter(
                    x=list(x) + list(x[::-1]),
                    y=list(górna) + list(dolna[::-1]),
                    fill="toself",
                    fillcolor=kolor,
                    opacity=0.12,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=label_sredni
                ))
                # Linia średnia
                fig2.add_trace(go.Scatter(
                    x=x,
                    y=profil_sredni,
                    mode="lines",
                    name=label_sredni,
                    legendgroup=label_sredni,
                    line=dict(color=kolor, width=2.5),
                    hovertemplate=(
                        f"<b>{label_sredni}</b><br>"
                        f"Liczba krzywych: {liczba_krzywych}<br>"
                        "X: %{x}<br>Średnia: %{y:.4f}<extra></extra>"
                    )
                ))

            fig2.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=1),
                xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
                yaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
                hovermode="closest"
            )
            st.plotly_chart(fig2, use_container_width=True)

        # =================================================================
        # DYNAMICZNY OPIS KOLORÓW I SKŁADU KLASTRÓW POD WYKRESEM
        # =================================================================
        if not ("Hierarchiczna" in metoda and "+" not in metoda):
            st.markdown("#### 📊 Szczegółowy skład wygenerowanych klastrów:")

            NAZWY_KOLOROW = [
                "Niebieski", "Pomarańczowy", "Zielony", "Czerwony", "Fioletowy",
                "Brązowy", "Różowy", "Szary", "Oliwkowy", "Jasnoniebieski"
            ]

            klastry_slownik = {}
            for i, col in enumerate(krzywe_do_wykresu.columns):
                k_id = numery_grup_do_wykresu[i]
                if k_id not in klastry_slownik:
                    klastry_slownik[k_id] = []
                klastry_slownik[k_id].append(str(col))

            posortowane_klastry = sorted(klastry_slownik.keys())
            liczba_klastrow = len(posortowane_klastry)

            if liczba_klastrow > 0:
                kolumny_klastrow = st.columns(min(liczba_klastrow, 4))
                for idx, k_id in enumerate(posortowane_klastry):
                    col_ui = kolumny_klastrow[idx % 4]
                    with col_ui:
                        if k_id == 0:
                                st.markdown("**⚪ Szum / Odrzuty**")
                                st.caption(f"Liczba: {len(klastry_slownik[k_id])}")
                                st.code(", ".join(klastry_slownik[k_id]), language="text")
                        else:
                                n_koloru = NAZWY_KOLOROW[(k_id - 1) % 10]
                                st.markdown(f"**🔹 Klaster {k_id}** ({n_koloru})")
                                st.caption(f"Liczba: {len(klastry_slownik[k_id])}")
                                st.code(", ".join(klastry_slownik[k_id]), language="text")

            # Zapisz dane eksportu do session_state — pobierze je panel w col_sidebar
            wiersze_eksportu = []
            for k_id in posortowane_klastry:
                if k_id == 0:
                    nazwa_klastra = "Szum / Odrzuty"
                    nazwa_koloru = "Szary"
                else:
                    nazwa_klastra = f"Klaster {k_id}"
                    nazwa_koloru = NAZWY_KOLOROW[(k_id - 1) % 10]
                for krzywa in klastry_slownik[k_id]:
                    wiersze_eksportu.append({
                        "Krzywa": krzywa,
                        "Klaster": nazwa_klastra,
                        "Kolor": nazwa_koloru,
                        "Nr Klastra": k_id
                    })
            st.session_state["df_eksport_klastry"] = pd.DataFrame(wiersze_eksportu)
            st.session_state["eksport_metoda"] = metoda

            st.write("---")

        # =================================================================
        # MSE ANOMALY DETECTION — odległość krzywej od centroidu klastra
        # =================================================================
        with st.expander("🔍 Detekcja Anomalii MSE: Odległość od centroidu klastra", expanded=True):
            st.markdown(
                "Dla każdej krzywej obliczana jest **fizyczna odległość MSE** od wzorca (centroidu) "
                "jej klastra. Krzywe z MSE powyżej progu `μ + 2σ` są automatycznie oznaczane jako anomalie."
            )

            dane_mse = StandardScaler().fit_transform(krzywe_do_wykresu.T)
            wyniki_mse = []

            unikalne_k = sorted(set(numery_grup_do_wykresu))
            for k_id in unikalne_k:
                if k_id == 0:
                    continue  # pomijamy szum HDBSCAN
                maska_k = np.array(numery_grup_do_wykresu) == k_id
                dane_klastra = dane_mse[maska_k]
                centroid = dane_klastra.mean(axis=0)
                for i, (nalezy, col) in enumerate(zip(maska_k, krzywe_do_wykresu.columns)):
                    if nalezy:
                        mse = float(np.mean((dane_mse[i] - centroid) ** 2))
                        wyniki_mse.append({
                            "Krzywa": str(col),
                            "Klaster": k_id,
                            "MSE od centroidu": round(mse, 4)
                        })

            df_mse = pd.DataFrame(wyniki_mse)
            if not df_mse.empty:
                prog_anomalii = df_mse["MSE od centroidu"].mean() + 2 * df_mse["MSE od centroidu"].std()
                df_mse["Anomalia"] = df_mse["MSE od centroidu"].apply(
                    lambda v: "🚨 TAK" if v > prog_anomalii else "✅ NIE"
                )
                df_mse_sorted = df_mse.sort_values("MSE od centroidu", ascending=False).reset_index(drop=True)

                col_mse1, col_mse2 = st.columns(2)
                with col_mse1:
                    st.markdown("##### 🚨 Anomalie (MSE > μ + 2σ):")
                    anomalie = df_mse_sorted[df_mse_sorted["Anomalia"] == "🚨 TAK"].reset_index(drop=True)
                    if not anomalie.empty:
                        st.dataframe(
                            anomalie[["Krzywa", "Klaster", "MSE od centroidu"]].style.format(
                                {"MSE od centroidu": "{:.4f}"}
                            ).background_gradient(subset=["MSE od centroidu"], cmap="Reds"),
                            hide_index=True, use_container_width=True
                        )
                        st.caption(f"Próg anomalii: {prog_anomalii:.4f}  |  Wykryto: {len(anomalie)} krzywych")
                    else:
                        st.success("Brak anomalii — wszystkie krzywe leżą blisko centroidów klastrów.")

                with col_mse2:
                    st.markdown("##### 📊 Ranking MSE wszystkich krzywych:")
                    st.dataframe(
                        df_mse_sorted[["Krzywa", "Klaster", "MSE od centroidu", "Anomalia"]].style.format(
                            {"MSE od centroidu": "{:.4f}"}
                        ),
                        hide_index=True, use_container_width=True, height=320
                    )

        # =================================================================
        # AUTOMATYCZNY RANKING METOD — silhouette_score + ARI + NMI
        # =================================================================
        st.write("---")
        st.subheader("Ranking Skuteczności Algorytmów")

        @st.cache_data(show_spinner=False)
        def oblicz_ranking(_dane, _etykiety_eksperta, _krzywe, lista_metod, liczba_grup, _metoda_glowna):
            """Cached: przelicza się tylko gdy zmienią się dane wejściowe lub parametry."""
            rekordy = []
            for m_nazwa in lista_metod:
                try:
                    pred = uruchom_silnik_klastrowania(m_nazwa, _dane, liczba_grup, liczba_grup, _df_sygnaly_raw=_krzywe)
                    unikalne = np.unique(pred[pred > 0]) if 0 in pred else np.unique(pred)
                    if len(unikalne) < 2:
                        raise ValueError("Za mało klastrów do silhouette")
                    m_ari = adjusted_rand_score(_etykiety_eksperta, pred) * 100
                    m_nmi = normalized_mutual_info_score(_etykiety_eksperta, pred) * 100
                    # silhouette tylko dla punktów nie będących szumem
                    maska_nie_szum = pred > 0
                    if maska_nie_szum.sum() >= 2 and len(np.unique(pred[maska_nie_szum])) >= 2:
                        m_sil = silhouette_score(_dane[maska_nie_szum], pred[maska_nie_szum]) * 100
                    else:
                        m_sil = silhouette_score(_dane, pred) * 100
                    rekordy.append({
                        "Algorytm AI": m_nazwa,
                        "ARI (%)": round(m_ari, 2),
                        "NMI (%)": round(m_nmi, 2),
                        "Silhouette (%)": round(m_sil, 2),
                        "Średnia (ARI+NMI+Sil) (%)": round((m_ari + m_nmi + m_sil) / 3, 2)
                    })
                except Exception:
                    pass
            return rekordy

        rekordy_rankingu = oblicz_ranking(
            dane_do_algorytmu, etykiety_eksperta, krzywe,
            lista_metod, liczba_grup, metoda
        )

        if len(rekordy_rankingu) > 0:
            df_leaderboard = pd.DataFrame(rekordy_rankingu).sort_values(
                by="Średnia (ARI+NMI+Sil) (%)", ascending=False
            ).reset_index(drop=True)
            df_leaderboard.index += 1
            st.table(df_leaderboard)
        else:
            st.info("Trwa inicjalizacja rankingu modeli...")

    except Exception as ob_blad:
        st.error(f"Błąd krytyczny podczas renderowania: {ob_blad}")

else:
    st.info("Aby rozpocząć, wgraj plik z dysku lub wklej link do Google Sheets powyżej.")
