# =====================================================================
# INTERAKTYWNY ANALIZATOR KRZYWYCH AI PRO — WERSJA 2 (refaktoryzacja)
# =====================================================================
# CHANGELOG względem wersji pierwotnej:
#
# [BŁĘDY]
#  1. Naprawiono martwy przycisk CSV w sidebarze (klucz "df_eksport_klastry"
#     nie był nigdy zapisywany; teraz przycisk dodawany jest po obliczeniach).
#  2. "Spectral + GMM" faktycznie używa embeddingu spektralnego przed GMM.
#  3. "Spectral + Hierarchiczna" używa embeddingu spektralnego zamiast
#     wierszy macierzy afiniczności.
#  4. SiecSOM to teraz prawdziwa mapa Kohonena: gaussowska funkcja
#     sąsiedztwa + malejący promień sigma (wcześniej: tylko BMU = online
#     k-means bez topologii).
#  5. Time Warping przepisany na spójne, monotoniczne odkształcenie osi
#     czasu (wcześniej mapowanie węzłów było niespójne).
#  6. Ranking przy "Augmentacji sygnału" porównuje z ekspertem tylko
#     oryginalne krzywe (wcześniej cichy wyjątek usuwał metody z rankingu).
#  7. Cache silnika klastrowania hashuje też surowe dane (usunięty
#     podkreślnik przy df_sygnaly_raw) — koniec ryzyka stęchłego cache'a.
#
# [METODOLOGIA]
#  8. ARI/NMI liczone TYLKO gdy istnieje realny Ground Truth (arkusz w
#     pliku). Domyślny sztywny podział y1–y43 stosowany wyłącznie, gdy
#     nazwy krzywych do niego pasują — z wyraźnym ostrzeżeniem w UI.
#     Bez GT metryki zewnętrzne są ukrywane zamiast liczone od fikcji.
#
# [WYDAJNOŚĆ]
#  9. Leave-One-Out uruchamiany dopiero po kliknięciu przycisku i cache'owany
#     (wcześniej ~43 pełne klasteryzacje przy KAŻDYM rerunie).
# 10. Klastrowanie konsensusowe zwektoryzowane: O(4·N²) w NumPy zamiast
#     potrójnej pętli w Pythonie.
# 11. Standaryzacja krzywych liczona raz i współdzielona (sugestia K,
#     MSE, LOO). Google Sheets pobierany jednym requestem (sheet_name=None).
#
# [PORZĄDKI]
# 12. Stałe konfiguracyjne zebrane w sekcji KONFIGURACJA.
# 13. Usunięte duplikaty (PLOTLY_KOLORY x2, importy metryk x2, martwy
#     import PyTorch), zmienna `górna` -> `gorna`.
# 14. Walidacja rozmiaru danych (zakres K ograniczony liczbą krzywych,
#     ochrona silhouette/elbow przy małych zbiorach).
# 15. Podgląd wczytanej tabeli, żeby heurystyka nagłówka była widoczna.
# 16. Tryb debug: pełny traceback zamiast jednolinijkowego błędu.
# =====================================================================

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
import streamlit.components.v1 as components

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

# --- Bezpieczne importy opcjonalnych zależności ---
try:
    import umap
    umap_dostepne = True
except ImportError:
    umap_dostepne = False

try:
    from tslearn.clustering import KShape
    from tslearn.utils import to_time_series_dataset
    tslearn_dostepne = True
except ImportError:
    tslearn_dostepne = False


# =====================================================================
# KONFIGURACJA — wszystkie "magiczne liczby" w jednym miejscu
# =====================================================================

KONFIG = {
    "ROLLING_WINDOW": 5,           # okno filtrowania szumów (rolling mean)
    "SOM_X": 5,                    # wymiary siatki Kohonena
    "SOM_Y": 5,
    "SOM_EPOKI": 50,
    "SOM_LR": 0.5,
    "UMAP_N_NEIGHBORS": 15,
    "UMAP_MIN_DIST": 0.05,
    "UMAP_MIN_DIST_PREPROC": 0.1,
    "K_MAX_SUGESTIA": 10,          # górna granica przeszukiwania K
    "PCA_KOMPONENTY": 3,
    "RANDOM_STATE": 42,
    "DEBUG": False,                # True = pełny traceback przy błędzie
}

PLOTLY_KOLORY = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

NAZWY_KOLOROW = [
    "Niebieski", "Pomarańczowy", "Zielony", "Czerwony", "Fioletowy",
    "Brązowy", "Różowy", "Szary", "Oliwkowy", "Jasnoniebieski",
]

st.set_page_config(page_title="Analizator Krzywych Pro AI", layout="wide")


# =====================================================================
# PRAWDZIWA SIEĆ SOM (SELF-ORGANIZING MAP) — mapa Kohonena
# z gaussowską funkcją sąsiedztwa i malejącym promieniem sigma
# =====================================================================

class SiecSOM:
    def __init__(self, x_size=5, y_size=5, input_dim=43,
                 lr=0.5, epochs=100, random_state=42):
        self.x_size = x_size
        self.y_size = y_size
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.random_state = random_state
        rng = np.random.default_rng(random_state)
        self.wagi = rng.random((x_size * y_size, input_dim))
        # Współrzędne neuronów na siatce 2D — potrzebne do sąsiedztwa
        gx, gy = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing="ij")
        self.pozycje = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
        self.sigma0 = max(x_size, y_size) / 2.0

    def fit_predict_features(self, X):
        for epoch in range(self.epochs):
            frac = epoch / max(self.epochs - 1, 1)
            biezacy_lr = self.lr * (1.0 - frac)
            # Promień sąsiedztwa maleje od sigma0 do ~1 neuronu
            sigma = max(self.sigma0 * (1.0 - frac), 0.5)
            for sample in X:
                bmu_idx = np.argmin(np.linalg.norm(self.wagi - sample, axis=1))
                # Odległość każdego neuronu od BMU NA SIATCE (topologia!)
                dist_grid = np.linalg.norm(
                    self.pozycje - self.pozycje[bmu_idx], axis=1
                )
                h = np.exp(-(dist_grid ** 2) / (2.0 * sigma ** 2))
                # Aktualizacja wszystkich neuronów ważona sąsiedztwem
                self.wagi += biezacy_lr * h[:, None] * (sample - self.wagi)
        # Cechy = wagi BMU + drobny deterministyczny jitter, żeby uniknąć
        # identycznych wektorów (zerowe odległości psują linkage)
        aktywowane = np.empty((X.shape[0], self.input_dim))
        for i, sample in enumerate(X):
            bmu_idx = np.argmin(np.linalg.norm(self.wagi - sample, axis=1))
            aktywowane[i] = self.wagi[bmu_idx]
        rng = np.random.default_rng(self.random_state)
        aktywowane += rng.normal(0, 1e-9, aktywowane.shape)
        return aktywowane


# =====================================================================
# SŁOWNIKI OPISÓW
# =====================================================================

OPISY_METOD = {
    "Hierarchiczna Aglomeracyjna (metoda Warda)": "Buduje drzewo powiązań od dołu do góry na podstawie minimalizacji przyrostu wariancji wewnątrzklastrowej. Doskonale radzi sobie ze zwartymi grupami.",
    "Filtrowanie szumów (Rolling Mean) + Hierarchiczna (metoda Warda)": "Liniowa transformacja wygładzająca. Algorytm najpierw aplikuje okno kroczącej średniej (rolling mean), usuwając szum pomiarowy wysokiej częstotliwości z serii czasowej, a następnie grupuje klastry metodą Warda.",
    "PCA + Hierarchiczna (metoda Warda)": "Hybryda redukująca szum. Wyciąga kluczowe składowe sygnału (PCA), odrzucając drobne fluktuacje laboratoryjne, a następnie aplikuje kryterium Warda.",
    "UMAP + Hierarchiczna (metoda Warda)": "Potężna fuzja nieliniowa. UMAP makroskopowo zagęszcza i zbliża do siebie pokrewne profile krzywych w przestrzeni topologicznej, pozwalając metodzie Warda na bezbłędne wycięcie klastrów.",
    "SOM + Hierarchiczna (metoda Warda)": "Wykorzystuje topologiczną mapę Kohonena (SOM, z gaussowskim sąsiedztwem) do kompresji krzywych, a następnie buduje drzewo aglomeracyjne metodą Warda na bazie wag neuronów BMU.",
    "Spectral + Hierarchiczna (metoda Warda)": "Rzutuje krzywe do nieliniowej przestrzeni spektralnej (wektory własne Laplasjanu grafu podobieństwa), po czym aplikuje hierarchiczne grupowanie Warda na embeddingu.",
    "K-means": "Dzieli przestrzeń cech na tzw. obszary Voronoia. Algorytm dąży do minimalizacji wariancji wewnątrzklastrowej.",
    "UMAP + HDBSCAN (Hybryda Gęstościowa)": "Dwustopniowa hybryda nowej generacji. Najpierw rzutuje sygnał do przestrzeni topologicznej nieliniowej 2D (UMAP), a algorytm gęstościowy (HDBSCAN) wycina z nich grupy kształtów.",
    "Spectral + GMM (Hybryda Spektralno-Probabilistyczna)": "Najpierw wyznacza embedding spektralny (dekompozycja wartości własnych grafu pokrewieństwa), a następnie dopasowuje do niego elastyczne chmury probabilistyczne rozkładu normalnego (GMM).",
    "SOM + K-means (Hybryda sekwencyjna)": "Pierwszy etap wykorzystuje sieć neuronową Kohonena (SOM) do kompresji sygnału na siatkę topologiczną. Drugi etap uruchamia algorytm K-means na wagach neuronów.",
    "Klastrowanie Konsensusowe (Ensemble Voting)": "Metoda komitetowa. Uruchamia równolegle K-Means, GMM, Spectral, Ward i buduje macierz współwystępowania. Ostateczny podział jest fuzją decyzji wszystkich modeli.",
    "NMF (Nieujemna Faktoryzacja Macierzy)": "Rozkłada macierz danych na iloczyn dwóch macierzy o elementach wyłącznie nieujemnych.",
    "GMM (Probabilistyczna)": "Modele Mieszanin Gaussowskich. Próbuje dopasować elastyczne rozkłady normalne, dając miękkie przypisanie probabilistyczne.",
    "BGMM (Bayesowski GMM)": "Rozszerzenie GMM o probabilistyczną wersję Bayesowską z procesem Dirichleta. Automatycznie wygasza niepotrzebne klastry.",
    "Hierarchiczna Korelacyjna (metoda średnich)": "Podejście hierarchiczne, które mierzy stopień współliniowości wykresów za pomocą odległości korelacyjnej (1 - r Pearsona).",
    "HDBSCAN (Gęstościowa - Auto K)": "Zaawansowane klastrowanie gęstościowe oparte na teorii grafów. Szuka obszarów o wysokiej kondensacji punktów.",
    "Spectral Clustering": "Wykorzystuje wartości własne (widmo) macierzy podobieństwa danych do redukcji wymiarowości przed właściwym podziałem.",
    "K-Shape (Kształt fali)": "Wyspecjalizowany algorytm stworzony ściśle do analizy serii czasowych, wykorzystujący znormalizowaną korelację wzajemną.",
}

OPISY_PREPROCESSING = {
    "Standardowa": "Polega na klasycznej standaryzacji (Z-score). Sprowadza wszystkie punkty pomiarowe krzywych do wspólnej skali statystycznej.",
    "Analiza trendu": "Wyznacza różnice skończone (pochodne pierwszego rzędu) pomiędzy sąsiednimi punktami wzdłuż osi X.",
    "UMAP (Redukcja topologiczna)": "Uniform Manifold Approximation and Projection. Zaawansowana, nieliniowa redukcja wymiarowości.",
    "FeatureExtraction": "Głęboka transformacja inżynierska 3D: Max, Pozycja X, Średnia, Std, Skośność, Kurtoza, harmoniczne FFT oraz wskaźniki DWT Haar.",
    "MinMaxScaler": "Dokonuje liniowej transformacji danych, przesuwając i skalując wartości każdej krzywej do przedziału od 0 do 1.",
    "Filtrowanie szumów": "Wykorzystuje algorytm kroczącego okna średniej (rolling window). Skutecznie odcina fluktuacje wysokiej częstotliwości.",
    "Augmentacja sygnału": "Data Augmentation dla sieci neuronowych. Sztucznie rozbudowuje zbiór danych przez generowanie zaszumionych wariantów każdej krzywej. Dostępne techniki: Jitter (szum Gaussowski), Time Warping (deformacja osi czasu), Amplitude Scaling (losowe skalowanie amplitudy), Window Slicing (losowe przycięcie okna) oraz Permutation (przestawienie segmentów).",
}

OPISY_AUGMENTACJI = {
    "Jitter": "Dodaje do każdej krzywej losowy szum Gaussowski. Symuluje szum pomiarowy — najprostsza i najszybsza technika augmentacji.",
    "Time Warping": "Monotonicznie rozciąga i ściska oś czasu przez interpolację na odkształconej siatce punktów. Symuluje zmienną prędkość procesu.",
    "Amplitude Scaling": "Mnoży amplitudę każdej krzywej przez losowy współczynnik bliski 1.0. Symuluje zmienność wzmocnienia sygnału.",
    "Window Slicing": "Wycina losowy fragment krzywej i rozciąga go z powrotem do oryginalnej długości. Uczy model rozpoznawania lokalnych wzorców.",
    "Permutation": "Dzieli krzywą na segmenty i losowo je przestawia. Testuje odporność modelu na zmiany kolejności fragmentów sygnału.",
}


# =====================================================================
# FUNKCJE POMOCNICZE — DANE WEJŚCIOWE
# =====================================================================

def inteligentne_pobranie_tabeli(df_raw):
    df_raw = df_raw.dropna(how="all", axis=0).dropna(how="all", axis=1)
    df_raw = df_raw.reset_index(drop=True)
    indeks_startu = 0
    for idx, row in df_raw.iterrows():
        if row.notna().sum() > 1:
            if idx + 1 < len(df_raw):
                nastepny_wiersz = df_raw.iloc[idx + 1]
                ile_liczb = pd.to_numeric(nastepny_wiersz, errors="coerce").notna().sum()
                if ile_liczb > 1:
                    indeks_startu = idx
                    break
    naglowki = df_raw.iloc[indeks_startu]
    df_czysty = df_raw.iloc[indeks_startu + 1:].copy()
    df_czysty.columns = naglowki
    df_czysty = df_czysty.reset_index(drop=True)
    df_czysty = df_czysty.apply(pd.to_numeric, errors="coerce")
    df_czysty = df_czysty.dropna(how="all", axis=1)
    df_czysty = df_czysty.dropna(subset=[df_czysty.columns[0]])
    return df_czysty.reset_index(drop=True)


@st.cache_data(show_spinner="Pobieram dane z Google Sheets...")
def pobierz_google_sheets(url_base):
    """Jeden request HTTP dla wszystkich arkuszy naraz."""
    return pd.read_excel(f"{url_base}/export?format=xlsx", sheet_name=None, header=None)


# =====================================================================
# AUGMENTACJA SYGNAŁU (poprawiony Time Warping)
# =====================================================================

def augmentuj_sygnal(krzywe_df, technika, sila, n_kopii, random_state=42):
    """
    Generuje n_kopii augmentowanych wariantów każdej krzywej i zwraca
    rozszerzony DataFrame wraz z etykietami źródłowymi.
    sila: float 0.0–1.0 — intensywność przekształcenia.
    Kolejność kolumn: NAJPIERW wszystkie oryginały, potem kopie.
    """
    rng = np.random.default_rng(random_state)
    wyniki = {}
    etykiety_zrodlowe = {}

    for col in krzywe_df.columns:
        wyniki[str(col)] = krzywe_df[col].values.copy()
        etykiety_zrodlowe[str(col)] = str(col)

    n_punktow = len(krzywe_df)
    t_orig = np.linspace(0, 1, n_punktow)

    for col in krzywe_df.columns:
        syg = krzywe_df[col].values.astype(float)

        for k in range(1, n_kopii + 1):
            nazwa_aug = f"{col}_aug{k}"

            if technika == "Jitter":
                szum = rng.normal(0, sila * np.std(syg), size=n_punktow)
                aug = syg + szum

            elif technika == "Time Warping":
                # Monotoniczne odkształcenie osi czasu:
                # węzły równomierne -> losowo przesunięte -> wymuszona
                # monotoniczność -> interpolacja mapy deformacji.
                n_wezlow = 6
                t_wezly = np.linspace(0, 1, n_wezlow)
                przesuniecia = rng.uniform(-sila * 0.3, sila * 0.3, n_wezlow)
                przesuniecia[0] = 0.0
                przesuniecia[-1] = 0.0
                t_zdeform = np.clip(t_wezly + przesuniecia, 0, 1)
                t_zdeform = np.maximum.accumulate(t_zdeform)  # monotonia
                t_zdeform[-1] = 1.0
                mapa_czasu = np.interp(t_orig, t_wezly, t_zdeform)
                aug = np.interp(mapa_czasu, t_orig, syg)

            elif technika == "Amplitude Scaling":
                wspolczynnik = rng.uniform(1.0 - sila * 0.5, 1.0 + sila * 0.5)
                aug = syg * wspolczynnik

            elif technika == "Window Slicing":
                min_dlugosc = max(int(n_punktow * (1.0 - sila * 0.4)), 3)
                if min_dlugosc >= n_punktow:
                    aug = syg.copy()
                else:
                    dlugosc_okna = int(rng.integers(min_dlugosc, n_punktow))
                    start = int(rng.integers(0, n_punktow - dlugosc_okna + 1))
                    wycinek = syg[start: start + dlugosc_okna]
                    aug = np.interp(
                        t_orig, np.linspace(0, 1, dlugosc_okna), wycinek
                    )

            elif technika == "Permutation":
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


# =====================================================================
# SILNIK KLASTROWANIA
# UWAGA: df_sygnaly_raw jest teraz częścią klucza cache'a (bez "_"),
# więc zmiana surowych danych zawsze unieważnia cache.
# =====================================================================

def _spectral_embedding(dane, n_komponentow, random_state=42):
    """Wspólny embedding spektralny dla hybryd Spectral+X."""
    n_comp = max(2, min(n_komponentow, dane.shape[0] - 2))
    emb = SpectralEmbedding(
        n_components=n_comp, affinity="rbf", random_state=random_state
    )
    return emb.fit_transform(dane)


@st.cache_data(show_spinner=False)
def uruchom_silnik_klastrowania(nazwa_metody, dane, k_grup, min_hdbscan=3,
                                df_sygnaly_raw=None):
    rs = KONFIG["RANDOM_STATE"]

    if nazwa_metody == "K-means":
        return KMeans(n_clusters=k_grup, random_state=rs, n_init=5).fit_predict(dane) + 1

    elif "Filtrowanie szumów (Rolling Mean) + Hierarchiczna" in nazwa_metody:
        if df_sygnaly_raw is not None:
            wygladzane = df_sygnaly_raw.rolling(
                window=KONFIG["ROLLING_WINDOW"], center=True, min_periods=1
            ).mean().T
            dane_ward = StandardScaler().fit_transform(wygladzane)
        else:
            dane_ward = dane
        return fcluster(linkage(dane_ward, method="ward"), t=k_grup, criterion="maxclust")

    elif "PCA + Hierarchiczna" in nazwa_metody:
        n_comp = min(KONFIG["PCA_KOMPONENTY"], dane.shape[1], dane.shape[0])
        komponenty_pca = PCA(n_components=n_comp, random_state=rs).fit_transform(dane)
        return fcluster(linkage(komponenty_pca, method="ward"), t=k_grup, criterion="maxclust")

    elif "UMAP + Hierarchiczna" in nazwa_metody and umap_dostepne:
        przestrzen_2d = umap.UMAP(
            n_neighbors=KONFIG["UMAP_N_NEIGHBORS"],
            min_dist=KONFIG["UMAP_MIN_DIST"], random_state=rs
        ).fit_transform(dane)
        return fcluster(linkage(przestrzen_2d, method="ward"), t=k_grup, criterion="maxclust")

    elif "SOM + Hierarchiczna" in nazwa_metody:
        model_som = SiecSOM(
            x_size=KONFIG["SOM_X"], y_size=KONFIG["SOM_Y"],
            input_dim=dane.shape[1], lr=KONFIG["SOM_LR"],
            epochs=KONFIG["SOM_EPOKI"], random_state=rs
        )
        cechy_som = model_som.fit_predict_features(dane)
        return fcluster(linkage(cechy_som, method="ward"), t=k_grup, criterion="maxclust")

    elif "Spectral + Hierarchiczna" in nazwa_metody:
        # POPRAWKA: Ward na embeddingu spektralnym, nie na macierzy afiniczności
        emb = _spectral_embedding(dane, k_grup, rs)
        return fcluster(linkage(emb, method="ward"), t=k_grup, criterion="maxclust")

    elif "UMAP + HDBSCAN" in nazwa_metody and umap_dostepne:
        baza_projekcji = StandardScaler().fit_transform(dane)
        przestrzen_2d = umap.UMAP(
            n_neighbors=KONFIG["UMAP_N_NEIGHBORS"],
            min_dist=KONFIG["UMAP_MIN_DIST"], random_state=rs
        ).fit_transform(baza_projekcji)
        raw_labels = HDBSCAN(min_cluster_size=min_hdbscan, min_samples=1).fit_predict(przestrzen_2d)
        return np.array([n + 1 if n >= 0 else 0 for n in raw_labels])

    elif "Spectral + GMM" in nazwa_metody:
        # POPRAWKA: GMM na embeddingu spektralnym (wcześniej Spectral
        # w ogóle nie był używany)
        emb = _spectral_embedding(dane, k_grup, rs)
        gmm = GaussianMixture(n_components=k_grup, random_state=rs, n_init=2)
        return gmm.fit_predict(emb) + 1

    elif "SOM + K-means" in nazwa_metody:
        model_som = SiecSOM(
            x_size=KONFIG["SOM_X"], y_size=KONFIG["SOM_Y"],
            input_dim=dane.shape[1], lr=KONFIG["SOM_LR"],
            epochs=KONFIG["SOM_EPOKI"], random_state=rs
        )
        cechy_som = model_som.fit_predict_features(dane)
        return KMeans(n_clusters=k_grup, random_state=rs, n_init=5).fit_predict(cechy_som) + 1

    elif "Konsensusowe" in nazwa_metody:
        N = dane.shape[0]
        matrix = np.zeros((N, N))
        p1 = KMeans(n_clusters=k_grup, random_state=rs, n_init=2).fit_predict(dane)
        p2 = GaussianMixture(n_components=k_grup, random_state=rs, n_init=1).fit_predict(dane)
        p3 = SpectralClustering(n_clusters=k_grup, random_state=rs,
                                assign_labels="discretize").fit_predict(dane)
        p4 = fcluster(linkage(dane, method="ward"), t=k_grup, criterion="maxclust") - 1
        # POPRAWKA WYDAJNOŚCI: wektoryzacja zamiast potrójnej pętli
        for p in [p1, p2, p3, p4]:
            p = np.asarray(p)
            matrix += (p[:, None] == p[None, :]).astype(float)
        # POPRAWKA: linkage wymaga skondensowanej macierzy odległości —
        # kwadratowa macierz byłaby zinterpretowana jak zwykłe obserwacje
        dyssymilarnosc = 1.0 - (matrix / 4.0)
        np.fill_diagonal(dyssymilarnosc, 0.0)
        link = linkage(squareform(dyssymilarnosc, checks=False), method="average")
        return fcluster(link, t=k_grup, criterion="maxclust")

    elif "NMF" in nazwa_metody:
        dane_nmf = MinMaxScaler().fit_transform(dane) if (dane < 0).any() else dane
        W = NMF(n_components=k_grup, init="nndsvd", random_state=rs,
                max_iter=200).fit_transform(dane_nmf)
        return np.argmax(W, axis=1) + 1

    elif nazwa_metody == "GMM (Probabilistyczna)":
        return GaussianMixture(n_components=k_grup, random_state=rs,
                               n_init=2).fit_predict(dane) + 1

    elif "BGMM" in nazwa_metody:
        return BayesianGaussianMixture(
            n_components=k_grup, covariance_type="diag",
            weight_concentration_prior=1e-3, random_state=rs, n_init=2
        ).fit_predict(dane) + 1

    elif "metoda Warda" in nazwa_metody:
        return fcluster(linkage(dane, method="ward"), t=k_grup, criterion="maxclust")

    elif "Korelacyjna" in nazwa_metody:
        return fcluster(linkage(dane, method="average", metric="correlation"),
                        t=k_grup, criterion="maxclust")

    elif nazwa_metody == "HDBSCAN (Gęstościowa - Auto K)":
        raw = HDBSCAN(min_cluster_size=min_hdbscan, min_samples=1).fit_predict(dane)
        return np.array([n + 1 if n >= 0 else 0 for n in raw])

    elif "Spectral Clustering" in nazwa_metody:
        return SpectralClustering(n_clusters=k_grup, random_state=rs,
                                  assign_labels="discretize").fit_predict(dane) + 1

    elif "K-Shape" in nazwa_metody and tslearn_dostepne:
        return KShape(n_clusters=k_grup, random_state=rs).fit_predict(
            to_time_series_dataset(dane)) + 1

    else:
        return fcluster(linkage(dane, method="ward"), t=k_grup, criterion="maxclust")


# =====================================================================
# CACHE'OWANE OBLICZENIA POMOCNICZE
# =====================================================================

@st.cache_data(show_spinner=False)
def oblicz_metryki_k(dane, k_min, k_max):
    inercje, silhouettes, db_scores, calinski = [], [], [], []
    zakres_k = list(range(k_min, k_max + 1))
    for k in zakres_k:
        km = KMeans(n_clusters=k, random_state=KONFIG["RANDOM_STATE"], n_init=5).fit(dane)
        labels = km.labels_
        inercje.append(km.inertia_)
        silhouettes.append(silhouette_score(dane, labels))
        db_scores.append(davies_bouldin_score(dane, labels))
        calinski.append(calinski_harabasz_score(dane, labels))
    return zakres_k, inercje, silhouettes, db_scores, calinski


@st.cache_data(show_spinner="Analiza Leave-One-Out w toku... (N pełnych klasteryzacji)")
def oblicz_leave_one_out(metoda, dane_loo, etykiety_gt, krzywe_raw,
                         nazwy, liczba_grup, ari_bazowe):
    """LOO cache'owane i uruchamiane wyłącznie na żądanie (przycisk)."""
    wyniki_loo = []
    N = dane_loo.shape[0]
    for odrzucona_idx in range(N):
        maska = np.ones(N, dtype=bool)
        maska[odrzucona_idx] = False
        dane_sub = dane_loo[maska]
        etykiety_sub = [etykiety_gt[i] for i in range(N) if maska[i]]
        if "Filtrowanie szumów (Rolling Mean) + Hierarchiczna" in metoda:
            krzywe_sub = krzywe_raw.iloc[:, maska]
        else:
            krzywe_sub = krzywe_raw
        pred_sub = uruchom_silnik_klastrowania(
            metoda, dane_sub, liczba_grup, liczba_grup, df_sygnaly_raw=krzywe_sub
        )
        sub_ari = adjusted_rand_score(etykiety_sub, pred_sub) * 100
        sub_nmi = normalized_mutual_info_score(etykiety_sub, pred_sub) * 100
        wyniki_loo.append({
            "Odrzucona Krzywa": str(nazwy[odrzucona_idx]),
            "ARI bez krzywej (%)": round(sub_ari, 2),
            "NMI bez krzywej (%)": round(sub_nmi, 2),
            "Wpływ na ARI": round(sub_ari - ari_bazowe, 2),
        })
    return wyniki_loo


@st.cache_data(show_spinner="Obliczam ranking...")
def oblicz_ranking(dane, etykiety, krzywe_raw, metody_tuple, k,
                   indeksy_oryginalow):
    """
    Ranking wszystkich metod. Przy augmentacji predykcje są przycinane
    do oryginalnych krzywych przed liczeniem ARI/NMI (POPRAWKA #6).
    etykiety=None -> tylko silhouette (brak Ground Truth).
    """
    rekordy = []
    bledy = []
    idx_oryg = list(indeksy_oryginalow) if indeksy_oryginalow is not None else None

    for m_nazwa in metody_tuple:
        try:
            pred = uruchom_silnik_klastrowania(
                m_nazwa, dane, k, k, df_sygnaly_raw=krzywe_raw
            )
            pred = np.asarray(pred)
            unikalne = np.unique(pred[pred > 0]) if 0 in pred else np.unique(pred)
            if len(unikalne) < 2:
                bledy.append(f"{m_nazwa}: mniej niż 2 klastry")
                continue

            rekord = {"Algorytm AI": m_nazwa}

            # Silhouette na pełnym zbiorze (z pominięciem szumu HDBSCAN)
            maska_ns = pred > 0
            if maska_ns.sum() >= 2 and len(np.unique(pred[maska_ns])) >= 2:
                m_sil = silhouette_score(dane[maska_ns], pred[maska_ns]) * 100
            else:
                m_sil = silhouette_score(dane, pred) * 100
            rekord["Silhouette (%)"] = round(m_sil, 2)

            if etykiety is not None:
                # POPRAWKA: przy augmentacji porównujemy tylko oryginały
                pred_gt = pred[idx_oryg] if idx_oryg is not None else pred
                m_ari = adjusted_rand_score(list(etykiety), pred_gt) * 100
                m_nmi = normalized_mutual_info_score(list(etykiety), pred_gt) * 100
                rekord["ARI (%)"] = round(m_ari, 2)
                rekord["NMI (%)"] = round(m_nmi, 2)
                rekord["Średnia (%)"] = round((m_ari + m_nmi + m_sil) / 3, 2)
            else:
                rekord["Średnia (%)"] = round(m_sil, 2)

            rekordy.append(rekord)
        except Exception as e:
            bledy.append(f"{m_nazwa}: {e}")
    return rekordy, bledy


# =====================================================================
# GŁÓWNY EKRAN APLIKACJI
# =====================================================================

st.title("📊 Interaktywny Analizator Krzywych AI Pro")
st.write("### Ustawienia analizy")

typ_zrodla = st.radio(
    "Wybierz źródło danych:",
    ["Plik Excel (.xlsx)", "Link do Google Sheets"],
    horizontal=True,
)

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
            elif len(excel_file.sheet_names) > 1:
                df_expert_raw = pd.read_excel(uploaded_file, sheet_name=1)
        except Exception:
            df_expert_raw = None
else:
    link_sheets = st.text_input(
        "Wklej link do Google Sheets:",
        placeholder="https://docs.google.com/spreadsheets/d/...",
    )
    if link_sheets and "docs.google.com/spreadsheets" in link_sheets:
        try:
            url_base = link_sheets.split("/edit")[0]
            # POPRAWKA: jeden request zamiast dwóch
            sheets_dict = pobierz_google_sheets(url_base)
            nazwy_arkuszy = list(sheets_dict.keys())
            df = inteligentne_pobranie_tabeli(sheets_dict[nazwy_arkuszy[0]])
            file_id = f"cloud_{link_sheets[-15:]}"
            if "Ground Truth" in sheets_dict:
                gt_raw = sheets_dict["Ground Truth"]
                gt_raw.columns = gt_raw.iloc[0]
                df_expert_raw = gt_raw.iloc[1:].reset_index(drop=True)
            elif len(nazwy_arkuszy) > 1:
                gt_raw = sheets_dict[nazwy_arkuszy[1]]
                gt_raw.columns = gt_raw.iloc[0]
                df_expert_raw = gt_raw.iloc[1:].reset_index(drop=True)
        except Exception:
            st.error("Nie udało się pobrać danych ze struktur Google Sheets.")

if df is None:
    st.info("Aby rozpocząć, wgraj plik z dysku lub wklej link do Google Sheets powyżej.")
    st.stop()

try:
    x = df.iloc[:, 0]
    krzywe = df.iloc[:, 1:]
    nazwy_krzywych = krzywe.columns.tolist()
    n_krzywych = len(nazwy_krzywych)

    if n_krzywych < 3:
        st.error("Za mało krzywych do analizy (wymagane minimum 3).")
        st.stop()

    # Podgląd wczytanych danych — heurystyka nagłówka przestaje być czarną skrzynką
    with st.expander("🗂️ Podgląd wczytanych danych", expanded=False):
        st.caption(
            f"Wykryto **{n_krzywych}** krzywych po **{len(df)}** punktów. "
            f"Pierwsza kolumna (`{df.columns[0]}`) traktowana jako oś X."
        )
        st.dataframe(df.head(8), use_container_width=True)

    # -----------------------------------------------------------------
    # LISTY METOD I PREPROCESSINGU
    # -----------------------------------------------------------------
    lista_metod = [
        "Hierarchiczna Aglomeracyjna (metoda Warda)",
        "Filtrowanie szumów (Rolling Mean) + Hierarchiczna (metoda Warda)",
        "PCA + Hierarchiczna (metoda Warda)",
        "SOM + Hierarchiczna (metoda Warda)",
        "Spectral + Hierarchiczna (metoda Warda)",
        "K-means",
        "Spectral + GMM (Hybryda Spektralno-Probabilistyczna)",
        "SOM + K-means (Hybryda sekwencyjna)",
        "Klastrowanie Konsensusowe (Ensemble Voting)",
        "NMF (Nieujemna Faktoryzacja Macierzy)",
        "GMM (Probabilistyczna)",
        "BGMM (Bayesowski GMM)",
        "Hierarchiczna Korelacyjna (metoda średnich)",
        "HDBSCAN (Gęstościowa - Auto K)",
        "Spectral Clustering",
    ]
    if umap_dostepne:
        lista_metod.insert(3, "UMAP + Hierarchiczna (metoda Warda)")
        lista_metod.insert(7, "UMAP + HDBSCAN (Hybryda Gęstościowa)")
    if tslearn_dostepne:
        lista_metod.append("K-Shape (Kształt fali)")

    lista_preprocessingow = ["Standardowa", "Analiza trendu"]
    if umap_dostepne:
        lista_preprocessingow.append("UMAP (Redukcja topologiczna)")
    lista_preprocessingow.extend(
        ["FeatureExtraction", "MinMaxScaler", "Filtrowanie szumów", "Augmentacja sygnału"]
    )

    if ("wybrana_metoda" not in st.session_state
            or st.session_state.wybrana_metoda not in lista_metod):
        st.session_state.wybrana_metoda = lista_metod[0]

    # Zakres K ograniczony realną liczbą krzywych (POPRAWKA #14)
    max_k = min(KONFIG["K_MAX_SUGESTIA"], n_krzywych - 1)

    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        metoda = st.selectbox("Wybierz metodę główną:", lista_metod, key="wybrana_metoda")
    with col_param2:
        if ("K-Shape" in metoda or "UMAP + HDBSCAN" in metoda
                or "UMAP + Hierarchiczna" in metoda):
            optymalizacja = "Standardowa"
            st.caption("Ta metoda używa standardowego przygotowania danych.")
        else:
            optymalizacja = st.selectbox(
                "Wybierz wstępne przygotowanie danych:", lista_preprocessingow
            )
    with col_param3:
        if "HDBSCAN" in metoda:
            slider_label = "Minimalna wielkość grupy (HDBSCAN):"
        elif "BGMM" in metoda:
            slider_label = "Maksymalna liczba grup (BGMM):"
        else:
            slider_label = "Liczba grup (K):"
        liczba_grup = st.slider(slider_label, min_value=2, max_value=max_k,
                                value=min(5, max_k))

    st.write("---")

    # -----------------------------------------------------------------
    # PANEL AUGMENTACJI
    # -----------------------------------------------------------------
    aug_technika = "Jitter"
    aug_sila = 0.1
    aug_kopie = 2

    if optymalizacja == "Augmentacja sygnału":
        with st.expander("⚙️ Ustawienia Augmentacji Sygnału", expanded=False):
            col_aug1, col_aug2, col_aug3 = st.columns(3)
            with col_aug1:
                aug_technika = st.selectbox(
                    "Technika augmentacji:",
                    ["Jitter", "Time Warping", "Amplitude Scaling",
                     "Window Slicing", "Permutation"],
                    help="Wybierz metodę przekształcania krzywych",
                )
                st.caption(OPISY_AUGMENTACJI.get(aug_technika, ""))
            with col_aug2:
                aug_sila = st.slider(
                    "Siła augmentacji:",
                    min_value=0.01, max_value=1.0, value=0.1, step=0.01,
                    help="Im wyższa wartość, tym większe zniekształcenie sygnału",
                )
            with col_aug3:
                aug_kopie = st.slider(
                    "Liczba kopii na krzywą:",
                    min_value=1, max_value=10, value=2,
                    help="Ile augmentowanych wariantów wygenerować dla każdej krzywej",
                )
            st.info(
                f"Zbiór zostanie rozszerzony z **{n_krzywych}** do "
                f"**{n_krzywych * (1 + aug_kopie)}** krzywych "
                f"({aug_kopie} kopii × {n_krzywych} oryginałów + oryginały). "
                f"Klasteryzacja działa na pełnym zbiorze, ARI/NMI liczone tylko dla oryginałów."
            )

    # =================================================================
    # GROUND TRUTH — uczciwa obsługa (POPRAWKA #8)
    # ARI/NMI liczone tylko gdy istnieje realny podział ekspercki.
    # =================================================================
    expert_mapping = {}
    if df_expert_raw is not None and len(df_expert_raw) > 0:
        try:
            df_expert_raw.columns = [str(c).strip().lower() for c in df_expert_raw.columns]
            col_k = df_expert_raw.columns[0]
            col_g = df_expert_raw.columns[1]
            for _, row in df_expert_raw.iterrows():
                k_str = str(row[col_k]).strip().lower()
                v_str = str(row[col_g]).strip()
                if k_str and k_str != "nan" and v_str and v_str != "nan":
                    expert_mapping[k_str] = v_str
        except Exception:
            expert_mapping = {}

    # Domyślny (historyczny) podział y1–y43 — używany TYLKO gdy nazwy pasują
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

    def znajdz_gt(nazwa, mapa):
        n_clean = str(nazwa).strip().lower()
        n_alt = f"y{n_clean}" if (not n_clean.startswith("y") and n_clean.isdigit()) else n_clean
        if n_clean in mapa:
            return mapa[n_clean]
        if n_alt in mapa:
            return mapa[n_alt]
        return None

    expert_list = []
    zrodlo_gt = None  # "plik" | "domyslny" | None

    if expert_mapping:
        trafienia = [znajdz_gt(n, expert_mapping) for n in nazwy_krzywych]
        if all(t is not None for t in trafienia):
            expert_list = trafienia
            zrodlo_gt = "plik"
    if zrodlo_gt is None:
        trafienia = [znajdz_gt(n, sztywny_podzial_eksperta) for n in nazwy_krzywych]
        if all(t is not None for t in trafienia):
            expert_list = trafienia
            zrodlo_gt = "domyslny"

    gt_dostepny = zrodlo_gt is not None

    # -----------------------------------------------------------------
    # SIDEBAR — edytor Grup Wzorcowych (tylko gdy GT dostępny)
    # -----------------------------------------------------------------
    etykiety_eksperta = None
    if gt_dostepny:
        df_current_gt = pd.DataFrame({
            "Krzywa": [str(n) for n in nazwy_krzywych],
            "Grupa Eksperta": expert_list,
        })

        if ("last_file_id" not in st.session_state
                or st.session_state.last_file_id != file_id):
            st.session_state.last_file_id = file_id
            st.session_state["tabela_editor_state"] = df_current_gt

        with st.sidebar:
            st.markdown("### 📋 Grupy Wzorcowe")
            if zrodlo_gt == "domyslny":
                st.warning(
                    "⚠️ Brak arkusza *Ground Truth* — użyto **domyślnego** "
                    "podziału y1–y43. Zweryfikuj przed interpretacją ARI/NMI!"
                )
            st.caption("Zmiana grupy natychmiast przelicza ARI, NMI i anomalie.")
            edited_gt = st.data_editor(
                st.session_state["tabela_editor_state"],
                hide_index=True,
                use_container_width=True,
                disabled=["Krzywa"],
                key=f"sidebar_editor_{file_id}",
                column_config={
                    "Krzywa": st.column_config.TextColumn("Krzywa", disabled=True),
                    "Grupa Eksperta": st.column_config.TextColumn(
                        "Grupa", help="Wpisz nazwę grupy (a, b, c...)", max_chars=20
                    ),
                },
            )
            st.session_state["tabela_editor_state"] = edited_gt

        etykiety_eksperta = edited_gt["Grupa Eksperta"].astype(str).tolist()
    else:
        with st.sidebar:
            st.markdown("### 📋 Grupy Wzorcowe")
            st.info(
                "Brak podziału eksperckiego (arkusz *Ground Truth* z kolumnami "
                "Krzywa | Grupa). Metryki ARI/NMI są wyłączone — dostępne "
                "pozostają metryki wewnętrzne (Silhouette, Davies-Bouldin...)."
            )

    # -----------------------------------------------------------------
    # Stylizacja sidebara + tooltip (best-effort: hack na data-testid,
    # może przestać działać po aktualizacji Streamlita — degraduje bez błędu)
    # -----------------------------------------------------------------
    st.markdown("""
    <style>
    [data-testid="collapsedControl"] {
        background: #1f77b4 !important;
        border-radius: 0 8px 8px 0 !important;
        width: 34px !important;
        height: auto !important;
        min-height: 80px !important;
        cursor: pointer !important;
        overflow: visible !important;
    }
    [data-testid="collapsedControl"] svg { display: none !important; }
    [data-testid="collapsedControl"]::after {
        content: 'Grupy Wzorcowe';
        color: white;
        font-size: 11px;
        font-weight: bold;
        writing-mode: vertical-rl;
        letter-spacing: 1px;
        padding: 10px 4px;
        display: block;
        text-align: center;
    }
    [data-testid="stSelectbox"] > div,
    div[data-baseweb="select"] > div,
    [data-testid="stSlider"] input[type="range"],
    [data-testid="stRadio"] label,
    button, select, [role="button"],
    [data-baseweb="select"] { cursor: pointer !important; }
    #sidebar-tooltip {
        position: fixed;
        background: #1a1a2e;
        color: #fff;
        padding: 7px 12px;
        border-radius: 7px;
        font-size: 13px;
        font-family: sans-serif;
        white-space: nowrap;
        pointer-events: none;
        z-index: 9999999;
        opacity: 0;
        transition: opacity 0.18s ease;
        box-shadow: 0 3px 12px rgba(0,0,0,0.35);
    }
    #sidebar-tooltip::before {
        content: '';
        position: absolute;
        left: -6px; top: 50%;
        transform: translateY(-50%);
        border: 6px solid transparent;
        border-right-color: #1a1a2e;
        border-left: 0;
    }
    [data-testid="stAppViewContainer"] > [data-testid="stMain"] {
        transition: margin-left 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    components.html("""
    <script>
    (function() {
        try {
            var doc = window.parent.document;
            var old = doc.getElementById('sidebar-tooltip');
            if (old) old.remove();
            var tip = doc.createElement('div');
            tip.id = 'sidebar-tooltip';
            tip.textContent = '📋 Otwórz panel Grupy Wzorcowe';
            doc.body.appendChild(tip);
            function showTooltip(e) {
                var rect = e.currentTarget.getBoundingClientRect();
                tip.style.left = (rect.right + 12) + 'px';
                tip.style.top = (rect.top + rect.height / 2 - 16) + 'px';
                tip.style.opacity = '1';
            }
            function hideTooltip() { tip.style.opacity = '0'; }
            function attachTooltip() {
                var btn = doc.querySelector('[data-testid="collapsedControl"]');
                if (!btn) { setTimeout(attachTooltip, 300); return; }
                btn.removeEventListener('mouseenter', showTooltip);
                btn.removeEventListener('mouseleave', hideTooltip);
                btn.addEventListener('mouseenter', showTooltip);
                btn.addEventListener('mouseleave', hideTooltip);
            }
            attachTooltip();
        } catch (err) { /* sandbox bez dostępu do parenta — cicha degradacja */ }
    })();
    </script>
    """, height=0)

    # =================================================================
    # WSPÓŁDZIELONA STANDARYZACJA (liczona raz — POPRAWKA #11)
    # =================================================================
    dane_std_oryginaly = StandardScaler().fit_transform(krzywe.T)

    # =================================================================
    # SUGESTIA LICZBY KLASTRÓW (K)
    # =================================================================
    with st.expander("📐 Sugestia Optymalnej Liczby Klastrów (K)", expanded=False):
        st.markdown(
            "Wykresy pomagają dobrać właściwą liczbę grup **K** przed uruchomieniem "
            "klasteryzacji. Każda metoda patrzy na problem z innej strony."
        )
        if max_k < 3:
            st.warning("Za mało krzywych, aby sensownie porównać różne wartości K.")
        else:
            zakres_k, inercje, silhouettes, db_scores, calinski = oblicz_metryki_k(
                dane_std_oryginaly, 2, max_k
            )

            # Elbow: druga pochodna inercji (guard na krótkie zakresy)
            if len(inercje) >= 3:
                diff2 = np.diff(np.diff(inercje))
                k_elbow = zakres_k[int(np.argmax(diff2)) + 1]
            else:
                k_elbow = zakres_k[0]
            k_silhouette = zakres_k[int(np.argmax(silhouettes))]
            k_db = zakres_k[int(np.argmin(db_scores))]
            k_calinski = zakres_k[int(np.argmax(calinski))]

            # Kneedle — punkt maksymalnej odległości od prostej łączącej końce
            inercje_arr = np.array(inercje, dtype=float)
            x_norm = (np.array(zakres_k) - zakres_k[0]) / max(zakres_k[-1] - zakres_k[0], 1)
            y_norm = (inercje_arr - inercje_arr.min()) / (inercje_arr.max() - inercje_arr.min() + 1e-12)
            x0, y0 = x_norm[0], y_norm[0]
            x1, y1 = x_norm[-1], y_norm[-1]
            dx, dy = x1 - x0, y1 - y0
            odleglosci = np.abs(dy * x_norm - dx * y_norm + x1 * y0 - y1 * x0) / (np.sqrt(dx**2 + dy**2) + 1e-12)
            k_kneedle = zakres_k[int(np.argmax(odleglosci))]

            st.info(
                f"**Sugerowane K:** "
                f"Elbow (2. pochodna) \u2192 **{k_elbow}** | "
                f"Kneedle \u2192 **{k_kneedle}** | "
                f"Silhouette \u2192 **{k_silhouette}** | "
                f"Davies-Bouldin \u2192 **{k_db}** | "
                f"Calinski-Harabasz \u2192 **{k_calinski}**"
            )

            fig_k = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    "Elbow (Inercja) \u2014 szukaj zgi\u0119cia",
                    "Kneedle \u2014 max odleg\u0142o\u015b\u0107 od prostej",
                    "Silhouette \u2014 im wy\u017cszy tym lepiej",
                    "Davies-Bouldin \u2014 im ni\u017cszy tym lepiej",
                    "Calinski-Harabasz \u2014 im wy\u017cszy tym lepiej",
                    "",
                ],
            )
            kolor_marker = "#d62728"

            def _trace(xk, yk, k_opt, kolor, row, col):
                fig_k.add_trace(go.Scatter(
                    x=xk, y=yk, mode="lines+markers",
                    line=dict(color=kolor, width=2),
                    marker=dict(
                        color=[kolor_marker if k == k_opt else kolor for k in xk],
                        size=[11 if k == k_opt else 7 for k in xk],
                    ),
                    hovertemplate="K=%{x}<br>=%{y:.4g}<extra></extra>",
                ), row=row, col=col)

            _trace(zakres_k, inercje,     k_elbow,      "#1f77b4", 1, 1)
            _trace(zakres_k, list(odleglosci), k_kneedle, "#8c564b", 1, 2)
            _trace(zakres_k, silhouettes, k_silhouette, "#2ca02c", 1, 3)
            _trace(zakres_k, db_scores,   k_db,         "#ff7f0e", 2, 1)
            _trace(zakres_k, calinski,    k_calinski,   "#9467bd", 2, 2)

            fig_k.update_layout(height=480, showlegend=False,
                                margin=dict(l=10, r=10, t=40, b=10))
            fig_k.update_xaxes(title_text="K", dtick=1)
            st.plotly_chart(fig_k, use_container_width=True)
            st.caption(
                "\U0001f534 Czerwony punkt = sugerowane K. "
                "Kneedle to geometryczna metoda wyznaczania 'kolana' krzywej inercji \u2014 "
                "szuka punktu o maksymalnej odleg\u0142o\u015bci od prostej \u0142\u0105cz\u0105cej kra\u0144ce."
            )

    with st.expander("Kompleksowy Opis Metodologiczny", expanded=False):
        c_d1, c_d2 = st.columns(2)
        with c_d1:
            st.markdown(f"#### Algorytm Główny: `{metoda}`")
            st.write(OPISY_METOD.get(metoda, ""))
        with c_d2:
            st.markdown(f"#### Obróbka Wstępna: `{optymalizacja}`")
            st.write(OPISY_PREPROCESSING.get(optymalizacja, ""))

    # =================================================================
    # PRZETWARZANIE DANYCH WEJŚCIOWYCH
    # =================================================================
    indeksy_oryginalow = None
    krzywe_aug = None

    if optymalizacja == "Augmentacja sygnału":
        krzywe_aug, etykiety_zrodlowe = augmentuj_sygnal(
            krzywe, aug_technika, aug_sila, aug_kopie,
            random_state=KONFIG["RANDOM_STATE"]
        )
        dane_do_algorytmu = StandardScaler().fit_transform(krzywe_aug.T)
        indeksy_oryginalow = list(range(n_krzywych))
    elif optymalizacja == "Analiza trendu":
        dane_do_algorytmu = StandardScaler().fit_transform(
            krzywe.diff(axis=0).fillna(0).T
        )
    elif optymalizacja == "UMAP (Redukcja topologiczna)" and umap_dostepne:
        dane_do_algorytmu = umap.UMAP(
            n_neighbors=KONFIG["UMAP_N_NEIGHBORS"],
            min_dist=KONFIG["UMAP_MIN_DIST_PREPROC"],
            random_state=KONFIG["RANDOM_STATE"],
        ).fit_transform(dane_std_oryginaly)
    elif optymalizacja == "FeatureExtraction":
        cechy = pd.DataFrame(index=nazwy_krzywych)
        cechy["Max"] = krzywe.max().values
        # Indeks df jest zresetowany (pozycyjny) — idxmax odpowiada pozycjom w x
        cechy["Poz_Max"] = krzywe.idxmax().apply(lambda idx: x.iloc[int(idx)]).values
        cechy["Srednia"] = krzywe.mean().values
        cechy["Std"] = krzywe.std().values
        cechy["Skosnosc"] = krzywe.skew().values
        cechy["Kurtoza"] = krzywe.kurt().values
        fft_amplitudy = np.abs(np.fft.rfft(krzywe, axis=0))
        for f_idx in range(1, min(4, fft_amplitudy.shape[0])):
            cechy[f"FFT_Skladowa_{f_idx}"] = fft_amplitudy[f_idx, :]
        dwt_a_mean, dwt_d_energy, dwt_d_std = [], [], []
        for col in krzywe.columns:
            sig = krzywe[col].values
            if len(sig) % 2 != 0:
                sig = sig[:-1]
            approx = (sig[0::2] + sig[1::2]) / np.sqrt(2)
            detail = (sig[0::2] - sig[1::2]) / np.sqrt(2)
            dwt_a_mean.append(np.mean(approx))
            dwt_d_energy.append(np.sum(detail ** 2))
            dwt_d_std.append(np.std(detail))
        cechy["DWT_Haar_A_Srednia"] = dwt_a_mean
        cechy["DWT_Haar_D_Energia"] = dwt_d_energy
        cechy["DWT_Haar_D_Std"] = dwt_d_std
        dane_do_algorytmu = StandardScaler().fit_transform(cechy)
    elif optymalizacja == "MinMaxScaler":
        dane_do_algorytmu = MinMaxScaler().fit_transform(krzywe.T)
    elif optymalizacja == "Filtrowanie szumów":
        dane_do_algorytmu = StandardScaler().fit_transform(
            krzywe.rolling(window=KONFIG["ROLLING_WINDOW"],
                           center=True, min_periods=1).mean().T
        )
    else:
        dane_do_algorytmu = dane_std_oryginaly

    # =================================================================
    # KLASTERYZACJA + METRYKI
    # =================================================================
    if optymalizacja == "Augmentacja sygnału":
        numery_grup = uruchom_silnik_klastrowania(
            metoda, dane_do_algorytmu, liczba_grup, liczba_grup,
            df_sygnaly_raw=krzywe_aug,
        )
        numery_grup = np.asarray(numery_grup)
        numery_grup_oryg = numery_grup[indeksy_oryginalow]
        krzywe_do_wykresu = krzywe
        nazwy_do_wykresu = nazwy_krzywych
        numery_grup_do_wykresu = numery_grup_oryg
    else:
        numery_grup = np.asarray(uruchom_silnik_klastrowania(
            metoda, dane_do_algorytmu, liczba_grup, liczba_grup,
            df_sygnaly_raw=krzywe,
        ))
        krzywe_do_wykresu = krzywe
        nazwy_do_wykresu = nazwy_krzywych
        numery_grup_do_wykresu = numery_grup

    ari_score = None
    nmi_score = None
    if gt_dostepny:
        ari_score = adjusted_rand_score(etykiety_eksperta, numery_grup_do_wykresu) * 100
        nmi_score = normalized_mutual_info_score(etykiety_eksperta, numery_grup_do_wykresu) * 100

        st.markdown("### Skuteczność dopasowania:")
        kpi_ari, kpi_nmi = st.columns(2)
        kpi_ari.metric("Indeks ARI", f"{ari_score:.1f}%")
        kpi_nmi.metric("Indeks NMI", f"{nmi_score:.1f}%")
        if zrodlo_gt == "domyslny":
            st.caption(
                "⚠️ ARI/NMI liczone względem **domyślnego** podziału y1–y43 "
                "(brak arkusza Ground Truth w pliku)."
            )
    else:
        st.markdown("### Skuteczność dopasowania:")
        st.warning(
            "Brak Ground Truth — metryki zewnętrzne (ARI/NMI) niedostępne. "
            "Dodaj do pliku arkusz **Ground Truth** (kolumny: Krzywa | Grupa), "
            "aby je odblokować."
        )

    # =================================================================
    # WYKRES 1: WSZYSTKIE KRZYWE
    # =================================================================
    st.subheader("Wykres 1: Wszystkie sklasterowane krzywe")

    czy_dendrogram = "Hierarchiczna" in metoda and "+" not in metoda

    if czy_dendrogram:
        fig_dend, ax_dend = plt.subplots(figsize=(10, 4.2))
        if optymalizacja == "Augmentacja sygnału":
            etykiety_dendro = list(krzywe_aug.columns)
        else:
            etykiety_dendro = [str(n) for n in nazwy_do_wykresu]
        dendrogram(
            linkage(dane_do_algorytmu,
                    method="ward" if "Warda" in metoda else "average"),
            labels=etykiety_dendro, leaf_rotation=90, ax=ax_dend,
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
                hovertemplate=(
                    f"<b>{col}</b><br>Klaster: "
                    f"{klaster_id if klaster_id > 0 else 'Szum'}"
                    "<br>X: %{x}<br>Y: %{y:.4f}<extra></extra>"
                ),
            ))
            dodane_do_legendy.add(klaster_id)
        fig1.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(groupclick="toggleitem",
                        bgcolor="rgba(255,255,255,0.8)", borderwidth=1),
            xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
            yaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
            hovermode="closest",
        )
        st.plotly_chart(fig1, use_container_width=True)

    # =================================================================
    # WYKRES 2: PROFILE MODELOWE (ŚREDNIE + CIEŃ WARIANCJI)
    # =================================================================
    if not czy_dendrogram:
        st.subheader("Wykres 2: Uśrednione profile modelowe (Wzorce kształtu fali)")

        fig2 = go.Figure()
        unikalne_klastry = sorted(set(int(v) for v in numery_grup_do_wykresu))

        for k_id in unikalne_klastry:
            maska_klastra = [int(numery_grup_do_wykresu[idx]) == k_id
                             for idx in range(len(numery_grup_do_wykresu))]
            krzywe_klastra = krzywe_do_wykresu.iloc[:, maska_klastra]
            if krzywe_klastra.shape[1] == 0:
                continue

            profil_sredni = krzywe_klastra.mean(axis=1)
            profil_std = krzywe_klastra.std(axis=1).fillna(0)
            gorna = profil_sredni + profil_std
            dolna = profil_sredni - profil_std

            if k_id > 0:
                kolor = PLOTLY_KOLORY[(k_id - 1) % 10]
                label_sredni = f"Wzorzec Klastra {k_id}"
            else:
                kolor = "#aaaaaa"
                label_sredni = "Średnia Szumu"
            liczba_krzywych_kl = krzywe_klastra.shape[1]

            fig2.add_trace(go.Scatter(
                x=list(x) + list(x[::-1]),
                y=list(gorna) + list(dolna[::-1]),
                fill="toself",
                fillcolor=kolor,
                opacity=0.12,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=label_sredni,
            ))
            fig2.add_trace(go.Scatter(
                x=x,
                y=profil_sredni,
                mode="lines",
                name=label_sredni,
                legendgroup=label_sredni,
                line=dict(color=kolor, width=2.5),
                hovertemplate=(
                    f"<b>{label_sredni}</b><br>"
                    f"Liczba krzywych: {liczba_krzywych_kl}<br>"
                    "X: %{x}<br>Średnia: %{y:.4f}<extra></extra>"
                ),
            ))

        fig2.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=1),
            xaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
            yaxis=dict(showgrid=True, gridcolor="#e0e0e0"),
            hovermode="closest",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # =================================================================
    # SKŁAD KLASTRÓW + EKSPORT
    # =================================================================
    if not czy_dendrogram:
        klastry_slownik = {}
        for i, col in enumerate(krzywe_do_wykresu.columns):
            k_id = int(numery_grup_do_wykresu[i])
            klastry_slownik.setdefault(k_id, []).append(str(col))

        posortowane_klastry = sorted(klastry_slownik.keys())
        liczba_klastrow = len(posortowane_klastry)

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
                    "Nr Klastra": k_id,
                })
        df_eksport_klastry = pd.DataFrame(wiersze_eksportu)

        # POPRAWKA #1: przycisk w sidebarze dodawany PO obliczeniach,
        # w tym samym rerunie — bez opóźnienia i bez martwego klucza
        with st.sidebar:
            st.markdown("---")
            st.caption("💾 Pobierz skład klastrów:")
            csv_bytes_sb = df_eksport_klastry.to_csv(
                index=False, encoding="utf-8-sig"
            ).encode("utf-8-sig")
            st.download_button(
                "⬇️ Pobierz CSV",
                data=csv_bytes_sb,
                file_name=f"klastry_{metoda[:15].replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with st.expander("📊 Szczegółowy skład wygenerowanych klastrów", expanded=False):
            if liczba_klastrow > 0:
                kolumny_klastrow = st.columns(min(liczba_klastrow, 4))
                for idx, k_id in enumerate(posortowane_klastry):
                    col_ui = kolumny_klastrow[idx % 4]
                    with col_ui:
                        if k_id == 0:
                            st.markdown("**⚪ Szum / Odrzuty**")
                        else:
                            n_koloru = NAZWY_KOLOROW[(k_id - 1) % 10]
                            st.markdown(f"**🔹 Klaster {k_id}** ({n_koloru})")
                        st.caption(f"Liczba: {len(klastry_slownik[k_id])}")
                        st.code(", ".join(klastry_slownik[k_id]), language="text")

            st.markdown("---")
            st.markdown("##### 💾 Pobierz wygenerowany skład klastrów:")
            col_exp_csv, col_exp_xlsx = st.columns(2)
            with col_exp_csv:
                csv_bytes = df_eksport_klastry.to_csv(
                    index=False, encoding="utf-8-sig"
                ).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ Pobierz CSV",
                    data=csv_bytes,
                    file_name=f"sklady_klastrow_{metoda[:20].replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_exp_xlsx:
                bufor_xlsx = io.BytesIO()
                with pd.ExcelWriter(bufor_xlsx, engine="openpyxl") as writer:
                    df_eksport_klastry.to_excel(writer, index=False,
                                                sheet_name="Skład Klastrów")
                    arkusz = writer.sheets["Skład Klastrów"]
                    arkusz.column_dimensions["A"].width = 20
                    arkusz.column_dimensions["B"].width = 20
                    arkusz.column_dimensions["C"].width = 18
                    arkusz.column_dimensions["D"].width = 12
                bufor_xlsx.seek(0)
                st.download_button(
                    label="⬇️ Pobierz Excel",
                    data=bufor_xlsx.getvalue(),
                    file_name=f"sklady_klastrow_{metoda[:20].replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

    # =================================================================
    # MSE ANOMALY DETECTION
    # =================================================================
    with st.expander("🔍 Detekcja Anomalii MSE: Odległość od centroidu klastra",
                     expanded=False):
        st.markdown(
            "Dla każdej krzywej obliczana jest **fizyczna odległość MSE** od wzorca "
            "(centroidu) jej klastra. Krzywe z MSE powyżej progu `μ + 2σ` są "
            "automatycznie oznaczane jako anomalie."
        )

        wyniki_mse = []
        unikalne_k = sorted(set(int(v) for v in numery_grup_do_wykresu))
        for k_id in unikalne_k:
            if k_id == 0:
                continue  # pomijamy szum HDBSCAN
            maska_k = np.array([int(v) for v in numery_grup_do_wykresu]) == k_id
            dane_klastra = dane_std_oryginaly[maska_k]
            centroid = dane_klastra.mean(axis=0)
            for i, (nalezy, col) in enumerate(zip(maska_k, krzywe_do_wykresu.columns)):
                if nalezy:
                    mse = float(np.mean((dane_std_oryginaly[i] - centroid) ** 2))
                    wyniki_mse.append({
                        "Krzywa": str(col),
                        "Klaster": k_id,
                        "MSE od centroidu": round(mse, 4),
                    })

        df_mse = pd.DataFrame(wyniki_mse)
        if not df_mse.empty:
            prog_anomalii = (df_mse["MSE od centroidu"].mean()
                             + 2 * df_mse["MSE od centroidu"].std())
            df_mse["Anomalia"] = df_mse["MSE od centroidu"].apply(
                lambda v: "🚨 TAK" if v > prog_anomalii else "✅ NIE"
            )
            df_mse_sorted = df_mse.sort_values(
                "MSE od centroidu", ascending=False
            ).reset_index(drop=True)

            col_mse1, col_mse2 = st.columns(2)
            with col_mse1:
                st.markdown("##### 🚨 Anomalie (MSE > μ + 2σ):")
                anomalie = df_mse_sorted[
                    df_mse_sorted["Anomalia"] == "🚨 TAK"
                ].reset_index(drop=True)
                if not anomalie.empty:
                    st.dataframe(
                        anomalie[["Krzywa", "Klaster", "MSE od centroidu"]]
                        .style.format({"MSE od centroidu": "{:.4f}"})
                        .background_gradient(subset=["MSE od centroidu"], cmap="Reds"),
                        hide_index=True, use_container_width=True,
                    )
                    st.caption(
                        f"Próg anomalii: {prog_anomalii:.4f}  |  "
                        f"Wykryto: {len(anomalie)} krzywych"
                    )
                else:
                    st.success("Brak anomalii — wszystkie krzywe leżą blisko "
                               "centroidów klastrów.")
            with col_mse2:
                st.markdown("##### 📊 Ranking MSE wszystkich krzywych:")
                st.dataframe(
                    df_mse_sorted[["Krzywa", "Klaster", "MSE od centroidu", "Anomalia"]]
                    .style.format({"MSE od centroidu": "{:.4f}"}),
                    hide_index=True, use_container_width=True, height=320,
                )
        else:
            st.info("Brak klastrów do analizy (same odrzuty?).")

    # =================================================================
    # LEAVE-ONE-OUT — teraz TYLKO na żądanie + cache (POPRAWKA #9)
    # =================================================================
    with st.expander("🔬 Silnik Diagnostyczny Leave-One-Out (ARI / NMI)",
                     expanded=False):
        if not gt_dostepny:
            st.info("Analiza Leave-One-Out wymaga Ground Truth (ARI/NMI).")
        else:
            st.markdown(
                "Algorytm izoluje po kolei każdą krzywą z bazy danych, uruchamia "
                "grupowanie od nowa i bada, jak jej brak wpływa na globalny wskaźnik "
                "ARI. **Czarne Owce** — usunięcie podnosi wynik. "
                "**Filary Modelu** — usunięcie obniża wynik."
            )
            st.caption(
                f"⏱️ Analiza uruchamia **{n_krzywych}** pełnych klasteryzacji — "
                "dlatego startuje dopiero po kliknięciu przycisku. Wynik jest "
                "cache'owany dla bieżących ustawień."
            )

            if st.button("▶️ Uruchom analizę Leave-One-Out", key="btn_loo"):
                st.session_state["loo_uruchomione"] = True

            if st.session_state.get("loo_uruchomione", False):
                wyniki_loo = oblicz_leave_one_out(
                    metoda, dane_std_oryginaly, tuple(etykiety_eksperta),
                    krzywe, tuple(str(n) for n in nazwy_do_wykresu),
                    liczba_grup, ari_score,
                )
                df_loo = pd.DataFrame(wyniki_loo).sort_values(
                    by="Wpływ na ARI", ascending=False
                ).reset_index(drop=True)

                col_loo1, col_loo2 = st.columns(2)
                fmt_loo = {
                    "ARI bez krzywej (%)": "{:.2f}%",
                    "NMI bez krzywej (%)": "{:.2f}%",
                }
                with col_loo1:
                    st.markdown('##### 🚨 "Czarne Owce" (usunięcie PODNOSI ARI):')
                    df_czarne = df_loo[df_loo["Wpływ na ARI"] > 0.01].reset_index(drop=True)
                    if not df_czarne.empty:
                        st.dataframe(
                            df_czarne.style.format(
                                {**fmt_loo, "Wpływ na ARI": "+{:.2f}%"}
                            ),
                            hide_index=True, use_container_width=True,
                        )
                    else:
                        st.info("Brak wyraźnych anomalii. Wszystkie krzywe "
                                "wspierają model.")
                with col_loo2:
                    st.markdown('##### 🧱 "Filary Modelu" (usunięcie drastycznie '
                                'OBNIŻA ARI):')
                    df_filary = df_loo[df_loo["Wpływ na ARI"] < -0.01].sort_values(
                        "Wpływ na ARI"
                    ).reset_index(drop=True)
                    if not df_filary.empty:
                        st.dataframe(
                            df_filary.style.format(
                                {**fmt_loo, "Wpływ na ARI": "{:.2f}%"}
                            ),
                            hide_index=True, use_container_width=True,
                        )
                    else:
                        st.info("Brak kluczowych filarów — podział grup jest stabilny.")

    # =================================================================
    # RANKING SKUTECZNOŚCI ALGORYTMÓW
    # =================================================================
    with st.expander("🏆 Ranking Skuteczności Algorytmów", expanded=False):
        opis_metryk = ("ARI, NMI oraz Silhouette Score" if gt_dostepny
                       else "Silhouette Score (brak Ground Truth — bez ARI/NMI)")
        st.markdown(
            f"Ranking uruchamia wszystkie metody klasteryzacji na tych samych "
            f"danych i porównuje {opis_metryk}. Wyniki są cachowane — ponowne "
            f"otwarcie jest natychmiastowe."
        )

        METODY_SZYBKIE = [
            "Hierarchiczna Aglomeracyjna (metoda Warda)",
            "PCA + Hierarchiczna (metoda Warda)",
            "K-means",
            "GMM (Probabilistyczna)",
            "Hierarchiczna Korelacyjna (metoda średnich)",
            "Spectral Clustering",
            "NMF (Nieujemna Faktoryzacja Macierzy)",
            "BGMM (Bayesowski GMM)",
        ]

        tryb_rankingu = st.radio(
            "Zakres rankingu:",
            ["⚡ Szybki (8 metod)", "🔬 Pełny (wszystkie metody)"],
            horizontal=True,
            key="tryb_rankingu",
        )
        lista_do_rankingu = (METODY_SZYBKIE if "Szybki" in tryb_rankingu
                             else lista_metod)

        krzywe_dla_rankingu = krzywe_aug if krzywe_aug is not None else krzywe
        rekordy, bledy_rankingu = oblicz_ranking(
            dane_do_algorytmu,
            tuple(etykiety_eksperta) if gt_dostepny else None,
            krzywe_dla_rankingu,
            tuple(lista_do_rankingu),
            liczba_grup,
            tuple(indeksy_oryginalow) if indeksy_oryginalow is not None else None,
        )

        if bledy_rankingu:
            with st.expander("⚠️ Metody pominięte w rankingu", expanded=False):
                for b in bledy_rankingu:
                    st.caption(f"• {b}")

        if rekordy:
            df_lb = pd.DataFrame(rekordy).sort_values(
                "Średnia (%)", ascending=False
            ).reset_index(drop=True)

            metryki = [c for c in ["ARI (%)", "NMI (%)", "Silhouette (%)"]
                       if c in df_lb.columns]

            klucz_sel = f"ranking_selekcja_{tryb_rankingu}_{liczba_grup}"
            if (klucz_sel not in st.session_state
                    or len(st.session_state[klucz_sel]) != len(df_lb)):
                st.session_state[klucz_sel] = [False] * len(df_lb)

            df_lb.insert(0, "Wybierz", st.session_state[klucz_sel])

            st.caption("Zaznacz metody które chcesz porównać, następnie kliknij "
                       "**Porównaj**.")

            konfiguracja_kolumn = {
                "Wybierz": st.column_config.CheckboxColumn("✔", width="small"),
                "Algorytm AI": st.column_config.TextColumn("Algorytm AI", disabled=True),
                "Średnia (%)": st.column_config.NumberColumn("Średnia (%)",
                                                             disabled=True,
                                                             format="%.2f"),
            }
            for m in metryki:
                konfiguracja_kolumn[m] = st.column_config.NumberColumn(
                    m, disabled=True, format="%.2f"
                )

            df_edytowalny = st.data_editor(
                df_lb,
                hide_index=True,
                use_container_width=True,
                column_config=konfiguracja_kolumn,
                key=f"ranking_editor_{klucz_sel}",
            )

            st.session_state[klucz_sel] = df_edytowalny["Wybierz"].tolist()
            zaznaczone = df_edytowalny[df_edytowalny["Wybierz"] == True]
            wszystkie_zaznaczone = len(zaznaczone) == len(df_edytowalny)

            col_btn1, col_btn2, col_info = st.columns([1, 1, 3])
            with col_btn1:
                if st.button(
                    "☑️ Odznacz wszystkie" if wszystkie_zaznaczone
                    else "✅ Wybierz wszystkie",
                    use_container_width=True,
                    key="btn_wybierz_wszystkie",
                ):
                    st.session_state[klucz_sel] = [not wszystkie_zaznaczone] * len(df_lb)
                    st.rerun()
            with col_btn2:
                porownaj = st.button(
                    "📊 Porównaj zaznaczone",
                    use_container_width=True,
                    disabled=len(zaznaczone) < 2,
                    key="btn_porownaj",
                )
            with col_info:
                if len(zaznaczone) < 2:
                    st.caption("⬅️ Zaznacz co najmniej 2 metody żeby porównać.")
                else:
                    st.caption(f"Zaznaczono **{len(zaznaczone)}** metod do porównania.")

            if porownaj or st.session_state.get("ranking_porownanie_aktywne", False):
                if porownaj:
                    st.session_state["ranking_porownanie_aktywne"] = True
                    st.session_state["ranking_porownanie_df"] = zaznaczone.drop(
                        columns=["Wybierz"]
                    ).reset_index(drop=True)

                df_por = st.session_state.get("ranking_porownanie_df", pd.DataFrame())
                if not df_por.empty:
                    st.markdown("---")
                    st.markdown("#### 📊 Porównanie wybranych metod")

                    metryki_por = [c for c in ["ARI (%)", "NMI (%)", "Silhouette (%)"]
                                   if c in df_por.columns]
                    fig_por = go.Figure()
                    for idx, row_por in df_por.iterrows():
                        fig_por.add_trace(go.Bar(
                            name=row_por["Algorytm AI"],
                            x=metryki_por,
                            y=[row_por[m] for m in metryki_por],
                            marker_color=PLOTLY_KOLORY[idx % 10],
                            text=[f"{row_por[m]:.1f}%" for m in metryki_por],
                            textposition="outside",
                        ))
                    fig_por.update_layout(
                        barmode="group",
                        height=380,
                        margin=dict(l=10, r=10, t=30, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        yaxis=dict(range=[0, 115], title="Wartość (%)"),
                        xaxis=dict(title="Metryka"),
                    )
                    st.plotly_chart(fig_por, use_container_width=True)

                    df_styl = df_por.set_index("Algorytm AI")

                    def podswietl_max(s):
                        return ["background-color: #d4edda; font-weight: bold"
                                if v == s.max() else "" for v in s]

                    st.dataframe(
                        df_styl.style
                        .apply(podswietl_max, subset=metryki_por)
                        .format({m: "{:.2f}%" for m in metryki_por + ["Średnia (%)"]}),
                        use_container_width=True,
                    )

                    najlepsza = df_por.loc[df_por["Średnia (%)"].idxmax(),
                                           "Algorytm AI"]
                    st.success(f"🏆 Najlepsza metoda w porównaniu: **{najlepsza}**")

                    if st.button("✖️ Zamknij porównanie", key="btn_zamknij_por"):
                        st.session_state["ranking_porownanie_aktywne"] = False
                        st.session_state["ranking_porownanie_df"] = pd.DataFrame()
                        st.rerun()
        else:
            st.info("Brak wyników — sprawdź dane wejściowe.")

except Exception as ob_blad:
    st.error(f"Błąd krytyczny podczas renderowania: {ob_blad}")
    if KONFIG["DEBUG"]:
        st.exception(ob_blad)  # pełny traceback do debugowania
