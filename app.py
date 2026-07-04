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
#
# [WERSJA 2.1]
# 17. Metody czysto hierarchiczne (dendrogramowe, np. Ward) pokazują teraz
#     także Wykres 2 (profile modelowe), szczegółowy skład klastrów oraz
#     eksport CSV/Excel — wcześniej sekcje te były dla nich ukryte, mimo że
#     przypisania z fcluster() były liczone (służyły do ARI/NMI).
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
