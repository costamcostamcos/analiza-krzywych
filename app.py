import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import io
import numpy as np

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

# =================================================================
# SŁOWNIK INTELIGENTNYCH OPISÓW METOD KLASTERYZACJI
# =================================================================
OPISY_METOD = {
    "K-means": "Dzieli przestrzeń cech na tzw. obszary Voronoia. Algorytm dąży do minimalizacji wariancji wewnątrzklastrowej poprzez naprzemienne przypisywanie obiektów do najbliższych prototypów (środków ciężkości) i aktualizację tych środków.",
    "Klastrowanie Konsensusowe (Ensemble Voting)": "Metoda komitetowa (Ensemble Learning). Uruchamia równolegle zróżnicowany zestaw algorytmów (K-Means, GMM, Spectral, Ward) i buduje macierz współwystępowania, rejestrującą jak często dane dwie krzywe były przypisywane do jednej grupy. Ostateczny podział jest fuzją decyzji wszystkich modeli.",
    "PSO (Optymalizacja Rojem Cząstek)": "Metaheurystyka inspirowana naturą, imitująca zachowanie stada ptaków. Zamiast pojedynczego punktu startowego, w wielowymiarowej przestrzeni porusza się populacja (rój) cząstek-zwiadowców.",
    "NMF (Nieujemna Faktoryzacja Macierzy)": "Algorytm nieliniowej redukcji wymiarowości, który rozkłada macierz danych na iloczyn dwóch macierzy o elementach wyłącznie nieujemnych. Traktuje Twoje krzywe jako kombinację liniową bazowych, nieujemnych klocków sygnałowych.",
    "GMM (Probabilistyczna)": "Modele Mieszanin Gaussowskich. Zakłada, że struktura danych pod wejściem składa się z określonej liczby wielowymiarowych rozkładów normalnych. Realizuje tzw. miękkie przypisanie (soft clustering).",
    "BGMM (Bayesowski GMM)": "Rozszerzenie GMM o probabilistyczną Bayesowską z procesem Dirichleta. Traktuje parametry klastrów jako zmienne losowe. Automatycznie wygasza niepotrzebne klastry.",
    "Hierarchiczna Aglomeracyjna (metoda Warda)": "Algorytm budujący drzewo powiązań od dołu do góry. Każda krzywa startuje jako osobny klaster, a w kolejnych krokach łączone są grupy, które generują najmniejszy możliwy wzrost całkowitej wariancji wewnątrzklastrowej.",
    "Hierarchiczna Korelacyjna (metoda średnich)": "Podejście hierarchiczne (UPGMA), które zamiast klasycznej odległości przestrzennej mierzy stopień współliniowości wykresów za pomocą odległości korelacyjnej (1 - r Pearsona).",
    "HDBSCAN (Gęstościowa - Auto K)": "Zaawansowane klastrowanie gęstościowe oparte na teorii grafów. Szuka obszarów o wysokiej kondensacji punktów oddzielonych strefami pustki. Nie wymaga definiowania liczby klastrów (K).",
    "Spectral Clustering": "Wykorzystuje wartości własne (widmo) macierzy podobieństwa danych do redukcji wymiarowości przed właściwym podziałem. Buduje graf powiązań między wszystkimi krzywymi.",
    "K-Shape (Kształt fali)": "Wyspecjalizowany algorytm stworzony ściśle do analizy kształtu serii czasowych. Wykorzystuje znormalizowaną korelację wzajemną. Rozpoznaje kształt fali przesuniętej w czasie.",
    "DEC (Głębokie Uczenie - Sieć Neuronowa)": "Sztuczna sieć neuronowa (Autoenkoder) szkolona na bazie danych namnożonej przez augmentację sygnału (z 44 do 2200 krzywych).",
    "ADEC (Adwersarialne Głębokie Uczenie)": "Pojedynek adwersarialny enkodera i dyskryminatora zasilany sztucznie namnożonym zbiorem danych (2200 prób). Wymusza ostre i bardzo zwarte granice między grupami, całkowicie zapobiegając przeuczeniu.",
    "RDEC (Regularizowane Głębokie Uczenie)": "Model DEC wyposażony w silne bariery regularyzacyjne (L2) oraz zaawansowany moduł augmentacji sygnału. Zmusza sieć neuronową do szukania najprostszych, najbardziej powtarzalnych wzorców geometrycznych fal.",
    "ADClust (Automatyczne Głębokie Uczenie)": "Autonomiczny kombajn AI, który sam decyduje o liczbie grup za pomocą wskaźnika Silhouette, wykonując uprzednio proces głębokiego uczenia na 2200 wygenerowanych matematycznie wariantach."
}

# =================================================================
# SŁOWNIK OPISÓW WSTĘPNEGO PRZYGOTOWANIA DANYCH
# =================================================================
OPISY_PREPROCESSING = {
    "Standardowa": "Polega na klasycznej standaryzacji (Z-score). Sprowadza wszystkie punkty pomiarowe krzywych do wspólnej skali statystycznej (średnia=0, odchylenie=1).",
    "Analiza trendu": "Wyznacza różnice skończone (pochodne pierwszego rzędu) pomiędzy sąsiednimi punktami wzdłuż osi X. Algorytmy badają prędkość narastania i opadania sygnału.",
    "FeatureExtraction": "Głęboka transformacja inżynierska 3D: Max, Pozycja X, Średnia, Std, Skośność, Kurtoza, pierwsze 3 harmoniczne FFT oraz 3 wskaźniki DWT Haar (Aproksymacja i Detale).",
    "MinMaxScaler": "Dokonuje liniowej transformacji danych, przesuwając i skalując wartości każdej krzywej tak, aby zamknęły się w ścisłym przedziale od 0 do 1.",
    "Filtrowanie szumów": "Wykorzystuje algorytm kroczącego okna średniej (rolling window). Skutecznie odcina fluktuacje wysokiej częstotliwości i przypadkowe szpilki pomiarowe."
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

def augmentuj_dane(X_oryginalne, czynniki_kopii=50, noise_level=0.02, scale_range=0.05):
    N, F = X_oryginalne.shape
    X_namnozone = []
    X_namnozone.append(X_oryginalne)
    for c in range(czynniki_kopii - 1):
        kopia = np.copy(X_oryginalne)
        szum = np.random.normal(0, noise_level, size=kopia.shape)
        kopia += szum
        skala = np.random.uniform(1.0 - scale_range, 1.0 + scale_range, size=(N, 1))
        kopia *= skala
        for i in range(N):
            stary_indeks = np.arange(F)
            nowy_indeks = stary_indeks + np.random.uniform(-0.4, 0.4, size=F)
            nowy_indeks = np.clip(nowy_indeks, 0, F - 1)
            kopia[i] = np.interp(stary_indeks, nowy_indeks, kopia[i])
        X_namnozone.append(kopia)
    return np.vstack(X_namnozone)

# =================================================================
# IMPLEMENTACJA ALGORYTMU ROJU CZĄSTEK (PSO CLUSTERING)
# =================================================================
class PSOClustering:
    def __init__(self, n_clusters, n_particles=15, max_iter=30, random_state=42):
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.random_state = random_state
        
    def _compute_sse(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2), labels

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        N, F = X.shape
        K = self.n_clusters
        positions = np.zeros((self.n_particles, K, F))
        for i in range(self.n_particles):
            idx = np.random.choice(N, K, replace=False)
            positions[i] = X[idx]
        velocities = np.zeros_like(positions)
        pbest_positions = np.copy(positions)
        pbest_fitness = np.full(self.n_particles, np.inf)
        gbest_position = None
        gbest_fitness = np.inf
        
        for i in range(self.n_particles):
            fit, _ = self._compute_sse(X, positions[i])
            pbest_fitness[i] = fit
            if fit < gbest_fitness:
                gbest_fitness = fit
                gbest_position = np.copy(positions[i])
                
        w, c1, c2 = 0.729, 1.494, 1.494
        for iteration in range(self.max_iter):
            r1 = np.random.rand(self.n_particles, K, 1)
            r2 = np.random.rand(self.n_particles, K, 1)
            velocities = (w * velocities + c1 * r1 * (pbest_positions - positions) + c2 * r2 * (gbest_position[np.newaxis, :, :] - positions))
            positions += velocities
            for i in range(self.n_particles):
                fit, _ = self._compute_sse(X, positions[i])
                if fit < pbest_fitness[i]:
                    pbest_fitness[i] = fit
                    pbest_positions[i] = np.copy(positions[i])
                if fit < gbest_fitness:
                    gbest_fitness = fit
                    gbest_position = np.copy(positions[i])
        _, labels = self._compute_sse(X, gbest_position)
        return labels + 1

# =================================================================
# ARCHITEKTURY SIECI NEURONOWYCH (DLA ENKODERÓW AI)
# =================================================================
if pytorch_dostepne:
    class AutoencoderKrzywych(nn.Module):
        def __init__(self, input_dim, latent_dim=4):
            super(AutoencoderKrzywych, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, input_dim))
        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return latent, reconstructed

    class DiscriminatorADEC(nn.Module):
        def __init__(self, latent_dim=4):
            super(DiscriminatorADEC, self).__init__()
            self.model = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        def forward(self, x):
            return self.model(x)

# Globalna funkcja wykonawcza silnika klastrowania
def uruchom_silnik_klastrowania(nazwa_metody, dane, k_grup, min_hdbscan=3):
    if nazwa_metody == "K-means":
        return KMeans(n_clusters=k_grup, random_state=42, n_init=5).fit_predict(dane) + 1
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
    elif "PSO" in nazwa_metody:
        return PSOClustering(n_clusters=k_grup, random_state=42).fit_predict(dane)
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
    elif "DEC" in nazwa_metody and pytorch_dostepne:
        X_aug = torch.FloatTensor(augmentuj_dane(dane, czynniki_kopii=15))
        net = AutoencoderKrzywych(input_dim=dane.shape[1], latent_dim=4)
        opt = optim.Adam(net.parameters(), lr=0.01)
        for e in range(15):
            opt.zero_grad()
            _, rec = net(X_aug)
            loss = nn.MSELoss()(rec, X_aug)
            loss.backward()
            opt.step()
        with torch.no_grad(): lat, _ = net(torch.FloatTensor(dane))
        return KMeans(n_clusters=k_grup, random_state=42, n_init=5).fit_predict(lat.numpy()) + 1
    else:
        return KMeans(n_clusters=k_grup, random_state=42, n_init=5).fit_predict(dane) + 1

# =================================================================
# SEKCJA INTERFEJSU UŻYTKOWNIKA (UI) STREAMLIT
# =================================================================
st.write("### Ustawienia analizy")
typ_zrodla = st.radio("Wybierz źródło danych:", ["Plik Excel (.xlsx)", "Link do Google Sheets"], horizontal=True)

df = None
if typ_zrodla == "Plik Excel (.xlsx)":
    uploaded_file = st.file_uploader("Wgraj plik Excel", type=["xlsx"])
    if uploaded_file is not None:
        df_raw = pd.read_excel(uploaded_file, header=None)
        df = inteligentne_pobranie_tabeli(df_raw)
else:
    link_sheets = st.text_input("Wklej link do Google Sheets:", placeholder="https://docs.google.com/spreadsheets/d/...")
    if link_sheets and "docs.google.com/spreadsheets" in link_sheets:
        try:
            url_eksportu = f'{link_sheets.split("/edit")[0]}/export?format=xlsx'
            df = inteligentne_pobranie_tabeli(pd.read_excel(url_eksportu, header=None))
        except Exception: st.error("Nie udało się pobrać danych z Google Sheets.")

if df is not None:
    try:
        x = df.iloc[:, 0]
        krzywe = df.iloc[:, 1:]
        nazwy_krzywych = krzywe.columns.tolist()
        
        lista_metod = ["K-means", "Klastrowanie Konsensusowe (Ensemble Voting)", "PSO (Optymalizacja Rojem Cząstek)", "NMF (Nieujemna Faktoryzacja Macierzy)", "GMM (Probabilistyczna)", "BGMM (Bayesowski GMM)", "Hierarchiczna Aglomeracyjna (metoda Warda)", "Hierarchiczna Korelacyjna (metoda średnich)", "HDBSCAN (Gęstościowa - Auto K)", "Spectral Clustering"]
        if tslearn_dostepne: lista_metod.append("K-Shape (Kształt fali)")
        if pytorch_dostepne: lista_metod.extend(["DEC (Głębokie Uczenie - Sieć Neuronowa)", "ADEC (Adwersarialne Głębokie Uczenie)", "RDEC (Regularizowane Głębokie Uczenie)", "ADClust (Automatyczne Głębokie Uczenie)"])

        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1: metoda = st.selectbox("Wybierz metodę główną:", lista_metod, key="wybrana_metoda", help=OPISY_METOD.get(st.session_state.wybrana_metoda, ""))
        with col_param2: optymalizacja = st.selectbox("Wybierz wstępne przygotowanie danych:", ["Standardowa", "Analiza trendu", "FeatureExtraction", "MinMaxScaler", "Filtrowanie szumów"]) if "K-Shape" not in metoda and "DEC" not in metoda and "RDEC" not in metoda and "ADClust" not in metoda and "NMF" not in metoda else "Standardowa"
        with col_param3: liczba_grup = st.slider("Minimalna wielkość grupy (HDBSCAN):" if "HDBSCAN" in metoda else "Maksymalna liczba grup (BGMM):" if "BGMM" in metoda else "Liczba grup (K):", min_value=2, max_value=10, value=5) if "ADClust" not in metoda else 5

        st.write("---")
        col_main, col_sidebar = st.columns([3, 1])
        
        with col_sidebar:
            st.markdown("### Spodziewany Podział Grup")
            st.caption("Wpisz litery (a, b, c, d, e...) odpowiadające rzeczywistym grupom krzywych:")
            if 'expert_dict' not in st.session_state or len(st.session_state.expert_dict) != len(nazwy_krzywych):
                st.session_state.expert_dict = pd.DataFrame({"Krzywa": nazwy_krzywych, "Grupa Eksperta": ["a"] * len(nazwy_krzywych)})
            
            edited_gt = st.data_editor(st.session_state.expert_dict, use_container_width=True, hide_index=True, disabled=["Krzywa"])
            st.session_state.expert_dict = edited_gt
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

            numery_grup = uruchom_silnik_klastrowania(metoda, dane_do_algorytmu, liczba_grup, liczba_grup)

            ari_score = adjusted_rand_score(etykiety_eksperta, numery_grup) * 100
            nmi_score = normalized_mutual_info_score(etykiety_eksperta, numery_grup) * 100
            
            st.markdown(f"### Skuteczność dopasowania do kryteriów spodziewanego podziału:")
            kpi_ari, kpi_nmi = st.columns(2)
            
            # DODANIE DYMKÓW Z OPISAMI MATEMATYCZNYMI WSKAŹNIKÓW (ARGUMENT HELP)
            kpi_ari.metric(
                "Indeks ARI (Zgodność par obiektów)", 
                f"{ari_score:.1f}%",
                help="Adjusted Rand Index (ARI): Miara zgodności podziału dokonanego przez algorytm ze spodziewanym podziałem. Wartość jest korygowana o losowe prawdopodobieństwo trafienia. Przyjmuje wartości z przedziału [-1, 1], gdzie 1 oznacza idealną zbieżność par obiektów."
            )
            kpi_nmi.metric(
                "Indeks NMI (Zbieżność informacji sygnału)", 
                f"{nmi_score:.1f}%",
                help="Normalized Mutual Information (NMI): Miara oparta na teorii informacji (entropii), określająca jak dużo wiedzy o podziale spodziewanym dostarcza podział wyznaczony przez model. Wynik jest normalizowany do przedziału [0, 1] (lub 0-100%)."
            )

            wyniki = pd.DataFrame({'Krzywa': nazwy_krzywych, 'Numer Grupy': numery_grup}).sort_values(by='Numer Grupy')
            fig, ax = plt.subplots(figsize=(10, 4.5))
            cmap = plt.get_cmap('tab10')
            if "Hierarchiczna" in metoda:
                dendrogram(linkage(dane_do_algorytmu, method='ward' if "Warda" in metoda else 'average', metric='euclidean' if "Warda" in metoda else 'correlation'), labels=nazwy_krzywych, leaf_rotation=90, ax=ax)
            else:
                for i, col in enumerate(krzywe.columns):
                    ax.plot(x, krzywe[col], color=cmap((numery_grup[i] - 1) % 10) if numery_grup[i]>0 else 'gray', alpha=0.6)
                ax.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

        # =================================================================
        # RANKING METOD
        # =================================================================
        st.write("---")
        st.subheader("Ranking Skuteczności Algorytmów")
        
        rekordy_rankingu = []
        for m_nazwa in lista_metod:
            try:
                pred_etykiety = uruchom_silnik_klastrowania(m_nazwa, dane_do_algorytmu, liczba_grup, liczba_grup)
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

        # Sekcje podpowiedzi matematycznych
        if "HDBSCAN" not in metoda and "ADClust" not in metoda:
            with st.expander("Zaawansowana Podpowiedź Matematyczna (Dobór liczby klastrów K)", expanded=False):
                t_elb, t_sil, t_bic, t_gap = st.tabs(["Metoda Łokcia & Kneedle", "Silhouette Score", "Indeks BIC", "Statystyka Gap"])
                z_k = range(2, 11)
                iner, silh, bics = [], [], []
                for k in z_k:
                    km = KMeans(n_clusters=k, random_state=42, n_init=3).fit(dane_do_algorytmu)
                    iner.append(km.inertia_)
                    silh.append(silhouette_score(dane_do_algorytmu, km.labels_))
                    g_t = GaussianMixture(n_components=k, covariance_type='diag', random_state=42, n_init=1).fit(dane_do_algorytmu)
                    bics.append(g_t.bic(dane_do_algorytmu))
                
                with t_elb:
                    p1, p2 = np.array([z_k[0], iner[0]]), np.array([z_k[-1], iner[-1]])
                    k_k = z_k[np.argmax([np.abs(np.cross(p2 - p1, p1 - np.array([k, iner[i]]))) / np.linalg.norm(p2 - p1) for i, k in enumerate(z_k)])]
                    st.info(f"Kneedle sugeruje: K = {k_k}.")
                    fig_e, ax_e = plt.subplots(figsize=(10, 2.5))
                    ax_e.plot(z_k, iner, 'ro-')
                    ax_e.axvline(x=k_k, color='purple', linestyle='--')
                    st.pyplot(fig_e); plt.close(fig_e)
                with t_sil:
                    fig_s, ax_s = plt.subplots(figsize=(10, 2.5))
                    ax_s.plot(z_k, silh, 'bo-')
                    st.pyplot(fig_s); plt.close(fig_s)
                with t_bic:
                    fig_b, ax_b = plt.subplots(figsize=(10, 2.5))
                    ax_b.plot(z_k, bics, 'go-')
                    st.pyplot(fig_b); plt.close(fig_b)
                with t_gap:
                    B, shape = 3, dane_do_algorytmu.shape
                    log_Wk = np.log(iner)
                    log_Wk_ref = np.zeros((B, len(z_k)))
                    for b in range(B):
                        rand_d = np.random.uniform(dane_do_algorytmu.min(axis=0), dane_do_algorytmu.max(axis=0), size=shape)
                        for i_k, k in enumerate(z_k): log_Wk_ref[b, i_k] = np.log(KMeans(n_clusters=k, random_state=42+b, n_init=2).fit(rand_d).inertia_)
                    gaps = np.mean(log_Wk_ref, axis=0) - log_Wk
                    sk = np.std(log_Wk_ref, axis=0) * np.sqrt(1 + 1.0/B)
                    ok_g = z_k[-1]
                    for idx in range(len(z_k)-1):
                        if gaps[idx] >= gaps[idx+1] - sk[idx+1]: ok_g = z_k[idx]; break
                    st.info(f"Reguła Tibshiraniego sugeruje: K = {ok_g}.")
                    fig_g, (ax_g1, ax_g2) = plt.subplots(1, 2, figsize=(11, 2.8))
                    ax_g1.plot(z_k, np.mean(log_Wk_ref, axis=0), 'co-', label='Szum')
                    ax_g1.plot(z_k, log_Wk, 'yo-', label='Dane')
                    ax_g2.errorbar(z_k, gaps, yerr=sk, fmt='mo-')
                    st.pyplot(fig_g); plt.close()

    except Exception as ob_blad: st.error(f"Błąd krytyczny aplikacji: {ob_blad}")
else:
    st.info("Aby rozpocząć, wgraj plik z dysku lub wklej link do Google Sheets powyżej.")
