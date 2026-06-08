import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.decomposition import NMF
from sklearn.metrics import silhouette_score
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
    pytorch_dostepne = True
except ImportError:
    pytorch_dostepne = False

# =================================================================
# SŁOWNIK INTELIGENTNYCH OPISÓW METOD KLASTERYZACJI
# =================================================================
OPISY_METOD = {
    "K-means": "Dzieli przestrzeń cech na tzw. obszary Voronoia. Algorytm dąży do minimalizacji wariancji wewnątrzklastrowej poprzez naprzemienne przypisywanie obiektów do najbliższych prototypów (środków ciężkości) i aktualizację tych środków. Najlepiej sprawdza się, gdy klastry są zwarte, odizolowane i sferyczne.",
    "PSO (Optymalizacja Rojem Cząstek)": "Metaheurystyka inspirowana naturą, imitująca zachowanie stada ptaków. Zamiast pojedynczego punktu startowego, w wielowymiarowej przestrzeni porusza się populacja (rój) cząstek-zwiadowców. Każda cząstka koryguje swój tor lotu na podstawie własnych doświadczeń oraz sukcesów całego roju, co pozwala skutecznie omijać lokalne minima matematyczne.",
    "NMF (Nieujemna Faktoryzacja Macierzy)": "Algorytm nieliniowej redukcji wymiarowości, który rozkłada macierz danych na iloczyn dwóch macierzy o elementach wyłącznie nieujemnych. Traktuje Twoje krzywe jako kombinację liniową bazowych, nieujemnych 'klocków' sygnałowych. Przypisanie do grupy następuje na podstawie dominującego komponentu fizycznego, co eliminuje nienaturalne matematycznie wartości ujemne.",
    "GMM (Probabilistyczna)": "Modele Mieszanin Gaussowskich. Zakłada, że struktura danych pod wejściem składa się z określonej liczby wielowymiarowych rozkładów normalnych. Realizuje tzw. 'miękkie przypisanie' (soft clustering) – zamiast suchej decyzji 0/1, wylicza procentową pewność (prawdopodobieństwo), z jaką dana krzywa pasuje do każdego z klastrów. Idealne do identyfikacji próbek granicznych.",
    "BGMM (Bayesowski GMM)": "Rozszerzenie GMM o probabilistykę Bayesowską z procesem Dirichleta. Traktuje parametry klastrów jako zmienne losowe. Posiada unikalną inżynierską zaletę: jeśli zadana maksymalna liczba grup jest zbyt duża, algorytm automatycznie wygasza niepotrzebne klastry (przypisuje im wagę bliską zeru), chroniąc model przed przeuczeniem na małych zbiorach danych.",
    "Hierarchiczna Aglomeracyjna (metoda Warda)": "Algorytm budujący drzewo powiązań od dołu do góry. Każda krzywa startuje jako osobny klaster, a w kolejnych krokach łączone są grupy, które generują najmniejszy możliwy wzrost całkowitej wariancji wewnątrzklastrowej (błędu SSE). Wynik końcowy w postaci dendrogramu pozwala na pełną ocenę struktury pokrewieństwa sygnałów.",
    "Hierarchiczna Korelacyjna (metoda średnich)": "Podejście hierarchiczne (UPGMA), które zamiast klasycznej odległości przestrzennej (metryki Euklidesowej) mierzy stopień współliniowości wykresów za pomocą odległości korelacyjnej (1 - r Pearsona). Łączy grupy na podstawie średnich powiązań, skupiając się wyłącznie na synchroniczności trendów i kształcie fali, ignorując skalę i przesunięcia pionowe Y.",
    "HDBSCAN (Gęstościowa - Auto K)": "Zaawansowane klastrowanie gęstościowe oparte na teorii grafów. Szuka obszarów o wysokiej kondensacji punktów oddzielonych strefami pustki. Nie wymaga definiowania liczby klastrów (K). Krzywe nietypowe lub zaszumione są automatycznie odrzucane i oznaczane jako grupa 0, dzięki czemu nie zaburzają one czystości głównych profili.",
    "Spectral Clustering": "Wykorzystuje wartości własne (widmo) macierzy podobieństwa danych do redukcji wymiarowości przed właściwym podziałem. Buduje graf powiązań między wszystkimi krzywymi i szuka optymalnych cięć topologicznych tego grafu. Genialnie radzi sobie z układami nieliniowymi i strukturami zagnieżdżonymi wewnątrz siebie.",
    "K-Shape (Kształt fali)": "Wyspecjalizowany algorytm stworzony ściśle do analizy kształtu serii czasowych. Wykorzystuje znormalizowaną korelację wzajemną (cross-correlation) jako miarę odległości geometrycznej. Potrafi rozpoznać, że dwie linie mają ten sam kształt, nawet jeśli ich piki charakterystyczne są przesunięte w czasie (w lewo lub w prawo).",
    "DEC (Głębokie Uczenie - Sieć Neuronowa)": "Sztuczna sieć neuronowa (Autoenkoder) uczy się nieliniowej kompresji danych do małej przestrzeni ukrytej (latent space), jednocześnie optymalizując centra klastrów poprzez minimalizację dywergencji Kullbacka-Leiblera (KL). Proces ten odrzuca skomplikowany, nieliniowy szum laboratoryjny.",
    "ADEC (Adwersarialne Głębokie Uczenie)": "Rozbudowanie sieci DEC o trening adwersarialny (koncepcja GAN). Dodatkowy blok Dyskryminatora walczy z Enkoderem, zmuszając go do ułożenia cech krzywych w idealnie gładki rozkład matematyczny. Eliminuje to puste przestrzenie w strukturze danych, generując niezwykle zwarte klastry o ostrych granicach.",
    "RDEC (Regularizowane Głębokie Uczenie)": "Model DEC wyposażony w silne bariery regularyzacyjne (L2 oraz weight decay). Nakłada matematyczną karę za zbyt skomplikowane wagi sieci oraz zbyt wysokie rozproszenie przestrzeni ukrytej. Zmusza to sieć neuronową do szukania najprostszych, najbardziej fundamentalnych trendów geometrycznych fali, chroniąc przed przeuczeniem.",
    "ADClust (Automatyczne Głębokie Uczenie)": "Autonomiczny kombajn AI. Głęboki Autoenkoder kompresuje sygnał do przestrzeni cech ukrytych, a zaimplementowany wewnątrz pętli uczącej moduł statystyczny skanuje przestrzeń indeksem Silhouette, samodzielnie zatwierdzając matematycznie najlepszą liczbę klastrów bez ingerencji inżyniera."
}

# =================================================================
# SŁOWNIK OPISÓW WSTĘPNEGO PRZYGOTOWANIA DANYCH
# =================================================================
OPISY_PREPROCESSING = {
    "Standardowa": "Polega na klasycznej standaryzacji (Z-score). Od każdej wartości punktu odejmowana jest średnia danej kolumny, a wynik dzielony jest przez jej odchylenie standardowe. Sprowadza to wszystkie punkty pomiarowe krzywych do wspólnej skali statystycznej (średnia=0, odchylenie=1), eliminując sytuację, w której bezwzględna wartość sygnału dominuje nad jego dynamiką.",
    "Analiza trendu": "Wyznacza różnice skończone (pochodne pierwszego rzędu) pomiędzy sąsiednimi punktami wzdłuż osi X (`y_next - y_current`), a następnie poddaje je standaryzacji. Transformacja ta przenosi analizę w obszar czystej dynamiki linii. Algorytmy badają prędkość narastania i opadania sygnału (nachylenie zboczy), całkowicie ignorując pozycję wykresów w pionie.",
    "FeatureExtraction": "Głęboka transformacja inżynierska. Zamiast surowych setek punktów, każda krzywa opisywana jest przez 9 zaawansowanych deskryptorów: wartość maksymalną, pozycję piku X, średnią, odchylenie standardowe, skośność (asymetrię fali), kurtozę (strzelistość pików) oraz amplitudy pierwszych 3 głównych składowych harmonicznych uzyskanych z Szybkiej Transformaty Fouriera (FFT). Pozwala algorytmom badać sygnał w dziedzinie częstotliwości.",
    "MinMaxScaler": "Dokonuje liniowej transformacji danych, przesuwając i skalując wartości każdej krzywej tak, aby zamknęły się w ścisłym, znormalizowanym przedziale od 0 do 1. Metoda ta zachowuje oryginalne proporcje amplitud i jest bezwzględnie wymagana przez algorytmy takie jak NMF, które matematycznie nie tolerują wartości ujemnych.",
    "Filtrowanie szumów": "Wykorzystuje algorytm kroczącego okna średniej (`rolling window`) o zadanym rozmiarze, centrując wynik. Każdy punkt wykresu zastępowany jest średnią arytmetyczną z jego bezpośredniego otoczenia. Operacja ta skutecznie odcina fluktuacje wysokiej częstotliwości, przypadkowe szpilki pomiarowe i zakłócenia aparatury, wygładzając nadrzędny profil fali."
}

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
            
            velocities = (w * velocities + 
                          c1 * r1 * (pbest_positions - positions) + 
                          c2 * r2 * (gbest_position[np.newaxis, :, :] - positions))
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
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim)
            )
        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return latent, reconstructed

    class DiscriminatorADEC(nn.Module):
        def __init__(self, latent_dim=4):
            super(DiscriminatorADEC, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.model(x)

# Konfiguracja strony
st.set_page_config(page_title="Analizator Krzywych Pro AI", layout="wide")

st.title("📊 Interaktywny Analizator Krzywych AI Pro")
st.write("Wgraj plik Excel lub wklej link do Google Sheets. System automatycznie dopasuje metody sztucznej inteligencji.")

# =================================================================
# INTELIGENTNY DETEKTYW STARTU TABELI
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
# SEKCJA INTERFEJSU
# =================================================================
st.write("### ⚙️ Ustawienia analizy")

typ_zrodla = st.radio(
    "Wybierz źródło danych:", 
    ["Plik Excel (.xlsx)", "Link do Google Sheets"], 
    horizontal=True
)

df = None

if typ_zrodla == "Plik Excel (.xlsx)":
    uploaded_file = st.file_uploader("Wgraj plik Excel", type=["xlsx"])
    if uploaded_file is not None:
        df_raw = pd.read_excel(uploaded_file, header=None)
        df = inteligentne_pobranie_tabeli(df_raw)
else:
    link_sheets = st.text_input(
        "Wklej link do Google Sheets:", 
        placeholder="https://docs.google.com/spreadsheets/d/..."
    )
    st.caption("⚠️ Ważne: Udostępnianie arkusza musi być ustawione na: 'Każdy użytkownik posiadający link może przeglądać'")
    
    if link_sheets:
        try:
            if "docs.google.com/spreadsheets" in link_sheets:
                bazowy_url = link_sheets.split("/edit")[0]
                url_eksportu = f"{bazowy_url}/export?format=xlsx"
                df_raw = pd.read_excel(url_eksportu, header=None)
                df = inteligentne_pobranie_tabeli(df_raw)
            else:
                st.error("To nie wygląda na poprawny link do Google Sheets.")
        except Exception as e:
            st.error("Nie udało się pobrać danych. Sprawdź udostępnienie linku.")

# Jeśli dane zostały poprawnie znalezione i przetworzone
if df is not None:
    try:
        x = df.iloc[:, 0]
        krzywe = df.iloc[:, 1:]
        nazwy_krzywych = krzywe.columns.tolist()
        
        # Słownik metod głównych
        lista_metod = [
            "K-means", 
            "PSO (Optymalizacja Rojem Cząstek)",
            "NMF (Nieujemna Faktoryzacja Macierzy)",
            "GMM (Probabilistyczna)", 
            "BGMM (Bayesowski GMM)",
            "Hierarchiczna Aglomeracyjna (metoda Warda)", 
            "Hierarchiczna Korelacyjna (metoda średnich)",
            "HDBSCAN (Gęstościowa - Auto K)", 
            "Spectral Clustering"
        ]
        if tslearn_dostepne:
            lista_metod.append("K-Shape (Kształt fali)")
        if pytorch_dostepne:
            lista_metod.append("DEC (Głębokie Uczenie - Sieć Neuronowa)")
            lista_metod.append("ADEC (Adwersarialne Głębokie Uczenie)")
            lista_metod.append("RDEC (Regularizowane Głębokie Uczenie)")
            lista_metod.append("ADClust (Automatyczne Głębokie Uczenie)")
            
        if 'wybrana_metoda' not in st.session_state:
            st.session_state.wybrana_metoda = lista_metod[0]

        # Układ parametrów - 3 kolumny
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            metoda = st.selectbox(
                "Wybierz metodę główną:", 
                lista_metod, 
                index=lista_metod.index(st.session_state.wybrana_metoda) if st.session_state.wybrana_metoda in lista_metod else 0,
                help=OPISY_METOD.get(st.session_state.wybrana_metoda, "")
            )
            st.session_state.wybrana_metoda = metoda
        
        with col_param2:
            if "K-Shape" not in metoda and "DEC" not in metoda and "RDEC" not in metoda and "ADClust" not in metoda and "NMF" not in metoda:
                optymalizacja = st.selectbox(
                    "Wybierz wstępne przygotowanie danych:", 
                    ["Standardowa", "Analiza trendu", "FeatureExtraction", "MinMaxScaler", "Filtrowanie szumów"]
                )
            else:
                optymalizacja = "Standardowa"
                st.selectbox("Optymalizacja wbudowana w algorytm", ["Wbudowana (Auto-Embedding / Skalowanie)"], disabled=True)
                
        with col_param3:
            if "HDBSCAN" in metoda:
                min_wielkosc = st.slider("Minimalna wielkość grupy (Min Cluster Size):", min_value=2, max_value=10, value=3)
            elif "ADClust" in metoda:
                st.text_input("Liczba grup (K):", value="Automatycznie przez AI 🤖", disabled=True)
            else:
                label_k = "Maksymalna liczba grup (K):" if "BGMM" in metoda else "Wybierz oczekiwaną liczbę grup (K):"
                liczba_grup = st.slider(label_k, min_value=2, max_value=10, value=5)

        # =================================================================
        # REWOLUCJA: ROZBUDOWANY, DYNAMICZNY PANEL METODOLOGICZNY (DWIE KOLUMNY)
        # =================================================================
        with st.expander("📚 Kompleksowy Opis Metodologiczny (Teoria & Synergia Operacyjna)", expanded=True):
            col_desc1, col_desc2 = st.columns(2)
            with col_desc1:
                st.markdown(f"#### 🤖 Algorytm Główny: `{metoda}`")
                st.write(OPISY_METOD.get(metoda, ""))
                # Dynamiczny opis synergii z uwzględnieniem wybranej obróbki danych
                st.markdown(f"**Kontekst operacyjny:** Wybranie obróbki *'{optymalizacja}'* sprawia, że algorytm `{metoda.split()[0]}` nie analizuje surowego sygnału Excela bezpośrednio w punktach, lecz przetwarza macierz matematyczną ustrukturyzowaną ściśle pod kątem specyfiki tego filtra. Zwiększa to stabilność grupowania i odporność na fluktuacje.")
            with col_desc2:
                st.markdown(f"#### ⚙️ Obróbka Wstępna: `{optymalizacja}`")
                st.write(OPISY_PREPROCESSING.get(optymalizacja, ""))

        # =================================================================
        # PRZETWARZANIE DANYCH WEJŚCIOWYCH
        # =================================================================
        krzywe_T = krzywe.T
        
        if optymalizacja == "Analiza trendu":
            krzywe_opt = krzywe.diff(axis=0).fillna(0).T
            scaler = StandardScaler()
            dane_do_algorytmu = scaler.fit_transform(krzywe_opt)
        elif optymalizacja == "FeatureExtraction":
            cechy = pd.DataFrame(index=nazwy_krzywych)
            cechy['Max'] = krzywe.max().values
            cechy['Poz_Max'] = krzywe.idxmax().apply(lambda idx: x.iloc[idx]).values
            cechy['Srednia'] = krzywe.mean().values
            cechy['Std'] = krzywe.std().values
            cechy['Skośność'] = krzywe.skew().values
            cechy['Kurtoza'] = krzywe.kurt().values
            
            fft_amplitudy = np.abs(np.fft.rfft(krzywe, axis=0))
            maks_czestotliwosci = min(4, fft_amplitudy.shape[0])
            for f_idx in range(1, maks_czestotliwosci):
                cechy[f'FFT_Składowa_{f_idx}'] = fft_amplitudy[f_idx, :]
            
            scaler = StandardScaler()
            dane_do_algorytmu = scaler.fit_transform(cechy)
            
        elif optymalizacja == "MinMaxScaler":
            scaler = MinMaxScaler()
            dane_do_algorytmu = scaler.fit_transform(krzywe_T)
        elif optymalizacja == "Filtrowanie szumów":
            krzywe_smooth = krzywe.rolling(window=5, center=True, min_periods=1).mean()
            scaler = StandardScaler()
            dane_do_algorytmu = scaler.fit_transform(krzywe_smooth.T)
        else:
            scaler = StandardScaler()
            dane_do_algorytmu = scaler.fit_transform(krzywe_T)
            
        # =================================================================
        # KLASTERYZACJA ALGORYTMAMI
        # =================================================================
        if metoda == "K-means":
            model = KMeans(n_clusters=liczba_grup, random_state=42, n_init=10)
            numery_grup = model.fit_predict(dane_do_algorytmu) + 1
            
        elif "PSO" in metoda:
            with st.spinner("🐝 Trwa symulacja lotu roju cząstek (PSO)..."):
                model_pso = PSOClustering(n_clusters=liczba_grup, random_state=42)
                numery_grup = model_pso.fit_predict(dane_do_algorytmu)

        elif "NMF" in metoda:
            with st.spinner("📐 Trwa faktoryzacja macierzy (NMF)..."):
                if (dane_do_algorytmu < 0).any():
                    scaler_nmf = MinMaxScaler()
                    dane_nmf = scaler_nmf.fit_transform(dane_do_algorytmu)
                else:
                    dane_nmf = dane_do_algorytmu
                
                model_nmf = NMF(n_components=liczba_grup, init='nndsvd', random_state=42, max_iter=500)
                W = model_nmf.fit_transform(dane_nmf)
                numery_grup = np.argmax(W, axis=1) + 1
            
        elif metoda == "GMM":
            model = GaussianMixture(n_components=liczba_grup, random_state=42, n_init=5)
            numery_grup = model.fit_predict(dane_do_algorytmu) + 1

        elif "BGMM" in metoda:
            with st.spinner("🔮 Trwa wnioskowanie bayesowskie (BGMM)..."):
                model_bgmm = BayesianGaussianMixture(
                    n_components=liczba_grup, 
                    covariance_type='diag', 
                    weight_concentration_prior=1e-3, 
                    random_state=42, 
                    n_init=5
                )
                numery_grup = model_bgmm.fit_predict(dane_do_algorytmu) + 1
            
        elif "metoda Warda" in metoda:
            powiazania = linkage(dane_do_algorytmu, method='ward')
            numery_grup = fcluster(powiazania, t=liczba_grup, criterion='maxclust')

        elif "Korelacyjna" in metoda:
            powiazania = linkage(dane_do_algorytmu, method='average', metric='correlation')
            numery_grup = fcluster(powiazania, t=liczba_grup, criterion='maxclust')
            
        elif "HDBSCAN" in metoda:
            model = HDBSCAN(min_cluster_size=min_wielkosc, min_samples=1)
            klastry_raw = model.fit_predict(dane_do_algorytmu)
            numery_grup = [n + 1 if n >= 0 else 0 for n in klastry_raw]
            
        elif "Spectral" in metoda:
            model = SpectralClustering(n_clusters=liczba_grup, random_state=42, assign_labels='discretize')
            numery_grup = model.fit_predict(dane_do_algorytmu) + 1
            
        elif "K-Shape" in metoda:
            dataset = to_time_series_dataset(dane_do_algorytmu)
            model = KShape(n_clusters=liczba_grup, random_state=42)
            numery_grup = model.fit_predict(dataset) + 1
            
        elif "DEC" in metoda:
            with st.spinner("🧠 Trwa trening sieci neuronowej (DEC)..."):
                X_tensor = torch.FloatTensor(dane_do_algorytmu)
                input_dim = dane_do_algorytmu.shape[1]
                net = AutoencoderKrzywych(input_dim=input_dim, latent_dim=4)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(net.parameters(), lr=0.01)
                
                net.train()
                for epoch in range(150):
                    optimizer.zero_grad()
                    latent, reconstructed = net(X_tensor)
                    loss = criterion(reconstructed, X_tensor)
                    loss.backward()
                    optimizer.step()
                
                net.eval()
                with torch.no_grad():
                    kodowanie_cech, _ = net(X_tensor)
                    dane_skalowane_przez_siec = kodowanie_cech.numpy()
                
                model_dec = KMeans(n_clusters=liczba_grup, random_state=42, n_init=10)
                numery_grup = model_dec.fit_predict(dane_skalowane_przez_siec) + 1

        elif "ADEC" in metoda:
            with st.spinner("⚔️ Trwa pojedynek sieci neuronowych (ADEC)..."):
                X_tensor = torch.FloatTensor(dane_do_algorytmu)
                N_samples = dane_do_algorytmu.shape[0]
                input_dim = dane_do_algorytmu.shape[1]
                latent_dim = 4
                
                autoencoder = AutoencoderKrzywych(input_dim=input_dim, latent_dim=latent_dim)
                discriminator = DiscriminatorADEC(latent_dim=latent_dim)
                
                criterion_recon = nn.MSELoss()
                criterion_gan = nn.BCELoss()
                
                opt_ae = optim.Adam(autoencoder.parameters(), lr=0.01)
                opt_disc = optim.Adam(discriminator.parameters(), lr=0.005)
                
                for epoch in range(150):
                    autoencoder.train()
                    opt_ae.zero_grad()
                    latent, reconstructed = autoencoder(X_tensor)
                    loss_recon = criterion_recon(reconstructed, X_tensor)
                    loss_recon.backward()
                    opt_ae.step()
                    
                    discriminator.train()
                    opt_disc.zero_grad()
                    
                    real_distribution = torch.randn(N_samples, latent_dim)
                    labels_real = torch.ones(N_samples, 1)
                    labels_fake = torch.zeros(N_samples, 1)
                    
                    out_real = discriminator(real_distribution)
                    loss_d_real = criterion_gan(out_real, labels_real)
                    
                    latent, _ = autoencoder(X_tensor)
                    out_fake = discriminator(latent.detach())
                    loss_d_fake = criterion_gan(out_fake, labels_fake)
                    
                    loss_d = loss_d_real + loss_d_fake
                    loss_d.backward()
                    opt_disc.step()
                    
                    autoencoder.train()
                    opt_ae.zero_grad()
                    latent, _ = autoencoder(X_tensor)
                    out_g = discriminator(latent)
                    
                    loss_g = criterion_gan(out_g, labels_real)
                    loss_g.backward()
                    opt_ae.step()
                
                autoencoder.eval()
                with torch.no_grad():
                    kodowanie_adec, _ = autoencoder(X_tensor)
                    dane_oczyszczone_adec = kodowanie_adec.numpy()
                
                model_adec = KMeans(n_clusters=liczba_grup, random_state=42, n_init=10)
                numery_grup = model_adec.fit_predict(dane_oczyszczone_adec) + 1

        elif "RDEC" in metoda:
            with st.spinner("🛡️ Trwa trening sieci z barierą regularyzacji (RDEC)..."):
                X_tensor = torch.FloatTensor(dane_do_algorytmu)
                input_dim = dane_do_algorytmu.shape[1]
                net = AutoencoderKrzywych(input_dim=input_dim, latent_dim=4)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
                
                net.train()
                for epoch in range(150):
                    optimizer.zero_grad()
                    latent, reconstructed = net(X_tensor)
                    loss_recon = criterion(reconstructed, X_tensor)
                    penalty_latent = torch.mean(torch.norm(latent, p=2, dim=1))
                    loss = loss_recon + 0.01 * penalty_latent
                    loss.backward()
                    optimizer.step()
                
                net.eval()
                with torch.no_grad():
                    kodowanie_rdec, _ = net(X_tensor)
                    dane_stabilne_rdec = kodowanie_rdec.numpy()
                
                model_rdec = KMeans(n_clusters=liczba_grup, random_state=42, n_init=10)
                numery_grup = model_rdec.fit_predict(dane_stabilne_rdec) + 1

        elif "ADClust" in metoda:
            with st.spinner("🤖 Trwa inteligentny trening ADClust. Sieć neuronowa sama ustala liczbę grup..."):
                X_tensor = torch.FloatTensor(dane_do_algorytmu)
                input_dim = dane_do_algorytmu.shape[1]
                net = AutoencoderKrzywych(input_dim=input_dim, latent_dim=4)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(net.parameters(), lr=0.01)
                
                net.train()
                for epoch in range(120):
                    optimizer.zero_grad()
                    latent, reconstructed = net(X_tensor)
                    loss = criterion(reconstructed, X_tensor)
                    loss.backward()
                    optimizer.step()
                
                net.eval()
                with torch.no_grad():
                    latent_features, _ = net(X_tensor)
                    dane_ukryte = latent_features.numpy()
                
                najlepsze_k = 2
                najwyzszy_wynik = -1
                
                for k_test in range(2, 9):
                    km_test = KMeans(n_clusters=k_test, random_state=42, n_init=5)
                    etykiety_test = km_test.fit_predict(dane_ukryte)
                    score = silhouette_score(dane_ukryte, etykiety_test)
                    if score > najwyzszy_wynik:
                        najwyzszy_wynik = score
                        najlepsze_k = k_test
                
                model_adclust = KMeans(n_clusters=najlepsze_k, random_state=42, n_init=10)
                numery_grup = model_adclust.fit_predict(dane_ukryte) + 1
                st.success(f"✨ Sieć ADClust automatycznie ustaliła, że optymalna liczba grup to: **{najlepsze_k}**")

        # Przygotowanie tabeli wynikowej
        wyniki = pd.DataFrame({
            'Krzywa': nazwy_krzywych,
            'Numer Grupy': numery_grup
        }).sort_values(by='Numer Grupy')
        
        # Sekcja: METODA ŁOKCIA (Dostępna dla wszystkich manualnych K)
        if "HDBSCAN" not in metoda and "ADClust" not in metoda:
            with st.expander("🔍 Podpowiedź matematyczna (Metoda Łokcia)"):
                st.write("Poniższy wykres inercji pomaga dobrać optymalną liczbę grup (K) dla aktualnie przygotowanych danych pomiarowych.")
                inercja = []
                zakres_k = range(2, 11)
                for k in zakres_k:
                    km = KMeans(n_clusters=k, random_state=42, n_init=5)
                    km.fit(dane_do_algorytmu)
                    inercja.append(km.inertia_)
                
                fig_elbow, ax_elbow = plt.subplots(figsize=(10, 3))
                ax_elbow.plot(zakres_k, inercja, 'ro-', linewidth=2)
                ax_elbow.set_xlabel('Liczba grup (K)')
                ax_elbow.set_ylabel('Inercja (Suma odległości)')
                ax_elbow.set_xticks(list(zakres_k))
                ax_elbow.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_elbow)
                plt.close(fig_elbow)

        # Prezentacja graficzna
        col_wykres, col_tabela = st.columns([3, 1])
        
        with col_wykres:
            st.subheader("📈 Wykres")
            fig, ax = plt.subplots(figsize=(10, 5))
            cmap = plt.get_cmap('tab10')
            
            if "Hierarchiczna" in metoda:
                if "metoda Warda" in metoda:
                    powiazania_tree = linkage(dane_do_algorytmu, method='ward')
                    ax.set_title("Dendrogram (Metoda Warda - Odległość Euklidesowa)")
                else:
                    powiazania_tree = linkage(dane_do_algorytmu, method='average', metric='correlation')
                    ax.set_title("Dendrogram (Metoda Średnich - Odległość Korelacyjna)")
                    
                dendrogram(powiazania_tree, labels=nazwy_krzywych, leaf_rotation=90, leaf_font_size=9, ax=ax)
            else:
                for i, kolumna in enumerate(krzywe.columns):
                    g = numery_grup[i]
                    if g == 0:
                        ax.plot(x, krzywe[kolumna], color='gray', linestyle=':', alpha=0.4, linewidth=1)
                    else:
                        ax.plot(x, krzywe[kolumna], color=cmap((g - 1) % 10), alpha=0.6, linewidth=1)
                ax.set_title(f"Metoda: {metoda} | Przygotowanie: {optymalizacja}")
                ax.grid(True, linestyle='--', alpha=0.5)
                
            st.pyplot(fig)
            plt.close(fig)
            
        with col_tabela:
            st.subheader("📋 Grupy")
            st.dataframe(wyniki, use_container_width=True, hide_index=True, height=300)
            
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                wyniki.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 Pobierz Excel",
                data=buffer.getvalue(),
                file_name=f"wyniki_{metoda.lower().split()[0]}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # Raport tekstowy
        st.write("---")
        st.subheader("📝 Podsumowanie tekstowe grup")
        unikalne_grupy = sorted(wyniki['Numer Grupy'].unique())
        
        for g in unikalne_grupy:
            krzywe_w_grupie = wyniki[wyniki['Numer Grupy'] == g]['Krzywa'].tolist()
            lista_str = ", ".join(krzywe_w_grupie)
            
            if g == 0:
                st.markdown(f"🔴 **Szum / Anomalie pomiarowe** ({len(krzywe_w_grupie)} krzywych):")
            else:
                st.markdown(f"🟢 **Grupa {g}** ({len(krzywe_w_grupie)} krzywych):")
            st.code(lista_str, language="")
            
    except Exception as ogolny_blad:
        st.error(f"Problem z przetworzeniem danych: {ogolny_blad}")
else:
    st.info("💡 Aby rozpocząć, wgraj plik z dysku lub wklej link do Google Sheets powyżej.")
