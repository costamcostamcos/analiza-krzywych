import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import io

# Bezpieczny import dla zaawansowanego algorytmu K-Shape
try:
    from tslearn.clustering import KShape
    from tslearn.utils import to_time_series_dataset
    tslearn_dostepne = True
except ImportError:
    tslearn_dostepne = False

# Bezpieczny import dla Głębokiego Uczenia (DEC / PyTorch)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    pytorch_dostepne = True
except ImportError:
    pytorch_dostepne = False

# =================================================================
# KLASA SIECI NEURONOWEJ (AUTOENKODER DLA METODY DEC)
# =================================================================
if pytorch_dostepne:
    class AutoencoderKrzywych(nn.Module):
        def __init__(self, input_dim, latent_dim=4):
            super(AutoencoderKrzywych, self).__init__()
            # Encoder - kompresuje surowy wykres do 4 kluczowych cech
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, latent_dim)
            )
            # Decoder - próbuje odtworzyć oryginalny wykres z tych 4 cech
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim)
            )
        def forward(self, x):
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return latent, reconstructed

# Konfiguracja strony
st.set_page_config(page_title="Analizator Krzywych Pro", layout="wide")

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
    st.caption("⚠️ Ważne: Arkusz w opcjach udostępniania musi mieć ustawione: 'Każdy użytkownik posiadający link może przeglądać'")
    
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
        
        # Dynamiczne budowanie listy metod
        lista_metod = [
            "K-means", 
            "Hierarchiczna (Euklidesowa)", 
            "HDBSCAN (Gęstościowa - Auto K)", 
            "GMM (Probabilistyczna)", 
            "Spectral Clustering"
        ]
        if tslearn_dostepne:
            lista_metod.append("K-Shape (Kształt fali)")
        if pytorch_dostepne:
            lista_metod.append("DEC (Głębokie Uczenie - Sieć Neuronowa)")
        
        # Wybór parametrów - 3 kolumny
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            metoda = st.selectbox("Wybierz metodę główną:", lista_metod)
        
        with col_param2:
            # Metody zaawansowane (K-Shape i DEC) mają wbudowane specyficzne przygotowanie danych
            if "K-Shape" not in metoda and "DEC" not in metoda:
                optymalizacja = st.selectbox(
                    "Wybierz wstępne przygotowanie danych:", 
                    ["Standardowa", "Analiza trendu", "FeatureExtraction", "MinMaxScaler", "Filtrowanie szumów"]
                )
            else:
                optymalizacja = "Standardowa"
                st.selectbox("Optymalizacja wbudowana w algorytm", ["Wbudowana (Auto-Embedding)"], disabled=True)
                
        with col_param3:
            if "HDBSCAN" in metoda:
                min_wielkosc = st.slider("Minimalna wielkość grupy (Min Cluster Size):", min_value=2, max_value=10, value=3)
            else:
                liczba_grup = st.slider("Wybierz oczekiwaną liczbę grup (K):", min_value=2, max_value=10, value=4)
            
        # Alerty o brakujących bibliotekach
        if not tslearn_dostepne or not pytorch_dostepne:
            brakujace = []
            if not tslearn_dostepne: brakujace.append("`tslearn` (dla K-Shape)")
            if not pytorch_dostepne: brakujace.append("`torch` (dla DEC)")
            st.info(f"💡 Aby odblokować wszystkie metody AI, dopisz do swojego `requirements.txt`: {', '.join(brakujace)}")

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
            
        elif metoda == "Hierarchiczna (Euklidesowa)":
            powiazania = linkage(dane_do_algorytmu, method='ward')
            numery_grup = fcluster(powiazania, t=liczba_grup, criterion='maxclust')
            
        elif "HDBSCAN" in metoda:
            model = HDBSCAN(min_cluster_size=min_wielkosc, min_samples=1)
            klastry_raw = model.fit_predict(dane_do_algorytmu)
            numery_grup = [n + 1 if n >= 0 else 0 for n in klastry_raw]
            
        elif "GMM" in metoda:
            model = GaussianMixture(n_components=liczba_grup, random_state=42, n_init=5)
            numery_grup = model.fit_predict(dane_do_algorytmu) + 1
            
        elif "Spectral" in metoda:
            model = SpectralClustering(n_clusters=liczba_grup, random_state=42, assign_labels='discretize')
            numery_grup = model.fit_predict(dane_do_algorytmu) + 1
            
        elif "K-Shape" in metoda:
            dataset = to_time_series_dataset(dane_do_algorytmu)
            model = KShape(n_clusters=liczba_grup, random_state=42)
            numery_grup = model.fit_predict(dataset) + 1
            
        elif "DEC" in metoda:
            # PROCES DEEP EMBEDDED CLUSTERING (TRENING W LOCIE)
            with st.spinner("🧠 Trwa trening sieci neuronowej (Autoenkodera)... Proszę czekać."):
                # Zamiana danych na sensory PyTorch
                X_tensor = torch.FloatTensor(dane_do_algorytmu)
                
                # Inicjalizacja sieci
                input_dim = dane_do_algorytmu.shape[1]
                net = AutoencoderKrzywych(input_dim=input_dim, latent_dim=4)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(net.parameters(), lr=0.01)
                
                # Szybki trening sieci (150 epok - ultra szybkie dla małego zestawu)
                net.train()
                for epoch in range(150):
                    optimizer.zero_grad()
                    latent, reconstructed = net(X_tensor)
                    loss = criterion(reconstructed, X_tensor)
                    loss.backward()
                    optimizer.step()
                
                # Wyciągnięcie skompresowanych cech (Latent Space) z sieci
                net.eval()
                with torch.no_grad():
                    kodowanie_cech, _ = net(X_tensor)
                    dane_skalowane_przez_siec = kodowanie_cech.numpy()
                
                # Klasteryzacja K-means na skompresowanych cechach z sieci neuronowej
                model_dec = KMeans(n_clusters=liczba_grup, random_state=42, n_init=10)
                numery_grup = model_dec.fit_predict(dane_skalowane_przez_siec) + 1
            
        # Tabela wynikowa
        wyniki = pd.DataFrame({
            'Krzywa': nazwy_krzywych,
            'Numer Grupy': numery_grup
        }).sort_values(by='Numer Grupy')
        
        # Metoda Łokcia
        if metoda == "K-means":
            with st.expander("🔍 Podpowiedź matematyczna (Metoda Łokcia)"):
                inercja = []
                zakres_k = range(2, 11)
                for k in zakres_k:
                    km = KMeans(n_clusters=k, random_state=42, n_init=5)
                    km.fit(dane_do_algorytmu)
                    inercja.append(km.inertia_)
                
                fig_elbow, ax_elbow = plt.subplots(figsize=(10, 3))
                ax_elbow.plot(zakres_k, inercja, 'ro-', linewidth=2)
                ax_elbow.set_xlabel('Liczba grup (K)')
                ax_elbow.set_ylabel('Inercja')
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
                powiazania_tree = linkage(dane_do_algorytmu, method='ward')
                dendrogram(powiazania_tree, labels=nazwy_krzywych, leaf_rotation=90, leaf_font_size=9, ax=ax)
                ax.set_title("Drzewo Podobieństwa (Dendrogram)")
            else:
                for i, kolumna in enumerate(krzywe.columns):
                    g = numery_grup[i]
                    if g == 0:
                        ax.plot(x, krzywe[kolumna], color='gray', linestyle=':', alpha=0.4, linewidth=1)
                    else:
                        ax.plot(x, krzywe[kolumna], color=cmap((g - 1) % 10), alpha=0.6, linewidth=1)
                ax.set_title(f"Metoda: {metoda}")
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
