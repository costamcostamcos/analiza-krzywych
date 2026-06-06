import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
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
# ARCHITEKTURY SIECI NEURONOWYCH (DEC, ADEC, RDEC)
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
        
        # Budowanie listy dostępnych metod
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
            lista_metod.append("ADEC (Adwersarialne Głębokie Uczenie)")
            lista_metod.append("RDEC (Regularizowane Głębokie Uczenie)")
            lista_metod.append("ADClust (Automatyczne Głębokie Uczenie - Nowość!)")
        
        # Układ parametrów - 3 kolumny
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            metoda = st.selectbox("Wybierz metodę główną:", lista_metod)
        
        with col_param2:
            if "K-Shape" not in metoda and "DEC" not in metoda and "RDEC" not in metoda and "ADClust" not in metoda:
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
            elif "ADClust" in metoda:
                st.slider("Liczba grup dobierana automatycznie przez AI", min_value=0, max_value=0, value=0, disabled=True)
            else:
                liczba_grup = st.slider("Wybierz oczekiwaną liczbę grup (K):", min_value=2, max_value=10, value=4)

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
            # PROCES AUTOMATYCZNEGO GŁĘBOKIEGO KLASTEROWANIA (ADClust)
            with st.spinner("🤖 Trwa inteligentny trening ADClust. Sieć neuronowa sama ustala liczbę grup..."):
                X_tensor = torch.FloatTensor(dane_do_algorytmu)
                input_dim = dane_do_algorytmu.shape[1]
                net = AutoencoderKrzywych(input_dim=input_dim, latent_dim=4)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(net.parameters(), lr=0.01)
                
                # 1. Wstępny trening enkodera, by ułożył dane
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
                
                # 2. Automatyczne poszukiwanie optymalnego K za pomocą metryki Silhouette wewnątrz sieci
                najlepsze_k = 2
                najwyzszy_wynik = -1
                
                # Skanujemy potencjalne podziały od 2 do 8 grup
                for k_test in range(2, 9):
                    km_test = KMeans(n_clusters=k_test, random_state=42, n_init=5)
                    etykiety_test = km_test.fit_predict(dane_ukryte)
                    score = silhouette_score(dane_ukryte, etykiety_test)
                    if score > najwyzszy_wynik:
                        najwyzszy_wynik = score
                        najlepsze_k = k_test
                
                # 3. Finalny podział na optymalnej liczbie grup wyznaczonej przez AI
                model_adclust = KMeans(n_clusters=najlepsze_k, random_state=42, n_init=10)
                numery_grup = model_adclust.fit_predict(dane_ukryte) + 1
                st.success(f"✨ Sieć ADClust automatycznie ustaliła, że optymalna liczba grup to: **{najlepsze_k}**")

        # Przygotowanie tabeli wynikowej
        wyniki = pd.DataFrame({
            'Krzywa': nazwy_krzywych,
            'Numer Grupy': numery_grup
        }).sort_values(by='Numer Grupy')
        
        # Sekcja: METOTA ŁOKCIA (Tylko dla K-means)
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
