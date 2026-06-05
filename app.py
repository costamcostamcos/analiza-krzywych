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

# Konfiguracja strony
st.set_page_config(page_title="Analizator Krzywych", layout="wide")

st.title("📊 Interaktywny Analizator Krzywych Pro")
st.write("Wgraj plik Excel lub wklej link do Google Sheets. Aplikacja automatycznie odnajdzie start tabeli.")

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
        
        # Dynamiczne budowanie listy metod na podstawie dostępności bibliotek
        lista_metod = [
            "K-means", 
            "Hierarchiczna (Euklidesowa)", 
            "HDBSCAN (Gęstościowa - Auto K)", 
            "GMM (Probabilistyczna)", 
            "Spectral Clustering"
        ]
        if tslearn_dostepne:
            lista_metod.append("K-Shape (Kształt fali)")
        
        # Wybór parametrów - 3 kolumny
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            metoda = st.selectbox("Wybierz metodę główną:", lista_metod)
        
        with col_param2:
            # Optymalizacje matematyczne wyłączamy dla K-Shape, bo ono ma własny system korelacji
            if "K-Shape" not in metoda:
                optymalizacja = st.selectbox(
                    "Wybierz wstępne przygotowanie danych:", 
                    ["Standardowa", "Analiza trendu", "FeatureExtraction", "MinMaxScaler", "Filtrowanie szumów"]
                )
            else:
                optymalizacja = "Standardowa"
                st.selectbox("Optymalizacja wbudowana w algorytm", ["Korelacja kształtu"], disabled=True)
                
        with col_param3:
            # Dostosowanie suwaka do specyfiki algorytmu
            if "HDBSCAN" in metoda:
                min_wielkosc = st.slider("Minimalna wielkość grupy (Min Cluster Size):", min_value=2, max_value=10, value=3)
            else:
                liczba_grup = st.slider("Wybierz oczekiwaną liczbę grup (K):", min_value=2, max_value=10, value=4)
            
        # Informacja o K-Shape, jeśli biblioteka nie jest jeszcze wgrana
        if not tslearn_dostepne:
            st.info("💡 Chcesz odblokować metodę **K-Shape (Kształt fali)**? Dopisz `tslearn` do swojego pliku `requirements.txt` na GitHubie!")

        # =================================================================
        # PRZETWARZANIE DANYCH
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
        # REALIZACJA NOWOCZESNYCH METOD KLASTERYZACJI
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
            # HDBSCAN zwraca -1 dla szumu/anomalii. Mapujemy: -1 zostaje jako 0 (Szum), reszta dostaje +1
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
            
        # Przygotowanie tabeli wynikowej
        wyniki = pd.DataFrame({
            'Krzywa': nazwy_krzywych,
            'Numer Grupy': numery_grup
        }).sort_values(by='Numer Grupy')
        
        # Metoda Łokcia (Tylko dla klasycznego K-means)
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

        # Prezentacja wyników graficznych i tabeli
        col_wykres, col_tabela = st.columns([3, 1])
        
        with col_wykres:
            st.subheader("📈 Wykres")
            fig, ax = plt.subplots(figsize=(10, 5))
            cmap = plt.get_cmap('tab10')
            
            if "Hierarchiczna" in metoda:
                # Dla hierarchicznej rysujemy tradycyjne drzewo powiązań
                powiazania_tree = linkage(dane_do_algorytmu, method='ward')
                dendrogram(powiazania_tree, labels=nazwy_krzywych, leaf_rotation=90, leaf_font_size=9, ax=ax)
                ax.set_title("Drzewo Podobieństwa (Dendrogram)")
            else:
                # Dla wszystkich pozostałych metod rysujemy pogrupowane krzywe pomiarowe
                for i, kolumna in enumerate(krzywe.columns):
                    g = numery_grup[i]
                    if g == 0:
                        # Szum/Anomalie w HDBSCAN rysujemy jako przerywane, szare linie
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

        # Raport tekstowy na dole strony
        st.write("---")
        st.subheader("📝 Podsumowanie tekstowe grup")
        
        unikalne_grupy = sorted(wyniki['Numer Grupy'].unique())
        
        for g in unikalne_grupy:
            krzywe_w_grupie = wyniki[wyniki['Numer Grupy'] == g]['Krzywa'].tolist()
            lista_str = ", ".join(krzywe_w_grupie)
            
            if g == 0:
                st.markdown(f"🔴 **Szum / Anomalie pomiarowe** ({len(krzywe_w_grupie)} krzywych) - odrzucone przez algorytm jako niepasujące:")
            else:
                st.markdown(f"🟢 **Grupa {g}** ({len(krzywe_w_grupie)} krzywych):")
            st.code(lista_str, language="")
            
    except Exception as ogolny_blad:
        st.error(f"Problem z przetworzeniem danych: {ogolny_blad}")
else:
    st.info("💡 Aby rozpocząć, wgraj plik z dysku lub wklej link do Google Sheets powyżej.")
