import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import io

# Konfiguracja nagłówka strony (Szeroki układ)
st.set_page_config(page_title="Analizator Krzywych", layout="wide")
st.title("📊 Interaktywny Analizator i Grupowanie Krzywych")
st.write("Wgraj swój plik Excel, wybierz parametry i przeanalizuj kształty swoich wykresów.")

# PANEL BOCZNY (Ustawienia)
st.sidebar.header("⚙️ Ustawienia analizy")

# 1. Wgrywanie pliku
uploaded_file = st.sidebar.file_uploader("Wgraj plik Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    # Wczytanie danych
    df = pd.read_excel(uploaded_file)
    x = df.iloc[:, 0]
    krzywe = df.iloc[:, 1:]
    nazwy_krzywych = krzywe.columns.tolist()
    
    # Transpozycja i skalowanie danych
    krzywe_T = krzywe.T
    scaler = StandardScaler()
    krzywe_skalowane = scaler.fit_transform(krzywe_T)
    
    # 2. Wybór metody i liczby grup
    metoda = st.sidebar.selectbox("Wybierz metodę grupowania:", ["K-means", "Hierarchiczna (Euklidesowa)"])
    liczba_grup = st.sidebar.slider("Wybierz liczbę grup (K):", min_value=2, max_value=10, value=4)
    
    # OBLICZENIA GŁÓWNE
    if metoda == "K-means":
        model = KMeans(n_clusters=liczba_grup, random_state=42, n_init=10)
        numery_grup = model.fit_predict(krzywe_skalowane) + 1
    else:
        powiazania = linkage(krzywe_skalowane, method='ward')
        numery_grup = fcluster(powiazania, t=liczba_grup, criterion='maxclust')
        
    # Uproszczone nazwy kolumn, aby uniknąć błędów renderowania w przeglądarce
    wyniki = pd.DataFrame({
        'Krzywa': nazwy_krzywych,
        'Numer Grupy': numery_grup
    }).sort_values(by='Numer Grupy')
    
    # =================================================================
    # NOWOŚĆ: SEKCJA PODPOWIEDZI MATEMATYCZNEJ (METODA ŁOKCIA)
    # =================================================================
    if metoda == "K-means":
        with st.expander("🔍 Podpowiedź matematyczna: Ile grup wybrać? (Metoda Łokcia)"):
            st.write("Poniższy wykres pokazuje, jak zmienia się spójność grup przy różnej liczbie klastrów. "
                     "**Szukaj punktu, w którym linia gwałtownie się załamuje (tworzy 'łokieć').** "
                     "To optymalna matematycznie liczba grup dla Twoich danych.")
            
            # Obliczanie inercji dla K od 2 do 10
            inercja = []
            zakres_k = range(2, 11)
            for k in zakres_k:
                km = KMeans(n_clusters=k, random_state=42, n_init=5)
                km.fit(krzywe_skalowane)
                inercja.append(km.inertia_)
            
            # Rysowanie małego wykresu łokcia
            fig_elbow, ax_elbow = plt.subplots(figsize=(10, 3))
            ax_elbow.plot(zakres_k, inercja, 'ro-', linewidth=2, markersize=6)
            ax_elbow.set_xlabel('Liczba grup (K)')
            ax_elbow.set_ylabel('Inercja (Suma odległości)')
            ax_elbow.set_xticks(list(zakres_k))
            ax_elbow.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig_elbow)
            plt.close(fig_elbow) # Czyszczenie pamięci podręcznej wykresu
    # =================================================================

    # GŁÓWNY PANEL APLIKACJI (ZOPTYMALIZOWANY UKŁAD)
    col1, col2 = st.columns([3.5, 1])
    
    with col1:
        st.subheader("📈 Wykres i wizualizacja grup")
        fig, ax = plt.subplots(figsize=(12, 5.5))
        
        if metoda == "K-means":
            cmap = plt.get_cmap('tab10')
            for i, kolumna in enumerate(krzywe.columns):
                g = numery_grup[i] - 1
                ax.plot(x, krzywe[kolumna], color=cmap(g), alpha=0.6, linewidth=1)
            ax.set_title(f"Krzywe podzielone na {liczba_grup} grupy (K-means)")
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            dendrogram(powiazania, labels=nazwy_krzywych, leaf_rotation=90, leaf_font_size=10, ax=ax)
            ax.set_title("Drzewo Podobieństwa (Dendrogram)")
            
        st.pyplot(fig)
        plt.close(fig)
        
    with col2:
        st.subheader("📋 Przypisanie")
        
        # Tabela z ograniczoną wysokością
        st.dataframe(wyniki, use_container_width=True, hide_index=True, height=380)
        
        # Generowanie pliku Excel do pobrania w pamięci RAM
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            wyniki.to_excel(writer, index=False)
        
        st.download_button(
            label="📥 Pobierz wyniki jako Excel",
            data=buffer.getvalue(),
            file_name=f"wyniki_{metoda.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
else:
    st.info("💡 Aby rozpocząć, wgraj swój plik Excel (`dane.xlsx`) za pomocą panelu po lewej stronie.")
