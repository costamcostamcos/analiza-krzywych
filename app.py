import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import io

# Konfiguracja strony - layout dopasowujący się do urządzeń
st.set_page_config(page_title="Analizator Krzywych", layout="wide")

st.title("📊 Interaktywny Analizator Krzywych")
st.write("Wgraj plik Excel lub wklej link do Google Sheets, aby pogrupować wykresy.")

# =================================================================
# NOWA SEKCJA KONFIGURACJI (PROSTA DLA TELEFONÓW I KOMPUTERÓW)
# =================================================================
st.write("### ⚙️ Ustawienia analizy")

# Wybór źródła danych na środku strony
typ_zrodla = st.radio(
    "Wybierz źródło danych:", 
    ["Plik Excel (.xlsx)", "Link do Google Sheets"], 
    horizontal=True
)

df = None

if typ_zrodla == "Plik Excel (.xlsx)":
    uploaded_file = st.file_uploader("Wgraj plik Excel", type=["xlsx"])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
else:
    link_sheets = st.text_input(
        "Wklej link do Google Sheets:", 
        placeholder="https://docs.google.com/spreadsheets/d/..."
    )
    st.caption("⚠️ Ważne: Arkusz w opcjach udostępniania musi mieć ustawione: 'Każdy użytkownik posiadający link może przeglądać'")
    
    if link_sheets:
        try:
            # Trik konwertujący zwykły link Google Sheets na bezpośredni eksport do Excela
            if "docs.google.com/spreadsheets" in link_sheets:
                bazowy_url = link_sheets.split("/edit")[0]
                url_eksportu = f"{bazowy_url}/export?format=xlsx"
                df = pd.read_excel(url_eksportu)
            else:
                st.error("To nie wygląda na poprawny link do Google Sheets.")
        except Exception as e:
            st.error("Nie udało się pobrać danych. Sprawdź, czy arkusz na pewno jest udostępniony dla każdego z linkiem.")

# Jeśli dane zostały poprawnie wczytane z dowolnego źródła
if df is not None:
    try:
        x = df.iloc[:, 0]
        krzywe = df.iloc[:, 1:]
        nazwy_krzywych = krzywe.columns.tolist()
        
        # Przygotowanie danych
        krzywe_T = krzywe.T
        scaler = StandardScaler()
        krzywe_skalowane = scaler.fit_transform(krzywe_T)
        
        # Ustawienia metody i suwaka w dwóch kolumnach (na telefonie będą jedna pod drugą)
        col_param1, col_param2 = st.columns(2)
        with col_param1:
            metoda = st.selectbox("Wybierz metodę grupowania:", ["K-means", "Hierarchiczna (Euklidesowa)"])
        with col_param2:
            liczba_grup = st.slider("Wybierz liczbę grup (K):", min_value=2, max_value=10, value=4)
            
        # Obliczenia
        if metoda == "K-means":
            model = KMeans(n_clusters=liczba_grup, random_state=42, n_init=10)
            numery_grup = model.fit_predict(krzywe_skalowane) + 1
        else:
            powiazania = linkage(krzywe_skalowane, method='ward')
            numery_grup = fcluster(powiazania, t=liczba_grup, criterion='maxclust')
            
        wyniki = pd.DataFrame({
            'Krzywa': nazwy_krzywych,
            'Numer Grupy': numery_grup
        }).sort_values(by='Numer Grupy')
        
        # Metoda Łokcia ukryta w pasku
        if metoda == "K-means":
            with st.expander("🔍 Podpowiedź matematyczna (Metoda Łokcia)"):
                inercja = []
                zakres_k = range(2, 11)
                for k in zakres_k:
                    km = KMeans(n_clusters=k, random_state=42, n_init=5)
                    km.fit(krzywe_skalowane)
                    inercja.append(km.inertia_)
                
                fig_elbow, ax_elbow = plt.subplots(figsize=(10, 3))
                ax_elbow.plot(zakres_k, inercja, 'ro-', linewidth=2)
                ax_elbow.set_xlabel('Liczba grup (K)')
                ax_elbow.set_ylabel('Inercja')
                ax_elbow.set_xticks(list(zakres_k))
                ax_elbow.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(fig_elbow)
                plt.close(fig_elbow)

        # Prezentacja wyników - na dużym ekranie obok siebie, na telefonie w pionie
        col_wykres, col_tabela = st.columns([3, 1])
        
        with col_wykres:
            st.subheader("📈 Wykres")
            fig, ax = plt.subplots(figsize=(10, 5))
            
            if metoda == "K-means":
                cmap = plt.get_cmap('tab10')
                for i, kolumna in enumerate(krzywe.columns):
                    g = numery_grup[i] - 1
                    ax.plot(x, krzywe[kolumna], color=cmap(g), alpha=0.6, linewidth=1)
                ax.set_title(f"Podział na {liczba_grup} grupy")
                ax.grid(True, linestyle='--', alpha=0.5)
            else:
                dendrogram(powiazania, labels=nazwy_krzywych, leaf_rotation=90, leaf_font_size=9, ax=ax)
                ax.set_title("Dendrogram")
                
            st.pyplot(fig)
            plt.close(fig)
            
        with col_tabela:
            st.subheader("📋 Grupy")
            st.dataframe(wyniki, use_container_width=True, hide_index=True, height=300)
            
            # Przygotowanie pliku Excel do pobrania
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                wyniki.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 Pobierz Excel",
                data=buffer.getvalue(),
                file_name=f"wyniki_{metoda.lower().replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    except Exception as ogolny_blad:
        st.error(f"Wykryto problem ze strukturą danych w pliku: {ogolny_blad}")
        st.info("Upewnij się, że pierwsza kolumna to argument (np. czas/X), a kolejne kolumny to Twoje krzywe.")
else:
    st.info("💡 Aby rozpocząć, wgraj plik z dysku lub wklej link do skonfigurowanego arkusza Google Sheets powyżej.")
