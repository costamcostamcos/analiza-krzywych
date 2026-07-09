# -*- coding: utf-8 -*-
"""
Moduł klasyfikacji regułowej widm EPR (Marciniak et al. 2025, Fig. 1A/1B)
dla Interaktywnego Analizatora Krzywych AI Pro.

Architektura:
  - logika obliczeniowa: klasyfikator_epr_marciniak2025.py (bez zależności od UI)
  - ten plik: wyłącznie warstwa Streamlit/Plotly

Integracja w app.py:
    from modul_klasyfikacji_regulowej import renderuj_klasyfikacje_regulowa
    ...
    with tab_klasyfikacja_regulowa:
        renderuj_klasyfikacje_regulowa(df_widma)

Oczekiwany format danych (zgodny z pipeline'em ujednoliconej osi X):
  df_widma: DataFrame, w którym indeks/kolumna 'g' to wspólna oś g-factor,
  a każda pozostała kolumna to jedno widmo (nazwa kolumny = nazwa próbki).
  Jeśli oś masz w mT, użyj przelicznika g = h*f / (mu_B * B) przed wywołaniem.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from klasyfikator_epr_marciniak2025 import (
    classify_nonirradiated,
    classify_irradiated,
    value_at_g,
)

# Punkty decyzyjne do zaznaczenia na wykresie
_PUNKTY_G = [2.0000, 2.0043, 2.0171]
_ZAKRESY_G = {
    "min III/V": (2.0001, 2.0040),
    "max II": (2.0200, 2.0250),
    "min I/IVA": (1.9930, 1.9990),
}


def _klasyfikuj_wszystkie(df, kolumna_g, tryb, smooth_window, prominence):
    """Czysta funkcja obliczeniowa – zwraca DataFrame wyników."""
    g = df[kolumna_g].to_numpy(dtype=float)
    fn = classify_nonirradiated if tryb == "0 Gy" else classify_irradiated

    wyniki = []
    for kol in df.columns:
        if kol == kolumna_g:
            continue
        y = df[kol].to_numpy(dtype=float)
        try:
            typ, sciezka = fn(g, y, smooth_window=smooth_window,
                              prominence=prominence)
            wyniki.append({
                "Próbka": kol,
                "Typ": typ,
                "f(2.0000)": value_at_g(g, y, 2.0000),
                "f(2.0171)": value_at_g(g, y, 2.0171),
                "Ścieżka decyzyjna": " → ".join(sciezka),
            })
        except ValueError as e:
            wyniki.append({"Próbka": kol, "Typ": "BŁĄD",
                           "Ścieżka decyzyjna": str(e)})
    return pd.DataFrame(wyniki)


def _wykres_z_punktami_decyzyjnymi(df, kolumna_g, wybrane_probki):
    """Widma z zaznaczonymi punktami i zakresami decyzyjnymi."""
    fig = go.Figure()
    g = df[kolumna_g]
    for kol in wybrane_probki:
        fig.add_trace(go.Scatter(x=g, y=df[kol], mode="lines", name=kol))

    for g0 in _PUNKTY_G:
        fig.add_vline(x=g0, line_dash="dot", line_color="gray",
                      annotation_text=f"g={g0}", annotation_font_size=10)
    for nazwa, (lo, hi) in _ZAKRESY_G.items():
        fig.add_vrect(x0=lo, x1=hi, fillcolor="lightblue", opacity=0.15,
                      line_width=0, annotation_text=nazwa,
                      annotation_font_size=9)

    fig.update_layout(
        xaxis_title="g-factor",
        yaxis_title="Sygnał EPR (a.u.)",
        xaxis_autorange="reversed",  # konwencja EPR: g maleje w prawo
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=30),
    )
    return fig


def renderuj_klasyfikacje_regulowa(df_widma: pd.DataFrame,
                                   kolumna_g: str = "g"):
    """Główna funkcja renderująca zakładkę klasyfikacji regułowej."""
    st.subheader("Klasyfikacja regułowa (Marciniak et al. 2025)")

    with st.expander("Parametry klasyfikacji", expanded=False):
        c1, c2, c3 = st.columns(3)
        tryb = c1.radio("Tryb widm", ["0 Gy", "napromienione"],
                        help="Wybiera drzewo decyzyjne: Fig. 1A lub 1B")
        smooth_window = c2.slider("Okno wygładzania", 3, 21, 7, step=2)
        prominence = c3.number_input(
            "Prominencja ekstremów (a.u.)", min_value=0.0, value=0.0,
            format="%.2e",
            help="Próg odsiewający ekstrema szumowe; sugerowane "
                 "~3σ szumu linii bazowej")

    if st.button("Klasyfikuj widma", type="primary"):
        st.session_state["wyniki_regulowe"] = _klasyfikuj_wszystkie(
            df_widma, kolumna_g, tryb, smooth_window, prominence)

    wyniki = st.session_state.get("wyniki_regulowe")
    if wyniki is None:
        st.info("Ustaw parametry i uruchom klasyfikację.")
        return

    # tabela wyników z podsumowaniem liczności typów
    st.dataframe(wyniki, use_container_width=True, hide_index=True)
    licznosci = wyniki["Typ"].value_counts()
    st.caption("Liczność typów: " +
               ", ".join(f"{t}: {n}" for t, n in licznosci.items()))

    with st.expander("Podgląd widm z punktami decyzyjnymi", expanded=False):
        probki = [k for k in df_widma.columns if k != kolumna_g]
        wybrane = st.multiselect("Próbki do wyświetlenia", probki,
                                 default=probki[:3])
        if wybrane:
            st.plotly_chart(
                _wykres_z_punktami_decyzyjnymi(df_widma, kolumna_g, wybrane),
                use_container_width=True)

    return wyniki  # do wykorzystania w eksporcie Excel / porównaniu z c-means
