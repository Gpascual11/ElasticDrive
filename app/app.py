import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(page_title="ElasticDrive AI", layout="wide", page_icon="⚡")

st.markdown("""
<style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    [data-testid="stSidebar"] { background-color: #1E1E1E; border-right: 1px solid #333; }

    /* Títulos y Texto resaltado */
    .big-yellow {
        color: #FFC107;
        font-size: 3rem !important;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .sub-text {
        color: #AAAAAA;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Botón Premium */
    .stButton > button {
        background: linear-gradient(45deg, #FFC107, #FFD54F);
        color: #000; font-weight: 800; border: none; border-radius: 8px;
        height: 3.5rem; text-transform: uppercase; letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.3); width: 100%;
    }

    /* Métricas */
    [data-testid="stMetricValue"] { color: #FFC107 !important; font-size: 2.8rem !important; font-weight: 700 !important; }
    .stAlert { background-color: #1E1E1E; border: 1px solid #333; border-left: 5px solid #FFC107; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def cargar_recursos():
    try:
        modelo = joblib.load('../models/tier_classifier.pkl')
        metadata = joblib.load('../models/tier_metadata.pkl')
        return modelo, metadata
    except:
        try:
            modelo = joblib.load('models/tier_classifier.pkl')
            metadata = joblib.load('models/tier_metadata.pkl')
            return modelo, metadata
        except:
            return None, None


clf_pipeline, tier_metadata = cargar_recursos()

TIER_INFO = {
    1: "Económico (Alta Depreciación / Trabajo)",
    2: "Gama Media-Baja (Uso Diario / Sedan)",
    3: "Gama Media-Alta (Buen Estado / Versátil)",
    4: "Premium (Lujo / Seminuevo / Colección)"
}

st.sidebar.title("⚡ Panel Control")
st.sidebar.markdown("---")

# Vehículo
modelo_input = st.sidebar.selectbox("Modelo", ["Ford F-150", "Toyota Camry", "Chevrolet Silverado 1500", "Honda Accord",
                                               "GMC Sierra 1500"])
manufacturer_input = modelo_input.split()[0].lower()
year_input = st.sidebar.slider("Año", 2000, 2025, 2018)
odometer_input = st.sidebar.number_input("Millas Actuales", 0, 400000, 65000)

col1, col2 = st.sidebar.columns(2)
cyl_input = col1.selectbox("Cilindros", [4, 6, 8])
drive_input = col2.selectbox("Tracción", ["fwd", "rwd", "4wd"])

# Mercado y Días Stock
st.sidebar.markdown("---")
st.sidebar.subheader("🌐 Situación de Mercado")
demanda = st.sidebar.slider("Demanda Web (%)", -50, 50, 10)
stock_rival = st.sidebar.slider("Stock Competencia (%)", -50, 50, -5)
dias_stock = st.sidebar.number_input("Días en Inventario (Aging)", 0, 365, 15)

analizar = st.sidebar.button("CALCULAR TASACIÓN COMPLETA")

st.title("Sistema Inteligente de Tasación ElasticDrive")

if analizar:
    if clf_pipeline and tier_metadata:
        # A. PROCESAMIENTO
        input_df = pd.DataFrame([{
            'manufacturer': manufacturer_input, 'model': modelo_input,
            'year': year_input, 'odometer': odometer_input, 'cylinders_num': cyl_input,
            'fuel': 'gas', 'drive': drive_input, 'transmission': 'automatic',
            'type': 'pickup' if 'F-150' in modelo_input or 'Sierra' in modelo_input else 'sedan',
            'paint_color': 'white', 'condition': 'good'
        }])

        tier_pred = int(clf_pipeline.predict(input_df)[0])

        if odometer_input > 300000:
            tier_pred = 1

        probs = clf_pipeline.predict_proba(input_df)[0]
        stats = tier_metadata.get(tier_pred, {})

        # B. LÓGICA ELASTICDRIVE (Precios)
        precio_base = stats.get('median_price', 0)

        # Factor mercado
        f_mercado = 1 + (0.3 * ((demanda / 100) - (stock_rival / 100)))
        # Factor depreciación por días en stock (2% cada 30 días)
        f_aging = (1 - 0.02) ** (dias_stock / 30)

        precio_final = precio_base * f_mercado * f_aging
        confianza = max(probs) * 100

        # Color dinámico
        variacion = ((precio_final / precio_base) - 1) * 100
        color_trend = "#2ECC71" if variacion >= 0 else "#E74C3C"

        # --- VISUALIZACIÓN DE RESULTADOS ---
        st.markdown(f'<p class="big-yellow">TIER {tier_pred}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sub-text">{TIER_INFO[tier_pred]}</p>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio Base", f"${precio_base:,}")
        m2.metric("Sugerido IA", f"${int(precio_final):,}", delta=f"{variacion:.1f}%")
        m3.metric("Aging Stock", f"{dias_stock} días", f"Impacto: {(f_aging - 1) * 100:.1f}%", delta_color="inverse")
        m4.metric("Confianza IA", f"{confianza:.1f}%")

        st.markdown("---")

        c1, c2 = st.columns([1, 1.5])

        with c1:
            st.markdown("#### 🧠 Justificación y Confianza")
            st.info(f"Análisis para {modelo_input}: Segmentado como **{TIER_INFO[tier_pred]}**.")

            # Gráfico Probabilidades
            fig_p, ax_p = plt.subplots(figsize=(6, 4))
            plt.style.use('dark_background')
            fig_p.patch.set_facecolor('#121212')
            ax_p.set_facecolor('#121212')
            colors = [color_trend if i + 1 == tier_pred else '#333' for i in range(4)]
            ax_p.barh([f"Tier {i + 1}" for i in range(4)], probs, color=colors)
            ax_p.set_title("Certidumbre del Algoritmo")
            st.pyplot(fig_p)

        with c2:
            st.markdown("#### 📈 Proyección de Activo (180 Días)")
            dias_p = np.arange(0, 185, 5)
            y_base = [precio_base * ((1 - 0.02) ** (d / 30)) for d in dias_p]
            y_sug = [precio_final * ((1 - 0.02) ** (d / 30)) for d in dias_p]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(dias_p, y_base, '--', color='#555', label='Depreciación Mercado')
            ax.plot(dias_p, y_sug, color=color_trend, linewidth=3, label='Sugerencia ElasticDrive')
            ax.fill_between(dias_p, y_base, y_sug, color=color_trend, alpha=0.1)

            ax.set_facecolor('#121212')
            fig.patch.set_facecolor('#121212')
            ax.legend(frameon=False)
            ax.set_xlabel("Días en Inventario")
            ax.set_ylabel("Valoración ($)")
            ax.grid(True, alpha=0.1)
            st.pyplot(fig)

    else:
        st.error("Error: No se detectan los modelos entrenados (.pkl).")

else:
    st.image("https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&q=80&w=1000",
             width="stretch")
    st.markdown(
        "<center><h3>⚡ Configure los parámetros y presione 'Calcular' para iniciar la tasación dinámica.</h3></center>",
        unsafe_allow_html=True)
