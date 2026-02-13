# ElasticDrive: Motor de Tasación Dinámica con IA

> **Proyecto de la Asignatura:** Datos y Negocio
>
> *Un sistema inteligente que transforma la tasación de vehículos: de tablas estáticas a precios dinámicos basados en oferta, demanda y envejecimiento del stock.*

---

## Descripción del Proyecto

**ElasticDrive** es una solución de ciencia de datos aplicada a negocio diseñada para resolver ineficiencias en el mercado de vehículos de segunda mano.

El proyecto implementa un pipeline completo que va desde la limpieza de datos hasta el despliegue de una aplicación web, utilizando algoritmos de **Clustering** para segmentar el mercado y **Modelos Predictivos** para establecer precios base, aplicando finalmente una capa de **Elasticidad** de negocio.

### Objetivos de Negocio
1.  **Segmentación Inteligente:** Clasificación automática del inventario en 4 "Tiers" (calidades) mediante K-Means, eliminando la subjetividad humana.
2.  **Tasación Justa:** Predicción del valor de mercado mediante Random Forest.
3.  **Pricing Dinámico:** Ajuste automático del precio según la presión de la demanda web y el stock de la competencia.
4.  **Gestión de Aging:** Penalización automática del valor para vehículos estancados en inventario.

---

## Estructura del Proyecto

```text
├── app/
│   └── app.py              # Aplicación Web interactiva (Streamlit)
├── data/                   # Datos del proyecto
│   ├── raw/vehicles.csv    # Dataset original (Kaggle)
│   ├── processed/df_master.csv
│   └── processed/df_master_clustered.csv
├── docs/                   # Documentación de Negocio
│   ├── AI_Canvas.pdf       # Estrategia del proyecto
│   ├── Calculo_Inversion_Costes_vs_Ingresos.xlsx # Modelo financiero
│   └── ElasticDrive.pdf    # Memoria del proyecto
├── models/                 # Artefactos serializados (.pkl)
│   ├── tier_classifier.pkl # Pipeline de clasificación (IA)
│   └── tier_metadata.pkl   # Metadatos de precios por Tier
├── notebooks/              # Desarrollo paso a paso
│   ├── 01_Limpieza_y_EDA.ipynb
│   ├── 02_Clustering_KMeans.ipynb
│   ├── 03_00_Clasificacion_RandomForest.ipynb
│   ├── 03_01_Validacion.ipynb
│   └── 04_ElasticDrive.ipynb
├── src/                    # Código fuente auxiliar
│   ├── clustering_viz.py
│   └── imputation.py
├── pyproject.toml          # Dependencias (UV)
├── uv.lock                 # Versiones exactas (Lockfile)
└── README.md               # Este archivo

```

---

## Flujo de Trabajo (Notebooks)

El desarrollo técnico se divide en 5 fases documentadas en la carpeta `notebooks/`:

1. **01_Limpieza_y_EDA:** Ingesta de datos crudos, tratamiento de nulos (imputación inteligente) y análisis exploratorio para entender la depreciación.
2. **02_Clustering_KMeans:** Aprendizaje no supervisado para descubrir los 4 segmentos naturales del mercado (Económico, Gama Media, Gama Alta, Premium).
3. **03_00_Clasificacion:** Entrenamiento de un modelo supervisado (Random Forest) capaz de asignar cualquier coche nuevo a uno de los segmentos descubiertos.
4. **03_01_Validacion:** Pruebas de generalización con marcas no vistas durante el entrenamiento (ej: Honda, GMC) para asegurar la robustez del modelo.
5. **04_ElasticDrive:** Simulación de escenarios de negocio y pruebas de estrés del algoritmo de elasticidad de precios.

---

## Tecnologías

Este proyecto utiliza **Python 3.14** y se gestiona con **uv** para una instalación rápida y fiable.

* **Gestión de Paquetes:** `uv`
* **Análisis y Datos:** `pandas`, `numpy`
* **Visualización:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`
* **Web App:** `streamlit`

---

## Instalación y Ejecución

Sigue estos pasos para ejecutar el proyecto en tu máquina local.

### 1. Clonar el repositorio

```bash
git clone https://github.com/Gpascual11/ElasticDrive.git
cd ElasticDrive

```

### 2. Instalar dependencias con UV

Si tienes `uv` instalado, el proyecto se sincronizará automáticamente con el archivo `uv.lock`.

```bash
uv sync

```

### 3. Ejecutar la Aplicación

Lanza el servidor de Streamlit utilizando el entorno virtual gestionado por uv:

```bash
uv run streamlit run app/app.py

```

La aplicación se abrirá automáticamente en tu navegador (normalmente en `http://localhost:8501`).

---

## Documentación de Negocio

En la carpeta `docs/` encontrarás los entregables estratégicos de la asignatura:

* **AI Canvas:** Definición de la propuesta de valor, datos necesarios y métricas de éxito.
* **Modelo Financiero (Excel):** Cálculo de ROI, costes de nube e ingresos estimados por el uso de la herramienta.

---

> *Este proyecto fue desarrollado como parte del Trabajo de la asignatura Datos y Negocio. Los datos utilizados provienen de un dataset público de Kaggle sobre ventas de vehículos usados en EE.UU.*
> https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data?resource=download
