import pandas as pd
import numpy as np

# ==============================================================================
# 1. DICCIONARIOS DE REGLAS (Lógica de Negocio)
# ==============================================================================

REGLAS_DRIVE = {
    '4wd': r'4wd|4x4|awd|all[- ]?wheel|4[- ]?wheel',  # Prioridad 1: Tracción total
    'fwd': r'fwd|front[- ]?wheel',  # Prioridad 2: Delantera
    'rwd': r'rwd|rear[- ]?wheel'  # Prioridad 3: Trasera
}

REGLAS_CONDITION = {
    'salvage': r'salvage|wrecked|damaged|rebuilt|parts only|mechanic special|needs? work|not running',
    'like new': r'like new|mint cond|pristine|showroom',
    'new': r'brand new car|brand new condition|never driven',
    'excellent': r'excellent|perfect|great cond|stunning|flawless',
    'good': r'good cond|runs well|runs good|drives well|well maintained|clean inside|clean in and out',
    'fair': r'fair cond|rough|dents|scratches|rust|needs tlc|high miles'
}

REGLAS_TYPE = {
    'bus': r'\bbus\b|school bus|transit bus',
    'offroad': r'off[- ]?road|rock crawler',
    'mini-van': r'mini[- ]?van|mpv|town and country|caravan|odyssey|sienna',
    'pickup': r'pickup|pick[- ]?up|silverado|f-?150|ram 1500|tacoma|tundra',
    'convertible': r'convertible|soft top|hard top|cabriolet|roadster|spyder|miata',
    'coupe': r'\bcoupe\b|2[- ]?door|two[- ]?door',
    'hatchback': r'hatchback|liftback|5[- ]?door|five[- ]?door',
    'wagon': r'wagon|station wagon|estate car|avant|touring',
    'SUV': r'suv|crossover|sport utility|jeep|wrangler|cherokee|explorer|cr-?v|rav4',
    'van': r'\bvan\b|cargo van|work van|sprinter|express van|econoline',
    'truck': r'truck|\bdually\b|box truck|flatbed',
    'sedan': r'sedan|camry|accord|civic|corolla|altima|fusion|sonata|malibu'
}

REGLAS_CYLINDERS = {
    '8 cylinders': r'v[- ]?8|8[- ]?cyl|8[- ]?cylinder|hemi|duramax|powerstroke|triton|5\.0[l ]|5\.3[l ]|5\.7[l ]|6\.2[l ]',
    '6 cylinders': r'v[- ]?6|i[- ]?6|6[- ]?cyl|6[- ]?cylinder|3\.5[l ]|3\.6[l ]|3\.8[l ]|cummins',
    '4 cylinders': r'4[- ]?cyl|4[- ]?cylinder|i[- ]?4|4[- ]?banger|2\.4[l ]|2\.5[l ]|1\.8[l ]|2\.0[l ]|1\.6[l ]|civic|corolla|prius|camry',
    '10 cylinders': r'v[- ]?10|10[- ]?cyl|10[- ]?cylinder|triton v10',
    '12 cylinders': r'v[- ]?12|12[- ]?cyl|12[- ]?cylinder',
    '5 cylinders': r'5[- ]?cyl|5[- ]?cylinder|i[- ]?5',
    '3 cylinders': r'3[- ]?cyl|3[- ]?cylinder|i[- ]?3',
    'other': r'tesla|nissan leaf|electric motor|zero emission|battery only'
}


# ==============================================================================
# 2. FUNCIONES DE IMPUTACIÓN
# ==============================================================================

def imputar_por_regex(df, columna_target, diccionario_reglas, columna_texto='description'):
    """
    Imputa valores nulos basándose en patrones regex encontrados en una columna de texto.
    """
    df = df.copy()
    nulos_inicio = df[columna_target].isna().sum()
    print(f"--- Imputando '{columna_target}' por Regex ---")

    mask_na = df[columna_target].isna()

    # Iteramos sobre el diccionario respetando el orden de inserción
    for valor_imputar, patron in diccionario_reglas.items():
        # Solo buscamos en las filas que siguen siendo Nulas
        mask_match = df.loc[mask_na, columna_texto].astype(str).str.contains(patron, case=False, regex=True, na=False)

        # Asignamos el valor
        df.loc[mask_na & mask_match, columna_target] = valor_imputar

        # Actualizamos la máscara de nulos para la siguiente iteración (Prioridad)
        mask_na = df[columna_target].isna()

    nulos_fin = df[columna_target].isna().sum()
    print(f"Recuperados: {nulos_inicio - nulos_fin} registros.\n")
    return df


def imputar_por_grupos(df):
    """
    Rellena valores técnicos faltantes usando la moda/mediana del Modelo o Fabricante.
    """
    df = df.copy()
    print("--- Imputando por Estadística de Grupo (Modelo/Fabricante) ---")

    # 1. Imputar Manufacturer perdido basado en Model
    print("Mapeando fabricantes...")
    mapping = df.dropna(subset=['manufacturer', 'model']).set_index('model')['manufacturer'].to_dict()
    df['manufacturer'] = df['manufacturer'].fillna(df['model'].map(mapping))

    # 2. Imputar Características Técnicas (Moda por Modelo)
    tech_cols = ['fuel', 'transmission', 'type', 'drive', 'cylinders']

    for col in tech_cols:
        if col in df.columns:
            nulos_antes = df[col].isna().sum()
            # Rellenar con la moda del modelo
            df[col] = df[col].fillna(
                df.groupby('model')[col].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan))
            # Rellenar con la moda del fabricante (fallback)
            df[col] = df[col].fillna(
                df.groupby('manufacturer')[col].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan))
            print(f"Columna '{col}': {nulos_antes - df[col].isna().sum()} recuperados.")

    # 3. Imputar Año y Millas (Mediana)
    print("Imputando Años y Millas...")
    df['year'] = df['year'].fillna(df.groupby('model')['year'].transform('median'))
    df['odometer'] = df['odometer'].fillna(df.groupby('year')['odometer'].transform('median'))

    # 4. Condition (Fallback final)
    df['condition'] = df['condition'].fillna('good')

    return df