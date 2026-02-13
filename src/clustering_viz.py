import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np

def plot_elbow_method(inertia_values, k_range):
    """
    Pinta la gráfica del codo para ayudar a decidir el número de clusters.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia_values, 'bo-', linewidth=2, markersize=8)
    plt.title('Método del Codo para determinar K óptimo', fontsize=16)
    plt.xlabel('Número de Clusters (k)', fontsize=12)
    plt.ylabel('Inercia (Suma de distancias al cuadrado)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_3d_clusters(df, col_x, col_y, col_z, col_cluster):
    """
    Genera un gráfico 3D interactivo (si se usa %matplotlib qt) o estático
    para visualizar la separación de clusters.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Definimos colores bonitos
    palette = sns.color_palette("viridis", n_colors=len(df[col_cluster].unique()))

    # Pintamos cada cluster
    for cluster in sorted(df[col_cluster].unique()):
        subset = df[df[col_cluster] == cluster]
        ax.scatter(
            subset[col_x],
            subset[col_y],
            subset[col_z],
            label=f'Cluster {cluster}',
            s=50,
            alpha=0.6,
            edgecolors='w'
        )

    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_zlabel(col_z)
    ax.set_title(f'Segmentación 3D: {col_x} vs {col_y} vs {col_z}', fontsize=15)
    ax.legend(title="Segmentos")
    plt.show()


def plot_cluster_profiles(df, cluster_col, features):
    """
    Genera un gráfico de barras comparativo para entender el perfil de cada cluster.
    Normaliza los datos para que se puedan comparar variables distintas (ej: Precio vs Año).
    """
    # Calculamos la media de cada feature por cluster
    cluster_means = df.groupby(cluster_col)[features].mean()

    # Normalizamos (Min-Max) solo para visualización
    normalized_means = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())

    # Transponemos para plotear
    normalized_means = normalized_means.T

    plt.figure(figsize=(14, 8))
    normalized_means.plot(kind='bar', width=0.8, colormap='viridis', figsize=(12, 6))

    plt.title('Perfil Relativo de los Clusters (Valores Normalizados)', fontsize=16)
    plt.ylabel('Intensidad (0=Mínimo del grupo, 1=Máximo del grupo)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_2d_segmentation(df, x_col, y_col, cluster_col, title):
    """
    Scatterplot 2D optimizado para visualizar clusters.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=cluster_col,
        palette='viridis',
        alpha=0.6,
        s=50
    )
    plt.title(title, fontsize=15)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()