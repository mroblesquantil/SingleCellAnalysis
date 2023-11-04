import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse as sp
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def log_1(x):
    return np.log10(x+1)

def create_subtitle(fig: plt.Figure, grid, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

def get_top_10_cells(adata: sc.AnnData) -> sc.AnnData:
    """
    Filters a adata in order to consider only the top 10% cells in the sample

    intput: 
    - adata: AnnData with counts matrix

    output:
    - adata: AnnData with only top 10% cells
    """
    adata.X = np.array(list(map(log_1, adata.X)))

    quantile = pd.qcut(adata.obs['total_counts'], 10, labels = False) 
    adata = adata[quantile == 9]

    return adata

def get_distances_matrix(X: sp.csr_matrix, tipo : str = 'similitud') -> np.array:
    """
    Creates a distance matrix from a count matrix

    input: 
    - X: sp.csr_matrix
    - tipo: one of (similitud, euclidea, correlacion_pearson, rango_spearman)

    output:
    - distancias: squared np.array with the distance of each pair of cells.
    """
    num_cells = X.shape[0]
    num_genes = X.shape[1]
    distancias = np.zeros((num_cells, num_cells))

    for c1 in tqdm(range(num_cells)):
        celula1 = X[c1,:]
        indices_nocero1 = celula1.nonzero()[1]
        for c2 in range(c1, num_cells):
            celula2 = X[c2,:]
            indices_nocero2 = celula2.nonzero()[1]

            if tipo == 'similitud':
                indices_comunes = set(indices_nocero1).intersection(set(indices_nocero2))
                distancias[c1][c2] = len(indices_comunes)
                distancias[c2][c1] = len(indices_comunes)

            elif tipo == 'euclidea':
                dist = euclidean(np.array(celula1.todense())[0,:], np.array(celula2.todense())[0,:])
                distancias[c1][c2] = dist
                distancias[c2][c1] = dist

            elif tipo == 'correlacion_pearson':
                indices_comunes = list(set(indices_nocero1).intersection(set(indices_nocero2)))
                x1 = np.array(celula1.todense())[0,indices_comunes]
                x2 = np.array(celula2.todense())[0,indices_comunes]
                corr = pearsonr(x1, x2)
                distancias[c1][c2] = corr.statistic
                distancias[c2][c1] = corr.statistic

            elif tipo == 'rango_spearman':
                indices_comunes = list(set(indices_nocero1).intersection(set(indices_nocero2)))
                x1 = np.array(celula1.todense())[0,indices_comunes]
                x2 = np.array(celula2.todense())[0,indices_comunes]
                corr = spearmanr(x1, x2)
                distancias[c1][c2] = corr.correlation 
                distancias[c2][c1] = corr.correlation 
            
    return distancias

def creates_distance_matrices(adata: sc.AnnData) -> tuple:
    """
    Creates four different distance matrices (similitud, euclidea, correlacion_pearson, rango_spearman)

    input:
    - adata: AnnData with counts matrix

    output:
    - sim: np.array with similarity between each cell
    - euc: np.array with euclidean distance between each cell
    - pearson: np.array with pearson correlation between each cell
    - spearman: np.array with spearman correlation between each cell
    """
    mat = sp.csr_matrix(adata.X)
    sim = get_distances_matrix(mat, tipo = 'similitud')
    euc = get_distances_matrix(mat, tipo = 'euclidea')
    pearson = get_distances_matrix(mat, tipo = 'correlacion_pearson')
    spearman = get_distances_matrix(mat, tipo = 'rango_spearman')

    return sim, euc, pearson, spearman

def plot_heatmaps(sim: np.array, euc: np.array, 
                  pearson: np.array, spearman: np.array) -> plt.Figure:
    """
    Plots 4 histograms, one for each 'distance' metric used.

    input:
    - sim: np.array with similarity between each cell
    - euc: np.array with euclidean distance between each cell
    - pearson: np.array with pearson correlation between each cell
    - spearman: np.array with spearman correlation between each cell

    output:
    - fig: plt.Figure with the four heatmaps
    """
    p = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

    fig, axes = plt.subplots(ncols = 2, nrows = 2, constrained_layout = True)

    sns.heatmap(sim, cmap=p, ax = axes[0][0]).set(title = 'Similitud')
    sns.heatmap(euc, cmap=p, ax = axes[0][1]).set(title = 'Euclideana')
    sns.heatmap(pearson, cmap=p, ax = axes[1][0]).set(title = 'Correlación de Pearson')
    sns.heatmap(spearman, cmap=p, ax = axes[1][1]).set(title = 'Correlación de Spearman')

    return fig

def distancias_mismo_diferente_cluster(distance_mat: np.array, adata: sc.AnnData) -> tuple:    
    """
    Separates the distances of cells from the same cluster and different clusters

    input:
    - distance_mat: np.array with distance between each pair of cells
    - adata: AnnData with counts matrix

    output:
    - distancia_mismo_cluster: np.array with the distances of cells in the same cluster
    - distancia_diferente_cluster: np.array with the distances of cells in different clusters
    """
    num_celulas = adata.X.shape[0]

    # Ahora miramos cómo cambia esta distribución en pares de células del mismo cluster 
    distancia_mismo_cluster = []
    distancia_diferente_cluster = []

    for c1 in tqdm(range(num_celulas)):
        g1 = adata.obs.iloc[c1].Group
        for c2 in range(c1+1, num_celulas):
            g2 = adata.obs.iloc[c2].Group

            if g1 == g2: distancia_mismo_cluster.append(distance_mat[c1][c2])
            else: distancia_diferente_cluster.append(distance_mat[c1][c2])

    return distancia_mismo_cluster, distancia_diferente_cluster


def plot_histogram_distances_clusters(adata: sc.AnnData, sim: np.array, 
                                      euc: np.array,  pearson: np.array, spearman: np.array):
    """
    
    """
    grid = plt.GridSpec(5, 2)
    fig, axes = plt.subplots(ncols = 1, nrows = 4, figsize = (10,20),
                             constrained_layout = True)

    for i, (d, name_d) in enumerate(zip([sim, euc, pearson, spearman], 
                              ["Similitud", "Euclideana", "Correlación de Pearson", "Correlación de Spearman"])):
        distancia_mismo_cluster, distancia_diferente_cluster = distancias_mismo_diferente_cluster(
            d, adata)

        sns.histplot(np.log(1+np.array(distancia_mismo_cluster)), 
                     bins = 50, ax = axes[i], color = 'red', alpha = 0.5, label = 'Mismo cluster').set(
            title = name_d + ' - células mismo y distinto cluster')
        sns.histplot(np.log(1+np.array(distancia_diferente_cluster)), 
                     bins = 50, ax = axes[i], color = 'blue', alpha = 0.5, label = 'Diferente cluster')
        axes[i].legend()

    return fig