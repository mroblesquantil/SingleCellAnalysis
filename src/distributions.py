import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from scipy import sparse


def plot_counts_per_cell(adata: sc.AnnData, 
                         xlim: tuple = None, 
                         ylim: tuple = None
                         ) -> plt.Figure:
    """
    Creates a joint plot of total counts per cell vs number of genes with counts per cell.
    The values are presented after a logarithmic tranformation. 

    input: 
    - adata: AnnData with counts matrix
    - path: str with path to save the figure
    - xlim: tuple with x lim in logarithm
    - ylim: tuple with y lim in logarithm

    output:
    - fig: figure with joint plot of the AnnData
    """
    fig = plt.Figure()
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    if xlim == None:
        xlim = (adata.obs.log1p_total_counts.min(), adata.obs.log1p_total_counts.max())
    if ylim == None:
        ylim = (adata.obs.log1p_n_genes_by_counts.min(), adata.obs.log1p_n_genes_by_counts.max())

    p = sns.jointplot(
        data=adata.obs,
        x="log1p_total_counts",
        y="log1p_n_genes_by_counts",
        kind="hex",
        xlim = xlim, ylim = ylim
    )
    old_ticks = plt.xticks()
    new_labels = [round(np.exp(i),2) for i in old_ticks[0]]
    plt.xticks(old_ticks[0], new_labels, rotation = 90)

    old_ticks = plt.yticks()
    new_labels = [round(np.exp(i),2) for i in old_ticks[0]]
    plt.yticks(old_ticks[0], new_labels)

    p.fig.suptitle('Joint plot total counts vs genes by counts per cell')

    return fig


def plot_counts_per_gene(adata: sc.AnnData, 
                         xlim: tuple = None, 
                         ylim: tuple = None
                         ) -> plt.Figure:
    """
    Creates a joint plot of total counts per gene vs number of genes with counts per gene.
    The values are presented after a logarithmic tranformation. 

    input: 
    - adata: AnnData with counts matrix
    - xlim: tuple with x lim in logarithm
    - ylim: tuple with y lim in logarithm

    output:
    - fig: figure with joint plot of the AnnData
    """
    fig = plt.Figure()
    print(adata.X.shape, adata.obs.shape)

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    if xlim == None:
        xlim = (adata.var.log1p_total_counts.min(), adata.var.log1p_total_counts.max())
    if ylim == None:
        ylim = (adata.var.n_cells_by_counts.min(), adata.var.n_cells_by_counts.max())

    p = sns.jointplot(
        data=adata.var,
        x="log1p_total_counts",
        y="n_cells_by_counts",
        kind="hex",
        xlim = xlim, ylim = ylim
    )
    old_ticks = plt.xticks()
    new_labels = [round(np.exp(i),2) for i in old_ticks[0]]
    plt.xticks(old_ticks[0], new_labels, rotation = 90)

    p.fig.suptitle('Joint plot total counts vs cells by counts per gene')

    return fig

def plot_general_counts(adata: sc.AnnData) -> plt.Figure:
    """
    Creates a histogram of the counts of the matrix for non-zero counts.

    input:
    - adata: AnnData with counts matrix

    output:
    - fig: figure with the histogram and percentage of 0
    """
    fig = plt.Figure()
    sparse_mat = sparse.csr_matrix(adata.X)
    nonzero_values = sparse_mat.nonzero()
    nonzero_values = np.array([sparse_mat[i,j] for (i,j) in zip(nonzero_values[0],nonzero_values[1])])
    
    per_less_10 = round(len(nonzero_values[nonzero_values<10])/len(nonzero_values)*100,2)

    nonzero_values = np.where(nonzero_values >= 10, '>10', nonzero_values)
    nonzero_values.sort()

    sns.countplot(x = nonzero_values, alpha = 0.5).set(
        title = f'Distribución valores distintos de 0 y menores de 10\nPorcentaje < 10: {per_less_10}%')
    return fig

def plot_heat_map_genes(adata: sc.AnnData, all_genes = True) -> plt.Figure:
    """
    Creates a heatmap of the counts per gene excluding 0 counts and grouping counts higher than 10.

    input:
    - adata: AnnData with counts matrix
    - all_genes: Indica si se muestran todos los genes o únicamente los que tienen conteos mayores que 10.

    outpu:
    - fig: figure with the heatmap
    """
    mat_heatmap = []

    sparse_mat = sparse.csr_matrix(adata.X)
    fig = plt.Figure()

    # Itero sobre los genes
    for gene in range(sparse_mat.shape[1]):
        gene_info = np.array(sparse_mat[:,gene].todense().flatten())[0]

        # Quitamos 0's
        gene_info = gene_info[gene_info > 0]

        # Cambiamos los mayores a 10 por 10
        gene_info = np.where(gene_info > 10, 10, gene_info)
        gene_info.sort()

        # Si solo queremos incluir genes con más de 10 conteos
        if all_genes or (
            not all_genes and 10 in gene_info):
            unique, counts = np.unique(gene_info, return_counts=True)
            
            all_values = np.arange(1, 11)
            all_counts = np.zeros_like(all_values)
            indices = np.searchsorted(all_values, unique)

            all_counts[indices] = counts

            mat_heatmap.append(all_counts)

    sns.heatmap(mat_heatmap,cmap='RdBu_r',center=0)
    
    # TODO Cambiar labels
    plt.title(f'Heatmap de conteos por gen')

    return fig

def plot_distribution_clusters(adata: sc.AnnData):
    """
    Plots the distribution of cells in gold label clusters.

    input:
    - adata: AnnData with counts matrix and cluster information

    output:
    - fig: figure with the distribution of clusters
    """
    fig = plt.Figure()
    y = adata.obs.Group
    y = y.astype(int)

    sns.countplot(x = y, order=y.value_counts().index).set(
        title = f'Número de clusters: {len(set(y))}')
    
    return fig