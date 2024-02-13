import argparse
import os

import h5py
import numpy as np
import scanpy as sc

from distributions import *
from top_cells_distributions import *

parser = argparse.ArgumentParser()
parser.add_argument("input_file")

def read_file(input_file: str)-> sc.AnnData:
    """
    Reads the file in h5 format with keys X and y.

    input:
    - input_file: with path

    output:
    - adata: AnnData with the information of counts and clusters.
    """
    with h5py.File(input_file, "r") as file:
        assert 'X' in file.keys() and 'Y' in file.keys()

        X = np.array(file['X'])
        X = X.astype(np.float64)
        y = np.array(file['Y'])
        y = y.astype(np.float64)

    adata = sc.AnnData(X, dtype=X.dtype)
    adata.obs['Group'] = y

    return adata

def create_distribution_plots(adata: sc.AnnData, results_path: str)-> None:
    """
    Creates and saves all distribution plots for the dataset.

    intput:
    - adata: AnnData with counts and clusters information
    - results_path: Path to save the plots
    """ 
    try:
        plot_counts_per_cell(adata)
        plt.savefig(results_path + '/counts_per_cell.png', bbox_inches='tight')
        plt.close()

    except:
        print('---------> pass')
        pass

    try:
        plot_counts_per_gene(adata)
        plt.savefig(results_path + '/counts_per_gene.png', bbox_inches='tight')
        plt.close()
    except:
        print('---------> pass')
        pass

    try:
        plot_general_counts(adata)
        plt.savefig(results_path + '/counts_distribution.png', bbox_inches='tight')
        plt.close()
    except:
        print('---------> pass')
        pass

    try:
        plot_heat_map_genes(adata)
        plt.savefig(results_path + '/heat_map_gene_counts.png', bbox_inches='tight')
        plt.close()
    except:
        print('---------> pass')
        pass

    try:
        plot_distribution_clusters(adata)
        plt.savefig(results_path + '/clusters_distribution.png', bbox_inches='tight')
        plt.close()
    except:
        print('---------> pass')
        pass

    print('\n-- Se guardaron las imagenes de distribuciones')

def create_top10_distribution_plots(adata: sc.AnnData, results_path: str)-> None:
    """
    """
    adata_10 = get_top_10_cells(adata)
    sim, euc, pearson, spearman = creates_distance_matrices(adata_10)

    plot_heatmaps(sim, euc, pearson, spearman)
    plt.savefig(results_path + '/top10_heatmap_distances.png', bbox_inches='tight')
    plt.close()

    plot_histogram_distances_clusters(adata_10, sim, euc, pearson, spearman)
    plt.savefig(results_path + '/top10_distogram_distances_clusters.png', bbox_inches='tight')
    plt.close()

    print('\n-- Se guardaron las imagenes de distribuciones de las células mejor representadas')



if __name__ == '__main__':
    args = parser.parse_args()
    adata = read_file(args.input_file)
    
    name = args.input_file.split('/')[-1].split('.')[0]
    results_path = '../results_exploration/' + name +'/' 
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    create_distribution_plots(adata, results_path)
    try:
        create_top10_distribution_plots(adata, results_path)
    except:
        print('---------> pass')
        pass 



    



