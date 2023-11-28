import scanpy as sc
import pandas as pd
import numpy as np
import argparse
import swifter
import h5py
import sys

from tqdm import tqdm
tqdm.pandas()

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score, normalized_mutual_info_score, adjusted_rand_score)
from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
import networkx as nx

sys.path.append('../')
from preprocess import *
from utils import *


def set_hyperparameters():
   # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("path_results")

def read_data(path):
  data_mat = h5py.File(path)
  assert 'Y' in data_mat.keys() and 'X' in data_mat.keys()

  x = np.array(data_mat['X'], dtype = np.float64)
  if 'Y' in data_mat.keys():
    y = np.array(data_mat['Y'], dtype = np.float64)
  else: y = None
  data_mat.close()

  return x, y


def format_normalize(x, y):
    adata = sc.AnnData(x)
    if not y is None: adata.obs['Group'] = y

    adata = read_dataset(adata,
                     transpose=False,
                     test_split=False,
                     copy=True)

    adata = normalize(adata,
                      size_factors=True,
                      normalize_input=True,
                      logtrans_input=True)
    
    return adata

def cluster_metrics(X, y, y_pred):
    # Supervised
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(adjusted_rand_score(y, y_pred), 5)
    
    # Unsupervised
    chs = np.round(calinski_harabasz_score(X, y_pred), 5)
    dbs = np.round(davies_bouldin_score(X, y_pred), 5)
    ss = np.round(silhouette_score(X, y_pred), 5)

    metrics = {'acc': acc, 'nmi': nmi, 'ari': ari,
               'chs': chs, 'dbs': dbs, 'ss': ss, 
               'num_clusters': len(set(y_pred))}
    
    return metrics

def clustering(X, N, resolution, correlation_function, pca = True, n_components = 32): 
    # Reducción de dimensionalidad
    if pca:
        pca = PCA(n_components = n_components)
        X = pca.fit_transform(X)
    
    # Creación del grafo KNN
    knn_graph = kneighbors_graph(X, n_neighbors = N)

    for i,j in zip(knn_graph.nonzero()[0], knn_graph.nonzero()[1]):
        # Seleccionamos las células 
        cell_i, cell_j = X[i,:], X[j,:]

        # Calculamos la correlación
        correlacion = correlation_function(cell_i, cell_j).statistic
        knn_graph[i,j] = correlacion
    
    # Creación del grafo
    knn_graph = nx.from_scipy_sparse_array(knn_graph)
    communities = nx.community.louvain_communities(knn_graph, seed=123, resolution = resolution)

    # Predicción
    y_pred = np.zeros(len(knn_graph.nodes))

    # Creación de la predicción final
    for c in range(len(communities)):
        com_c = communities[c]
        for i in com_c:
            y_pred[i] = c
    
    return X, y_pred

def hyperparameter_exploration(X, y, params = {
    'N': [5+i*2 for i in range(10)],
    'resolution': [0.1*i for i in range(1,11)], 
    'correlation_function': [pearsonr]
    }, pca = True):
   
    def completo(X, N, r, cf):
        N = int(N)
        X, y_pred = clustering(X, N = N, resolution = r, correlation_function = cf, pca = pca)
        results = cluster_metrics(X, y, y_pred)
        return results

    results = pd.DataFrame()
    
    comb = [(n, r) for n in params['N'] for r in params['resolution']]
    results['N'] = [c[0] for c in comb]
    results['r'] = [c[1] for c in comb]

    solution = pd.DataFrame(list(results.progress_apply(lambda row: completo(X, row.N, row.r, pearsonr), axis = 1).values))
    results = pd.concat([results,solution], axis = 1)

    return results

def main_no_pca(path, y, path_results = None):
    # Read the data
    X = pd.read_csv(path).values
    
    results = hyperparameter_exploration(X, y, pca = False)

    return results

def main_pca(path, path_results = None):
    # Reading the data
    x, y = read_data(path)

    # processing of scRNA-seq read counts matrix
    adata = format_normalize(x, y)

    results = hyperparameter_exploration(adata.X, y, pca = True)

    return results

if __name__ == "__main__":
    # Set hyperparameters
    args = set_hyperparameters()

    results = main_pca(args.data_file, args.path_results)
    





