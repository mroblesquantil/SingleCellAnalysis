from tqdm import tqdm 
import networkx as nx
import scanpy as sc
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle 
import h5py
import seaborn as sns
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial import distance
import os

# Funciones auxiliares
def read_data(path):
  data_mat = h5py.File(path)
  assert 'Y' in data_mat.keys() and 'X' in data_mat.keys()

  x = np.array(data_mat['X'], dtype = np.float64)
  if 'Y' in data_mat.keys():
    y = np.array(data_mat['Y'], dtype = np.float64)
  else: y = None
  data_mat.close()

  return x, y

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def paint_matrix(correlaciones, output_path, name):
    plt.Figure()
    sns.heatmap(correlaciones)
    plt.savefig(output_path + name + '.png')

def get_tfidf(X):
    tfidf = TfidfTransformer(sublinear_tf=True)
    x_tfidf = tfidf.fit_transform(X).toarray()

    return x_tfidf

def get_correlations(X):
    celulas = X.shape[0]
    # Function to calculate Pearson correlation for a pair of ro
    def calculate_correlation(i, j):
        return i, j, pearsonr(X[i, :], X[j, :])[0]
    
    correlaciones = np.zeros((celulas, celulas))
    for i in tqdm(range(celulas)):
        for j in range(i, celulas):
            corr = pearsonr(X[i, :], X[j, :])[0]
            correlaciones[i, j] = corr 
            correlaciones[j, i] = corr 

    return correlaciones

def get_euclidean_distance(X):
    celulas = X.shape[0]
    
    distancias = np.zeros((celulas, celulas))
    for i in tqdm(range(celulas)):
        for j in range(i, celulas):
            dst = distance.euclidean(X[i, :], X[j, :])

            distancias[i, j] = dst 
            distancias[j, i] = dst 

    return distancias


def get_cosine_similarity(X):
    celulas = X.shape[0]
    # Function to calculate Pearson correlation for a pair of ro
    def calculate_cosine(i, j):
        return i, j, np.dot(X[i, :], X[j, :])/(norm(X[i, :])*norm(X[j, :]))
    
    correlaciones = np.zeros((celulas, celulas))
    for i in tqdm(range(celulas)):
        for j in range(i, celulas):
            corr = np.dot(X[i, :], X[j, :])/(norm(X[i, :])*norm(X[j, :]))
            correlaciones[i, j] = corr 
            correlaciones[j, i] = corr 

    # Number of threads (adjust as needed)
    # num_threads = 4

    # # Create a ThreadPoolExecutor
    # with ThreadPoolExecutor(max_workers=num_threads) as executor:
    #     futures = []

    #     # Iterate over pairs of indices (i, j) in parallel
    #     for i in tqdm(range(celulas)):
    #         for j in range(i, celulas):
    #             futures.append(executor.submit(calculate_cosine, i, j))

    #     # Wait for all tasks to complete
    #     results = [future.result() for future in tqdm(futures)]

    # # Create the correlation matrix using the results
    # correlaciones = np.zeros((celulas, celulas))

    # for i, j, corr in results:
    #     correlaciones[i, j] = corr
    #     correlaciones[j, i] = corr

    return correlaciones

def save_matrix(correlaciones, output_path, name):
    with open(output_path + name + '.pickle', 'wb') as f:
        pickle.dump(correlaciones, f)

if __name__ == "__main__":
    # Set hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input")
    parser.add_argument("path_results")
    parser.add_argument("tipo")
    args = parser.parse_args()

    if not os.path.exists(args.path_results):
        os.makedirs(args.path_results)
        print(f'-----> Se creó la carpeta {args.path_results}')


    # Reading the data
    X, y = read_data(args.path_input)

    # Converting to AnnData
    anndata_p = sc.AnnData(X)

    # Normalize
    if args.tipo in ['correlacion', 'euclideana']:
        normalize(anndata_p) 
        print(f"Se normalizaron correctamente los datos. Dimensiones: {anndata_p.X.shape}")

    X = anndata_p.X

    # Correlaciones
    if args.tipo == 'correlacion':
        correlaciones = get_correlations(X)

        # Guardar datos
        paint_matrix(correlaciones, args.path_results, name = "correlaciones_heatmap")
        save_matrix(correlaciones,  args.path_results, name = "correlaciones")
        print('Se guardaron correctamente las correlaciones y el heatmap en la carpeta ' + args.path_results)

    # Cosenos
    if args.tipo == 'cosenos':
        cosenos = get_cosine_similarity(X)

        paint_matrix(cosenos, args.path_results, name = "cosenos_heatmap")
        save_matrix(cosenos,  args.path_results, name= "cosenos")
        print('Se guardaron correctamente los cosenos y el heatmap en la carpeta ' + args.path_results)

    # TFIDF
    if args.tipo == 'tfidf':
        tfidf = get_tfidf(X)
        cosenos_tfidf = get_cosine_similarity(tfidf)
        
        paint_matrix(cosenos_tfidf, args.path_results, name = "tfidf_cosine_heatmap")
        
        save_matrix(tfidf,  args.path_results, name= "tfidf_heatmap")
        save_matrix(cosenos_tfidf,  args.path_results, name= "tfidf_cosine")
        
        print('Se guardaron correctamente los tfidf cosine y el heatmap en la carpeta ' + args.path_results)


    # Euclideana
    if args.tipo == 'euclideana':
        distancias = get_euclidean_distance(X)

        # Guardar datos
        paint_matrix(distancias, args.path_results, name = "euclideana_heatmap")
        save_matrix(distancias,  args.path_results, name = "euclideana")
        print('Se guardaron correctamente las distancias y el heatmap en la carpeta ' + args.path_results)

