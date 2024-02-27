import networkx as nx
import numpy as np
import argparse
from sklearn.decomposition import PCA
import pickle
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score, silhouette_score)
from scipy.optimize import linear_sum_assignment
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt 
import seaborn as sns
import h5py
import os

def louvain(grafo):
    particiones = nx.community.louvain_communities(grafo, seed=123)

    diccionario = {}

    # Crear el diccionario
    for i, conjunto in enumerate(particiones):
        for elemento in conjunto:
            diccionario[elemento] = i

    # Crear la lista deseada
    max_elemento = max(max(particiones, key=max), default=-1)
    clusters = np.array([diccionario.get(i, -1) for i in range(max_elemento + 1)])

    return clusters

def girvan_newman(grafo):
    comp = nx.community.girvan_newman(grafo)
    communities = tuple(sorted(c) for c in next(comp))
    node_community_map = {node: i for i, nodes in enumerate(communities) for node in nodes}
    clusters = np.array([node_community_map[node] for node in grafo.nodes()])
    
    return clusters

def betweeness_centrality_clustering(grafo,num_clusters):
    edge_betweennes = nx.edge_betweenness_centrality(grafo)
    edge_betweennes_sorted = list(dict(sorted(edge_betweennes.items(), 
                                              key=lambda item: item[1], reverse=True)))

    stop = False 
    while not stop:
        edge_to_remove = edge_betweennes_sorted.pop(0)
        grafo.remove_edge(edge_to_remove[0], edge_to_remove[1])

        num_cc = nx.number_connected_components(grafo)
        if num_cc >= num_clusters:
            stop = True 

    connected_components  = nx.connected_components(grafo)
    connected_components_dict = {node: i for i, nodes in enumerate(connected_components) for node in nodes}
    clusters = np.array([connected_components_dict[node] for node in grafo.nodes()])

    return clusters

def unsupervised_metrics(X, y_pred):
    # Realizamos PCA a 32 componentes
    X = PCA(n_components=32).fit_transform(X)

    # Evaluación final de resultados: métricas comparando con los clusters reales
    try:
        sil = np.round(silhouette_score(X, y_pred), 5)
        chs = np.round(calinski_harabasz_score(X, y_pred), 5)
        dbs = np.round(davies_bouldin_score(X, y_pred), 5)
    except:
        sil, chs, dbs = None, None, None

    return {'sil': sil, 'chs': chs, 'dbs': dbs}

def supervised_metrics(y_true, y_pred):
    acc = round(cluster_acc(y_true, y_pred),3)
    nmi = round(metrics.normalized_mutual_info_score(y_true, y_pred),3)
    ari = round(metrics.adjusted_rand_score(y_true, y_pred),3)

    return {'acc': acc, 'nmi': nmi, 'ari': ari}

def cluster_acc_plot(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    
    df_cm = pd.DataFrame(w, index = [i for i in range(D)], columns = [i for i in range(D)])
    plt.figure(figsize = (10,7))
    w_order = np.zeros((D, D), dtype=np.int64)
    for i in range(D):
        for j in range(D):
            w_order[i,j] = w[i, ind[1][j]]

    df_cm = pd.DataFrame(w_order, index = [i for i in range(D)], columns = [i for i in ind[1]])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.ylabel("Prediction")
    plt.xlabel("Ground Truth")
    plt.show()
    
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)

    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
       

def run_clustering_graph(path_graph, path_results, path_data, nombre, tipo):
    # Lectura de datos iniciales para métricas
    data_mat = h5py.File(path_data)
    X = np.array(data_mat['X'], dtype = np.float64)
    y = np.array(data_mat['Y'], dtype = np.float64)

    if len(np.shape(y)) == 2:
        y = np.squeeze(y)

    # Lectura de grafo
    with open(path_graph, "rb") as file:
        grafo = pickle.load(file)
    
    # Clusterización
    if tipo == 'Louvain':
        clusters = louvain(grafo)

    elif tipo == 'Girvan-Newman':
        clusters = girvan_newman(grafo)

    elif tipo == 'Betweenesss':
        num_clusters = len(set(y))
        clusters = betweeness_centrality_clustering(grafo, num_clusters)

    # Guardar resulados
    if not os.path.exists(path_results):
        os.makedirs(path_results)
        print(f'-----> Se creó la carpeta {path_results}')

    with open(path_results + f'clusers_{tipo}_{nombre}.pickle', "wb") as file:
        pickle.dump(clusters, file)

    # Calcular métricas
    sup_metrics = supervised_metrics(y_true=y, y_pred=clusters)
    unsup_metrics = unsupervised_metrics(y_pred=clusters, X = X)

    return {**sup_metrics, **unsup_metrics}

    

