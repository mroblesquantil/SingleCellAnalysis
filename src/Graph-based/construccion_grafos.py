from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from tqdm import tqdm 
import pandas as pd
import numpy as np
import argparse
import pickle
import os 

def create_kMST(distance_matrix, inverse = True, k = None, threshold = 1e-5):
    if k is None:
        N = np.log(len(distance_matrix))
        k = int(np.floor(N))
    
    print(f'k = {k}')
    grafo = nx.Graph()
    nodos = range(len(distance_matrix))

    # Crear nodo inicial
    grafo.add_nodes_from(nodos)

    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix[i])):
            peso = distance_matrix[i][j]
            if peso > threshold:
                # para MST necesito el inverso de las correlaciones
                if inverse:
                    grafo.add_edge(i, j, weight=1-peso)
                else:
                    grafo.add_edge(i, j, weight=peso)


    print(f'---> Number of edges: {grafo.number_of_edges()}')

    mst_antes = None
    # Creamos los MSTs
    for iter in tqdm(range(k)):
        mst_new = nx.minimum_spanning_tree(grafo)

        edges_to_remove = list(mst_new.edges)
        grafo.remove_edges_from(edges_to_remove)

        if mst_antes is None:
            mst_antes = mst_new.copy()
        else:
            mst_new.add_edges_from(list(mst_antes.edges()))
            mst_antes = mst_new.copy()

    return mst_antes 


def create_complete(distance_matrix, inverse = True, threshold = 1e-5):    
    grafo = nx.Graph()
    nodos = range(len(distance_matrix))

    # Crear nodo inicial
    grafo.add_nodes_from(nodos)

    for i in range(len(distance_matrix)):
        for j in range(i + 1, len(distance_matrix[i])):
            peso = distance_matrix[i][j]
            if peso > threshold:
                # inverso de las correlaciones
                if inverse:
                    grafo.add_edge(i, j, weight=1-peso)
                else:
                    grafo.add_edge(i, j, weight=peso)
    
    return grafo

def create_knn(distance_matrix, inverse = True,k = None, threshold = 1e-5):    
    if k is None:
        N = np.log(len(distance_matrix))
        k = int(np.floor(N))
    
    distance_matrix = np.where(distance_matrix > threshold, distance_matrix, 0)

    if inverse:  dm = 1 - distance_matrix
    else: dm = distance_matrix

    matriz_knn = kneighbors_graph(dm, k)
    
    matriz_knn = np.multiply(matriz_knn.toarray(), dm)

    grafo = nx.from_numpy_array(matriz_knn)

    return grafo

if __name__ == "__main__":
    # Set hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("path_similarity_matrix")
    parser.add_argument("path_results")
    parser.add_argument("inverse") # Para sacar MST necesito que los mejores valores sean los más pequeños
                                   # True por ejemplo para correlaciones, distancias cosenos
                                   # False para distancias
    parser.add_argument("threshold")
    parser.add_argument("name")
    parser.add_argument("tipo")

    args = parser.parse_args()

    with open(args.path_similarity_matrix, "rb") as file:
        matrix = pickle.load(file)
    
    if not os.path.exists(args.path_results):
            os.makedirs(args.path_results)
            print(f'-----> Se creó la carpeta {args.path_results}')

    # Guardar grafo
    if args.tipo == 'kMST':
        # Crear MST
        kmst = create_kMST(distance_matrix = matrix, inverse = bool(args.inverse), threshold = float(args.threshold))
        print('-----> Terminó la creación del grafo kMST para k = logN')

        with open(args.path_results + f'kmst_graph_{args.name}.pickle', 'wb') as file:
            pickle.dump(kmst, file)
        print(f'-----> Se guardó correctamente el grafo kMST en {args.path_results}')
    
    if args.tipo == 'Complete':
        # Crear grafo 
        grafo = create_complete(distance_matrix = matrix, inverse = bool(args.inverse), threshold = float(args.threshold))
        print('-----> Terminó la creación del grafo de similitud')

        with open(args.path_results + f'complete_graph_{args.name}.pickle', 'wb') as file:
            pickle.dump(grafo, file)
        print(f'-----> Se guardó correctamente el grafo en {args.path_results}')
        
    if args.tipo == 'KNN':
        # Crear grafo 
        grafo = create_knn(distance_matrix = matrix, inverse = bool(args.inverse), threshold = float(args.threshold))
        print('-----> Terminó la creación del grafo de similitud')

        with open(args.path_results + f'knn_graph_{args.name}.pickle', 'wb') as file:
            pickle.dump(grafo, file)
        print(f'-----> Se guardó correctamente el grafo en {args.path_results}')
        
    
