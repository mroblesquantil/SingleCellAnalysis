from tqdm import tqdm 
from glob import glob
from clustering_grafos import *

####################################################################################
################################ Clusterización ####################################
####################################################################################

# print('--------------- LOUVAIN')
# results = []
# for f in tqdm(glob('../../grafos/*/*')):
#     if 'macosko' not in f:
#         path_graph = f 
#         path_results = '/'.join(f.split('/')[:4]).replace('grafos', 'results_graph_based')+'/'
        
#         dato = path_results.split('/')[-2]
#         if "symsim" in dato:
#             path_data = f'../../Datos de prueba/generados/symsim/clusters_{dato}.h5'
#         else:
#             path_data = f'../../data/{dato}.h5'

#         nombre = path_graph.split('/')[-1].split('.pickle')[0]
#         metrics = run_clustering_graph(path_graph, path_results, path_data, nombre, 'Louvain') 

#         results.append({**{'dataset': dato, 'name': nombre, 'algorithm': 'Louvain'}, **metrics})

# results = pd.DataFrame(results)

# results.to_csv('../../results_graph_based/resultados_louvain.csv', index = None)


# print('--------------- BETWEENNESS DEGREE')
# results = []
# for f in tqdm(glob('../../grafos/*/*')):
#     if 'macosko' not in f and 'Liver' not in f and "10" not in f:
#         path_graph = f 
#         path_results = '/'.join(f.split('/')[:4]).replace('grafos', 'results_graph_based')+'/'
        
#         dato = path_results.split('/')[-2]
#         if "symsim" in dato:
#             path_data = f'../../Datos de prueba/generados/symsim/clusters_{dato}.h5'
#         else:
#             path_data = f'../../data/{dato}.h5'

#         nombre = path_graph.split('/')[-1].split('.pickle')[0]
#         metrics = run_clustering_graph(path_graph, path_results, path_data, nombre, 'Betweenesss') 

#         results.append({**{'dataset': dato, 'name': nombre, 'algorithm': 'Betweenesss'}, **metrics})

# results = pd.DataFrame(results)

# results.to_csv('../../results_graph_based/resultados_betweeness.csv', index = None)


print('--------------- CLUSTERING GIRVAN-NEWMAN')
results = []
for f in tqdm(glob('../../grafos/*/*')):
    if 'macosko' not in f:
        path_graph = f 
        path_results = '/'.join(f.split('/')[:4]).replace('grafos', 'results_graph_based')+'/'
        
        dato = path_results.split('/')[-2]
        if "symsim" in dato:
            path_data = f'../../Datos de prueba/generados/symsim/clusters_{dato}.h5'
        else:
            path_data = f'../../data/{dato}.h5'

        nombre = path_graph.split('/')[-1].split('.pickle')[0]
        metrics = run_clustering_graph(path_graph, path_results, path_data, nombre, 'Girvan-Newman') 

        results.append({**{'dataset': dato, 'name': nombre, 'algorithm': 'Girvan-Newman'}, **metrics})

results = pd.DataFrame(results)

results.to_csv('../../results_graph_based/resultados_girvan_newman.csv', index = None)

