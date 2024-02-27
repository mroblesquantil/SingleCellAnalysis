from glob import glob
import os
from tqdm import tqdm

####################################################################################
########################## Creación de las matrices ################################
####################################################################################

for f in glob('../../Datos de prueba/generados/symsim/*') + [
   '../../data/10X_PBMC_select_2100.h5', 
   '../../data/HumanLiver_counts_top5000.h5']:
    path_input = f

    try:
        folder = f.split('/')[-1].split('.')[0].split('clusters_')[1]
    except:
        folder = f.split('/')[-1].split('.')[0]

    path_results = '../../matrices_grafos/' + folder + '/'

    for tipo in ["euclideana", "tfidf", "cosenos", "correlacion"]:
        os.system(f'python construccion_matrices.py "{path_input}" {path_results} {tipo}')

print('----> Finalizó la creación de las matrices')

####################################################################################
########################## Creación de los grafos ##################################
####################################################################################

clases = ["kMST", "Complete", "KNN"]

# Grafos de correlación        
print('----> Creación de grafos de correlación')

threshold = 0.1
nombre = f"correlaciones_th_{threshold}"
for c in clases:
    for f in glob('../../matrices_grafos/*'):
        path = f + '/correlaciones.pickle'
        output_path = '../../grafos/'+ f.split('/')[-1] + '/'

        os.system(f'python construccion_grafos.py {path} {output_path} True {threshold} {nombre} {c}')

# Grafos de TFIDF 
print('----> Creación de grafos de TFIDF')

threshold = 0.4
nombre = f"tfidf_cosine_th_{threshold}"

for c in clases:
    for f in glob('../../matrices_grafos/*'):
        if 'macosko' not in f:
            path = f + '/tfidf_cosine.pickle'
            output_path = '../../grafos/'+ f.split('/')[-1] + '/'

            os.system(f'python construccion_grafos.py {path} {output_path} True {threshold} {nombre} {c}')

# Grafos de cosenos 
print('----> Creación de grafos de Cosenos')

threshold = 0.4
nombre = f"cosenos_th_{threshold}"

for c in clases:
    for f in glob('../../matrices_grafos/*'):
        if 'macosko' not in f:
            path = f + '/cosenos.pickle'
            output_path = '../../grafos/'+ f.split('/')[-1] + '/'

            os.system(f'python construccion_grafos.py {path} {output_path} True {threshold} {nombre} {c}')

# Grafos de distancias euclideanas de datos normalizados
print('----> Creación de grafos de distancias euclideanas')

threshold = 0
nombre = f"euclideana_th_{threshold}"

for c in clases:
    for f in glob('../../matrices_grafos/*'):
        if 'macosko' not in f:
            path = f + '/euclideana.pickle'
            output_path = '../../grafos/'+ f.split('/')[-1] + '/'

            os.system(f'python construccion_grafos.py {path} {output_path} False {threshold} {nombre} {c}')
