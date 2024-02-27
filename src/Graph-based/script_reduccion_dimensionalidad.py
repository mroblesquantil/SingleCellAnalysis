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

    path_results = '../../data_dimensionalidad/' + folder + '/'

    os.system(f'python dimensionalidad_datos.py "{path_input}" {path_results}')

print('----> Finalizó la reducción de dimensionalidad')