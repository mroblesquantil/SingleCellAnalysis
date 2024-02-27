from sklearn.decomposition import PCA
import numpy as np
import argparse 
import os 
import h5py

def read_data(path):
  data_mat = h5py.File(path)
  assert 'Y' in data_mat.keys() and 'X' in data_mat.keys()

  x = np.array(data_mat['X'], dtype = np.float64)
  if 'Y' in data_mat.keys():
    y = np.array(data_mat['Y'], dtype = np.float64)
  else: y = None
  data_mat.close()

  return x, y

def reduccion_pca(X):
    pca = PCA(0.95)
    X_pca = pca.fit_transform(X)
    return X_pca

if __name__ == "__main__":
    # Set hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("path_input")
    parser.add_argument("path_results")
    args = parser.parse_args()

    if not os.path.exists(args.path_results):
        os.makedirs(args.path_results)
        print(f'-----> Se creó la carpeta {args.path_results}')

    X, y = read_data(args.path_input)
    X_pca = reduccion_pca(X)
    
