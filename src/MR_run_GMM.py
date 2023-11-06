import argparse
import os
import pickle
from time import time

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn import metrics

from MR_GMM import scDCC
from preprocess import normalize, read_dataset
from utils import cluster_acc


def set_hyperparameters():
   # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file")
    parser.add_argument("path_results")

    parser.add_argument('--label_cells', default=0.1, type=float)
    parser.add_argument('--label_cells_files', default='label_selected_cells_1.txt')
    parser.add_argument('--n_pairwise', default=0, type=int)
    parser.add_argument('--n_pairwise_error', default=0, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--maxiter', default=100, type=int) 
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--ml_weight', default=1., type=float,
                        help='coefficient of must-link loss')
    parser.add_argument('--cl_weight', default=1., type=float,
                        help='coefficient of cannot-link loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scDCC_p0_1/')
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')

    args = parser.parse_args()

    return args

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

def create_train_model(args):
  # Create saving directory
  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

  sd = 2.5
  
  # Model
  model = scDCC(input_dim=adata.n_vars, z_dim=32, n_clusters=n_clusters, 
              encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=args.gamma,
              cov_identidad = args.cov_identidad, path = args.path_results).cuda()
  
  print(str(model))

  # Training
  t0 = time()
  if args.ae_weights is None:
      model.pretrain_autoencoder(x=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.size_factors, 
                              batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
  else:
      if os.path.isfile(args.ae_weights):
          print("==> loading checkpoint '{}'".format(args.ae_weights))
          checkpoint = torch.load(args.ae_weights)
          model.load_state_dict(checkpoint['ae_state_dict'])
      else:
          print("==> no checkpoint found at '{}'".format(args.ae_weights))
          raise ValueError

  print('Pretraining time: %d seconds.' % int(time() - t0))

  return model

def second_training(args, model):
    t0 = time()
    
    # Second training: clustering loss + ZINB loss
    y_pred,  mu, pi, cov, z, epochs, clustering_metrics, clustering_metrics_id, losses = model.fit(X=adata.X, X_raw=adata.raw.X, sf=adata.obs.size_factors,  
                                    batch_size=args.batch_size,  num_epochs=args.maxiter,
                                    update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir, lr = 0.001, y = y)

    # Se guardan los resultados
    pd.DataFrame(z.cpu().detach().numpy()).to_csv(args.path_results + 'Z.csv')
    pd.DataFrame(mu.cpu().detach().numpy()).to_csv(args.path_results + 'Mu.csv')
    pd.DataFrame(pi.cpu().detach().numpy()).to_csv(args.path_results + 'Pi.csv')
    pd.DataFrame(cov.cpu().detach().numpy()).to_csv(args.path_results + 'DiagCov.csv')

    with open(args.path_results + '/prediccion.pickle', 'wb') as handle:
        pickle.dump(y_pred, handle)

    print('Time: %d seconds.' % int(time() - t0))

    return y_pred

def supervised_metrics(y, y_pred, label_cell_indx):
    # Evaluación final de resultados: métricas comparando con los clusters reales
    if not y is None:
      eval_cell_y_pred = np.delete(y_pred, label_cell_indx)
      eval_cell_y = np.delete(y, label_cell_indx)
      acc = np.round(cluster_acc(eval_cell_y, eval_cell_y_pred), 5)
      nmi = np.round(metrics.normalized_mutual_info_score(eval_cell_y, eval_cell_y_pred), 5)
      ari = np.round(metrics.adjusted_rand_score(eval_cell_y, eval_cell_y_pred), 5)
      print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
  
      if not os.path.exists(args.label_cells_files):
          np.savetxt(args.label_cells_files, label_cell_indx, fmt="%i")

if __name__ == "__main__":
    # Set hyperparameters
    args = set_hyperparameters()

    # Reading the data
    x, y = read_data(args.data_file)

    # processing of scRNA-seq read counts matrix
    adata = format_normalize(x, y)
    input_size = adata.n_vars

    # Set k 
    if not y is None: n_clusters = len(set(y))
    else: n_clusters = input("Ingrese el número de clusters: ")

    n_clusters = int(n_clusters)
    
    if not os.path.exists(args.label_cells_files):
        indx = np.arange(len(y))
        np.random.shuffle(indx)
        label_cell_indx = indx[0:int(np.ceil(args.label_cells*len(y)))]
    else:
        label_cell_indx = np.loadtxt(args.label_cells_files, dtype=np.int)

    # Model training
    model = create_train_model(args)   
    y_pred = second_training(args, model)

    # Supervised metrics
    if not y is None: 
       supervised_metrics(y, y_pred, label_cell_indx)
