import graphweave, reconstruct
import numpy as np
import networkx as nx
import torch
import pandas as pd
from pandas import Series, DataFrame
import scipy.stats
import stats
import re
import sys
import os
import random
from functools import partial
from itertools import product

def unit_test(graph_type='SBM', num_graphs=200, norm_start_vec=True, num_generated_graphs=2, num_nodes=100, do_integer_prog=True, test_frac=0.5, max_time=120, use_train_degrees=False, lr=1e-2, epochs=100):
  # set test_frac=1 for one-training-point

  deg_powers = [1.0, -1.0, 2.0, -2.0]
  alpha = 0.9
  random.seed(0)
  np.random.seed(0)
  torch.manual_seed(0)

  df, all_M, all_L, all_A = graphweave.create_graphs_df(num_nodes=num_nodes, num_graphs=num_graphs, num_steps=10, seed=0, use_norm_lap=True, norm_lap_alpha=alpha, deg_powers=deg_powers, graph_type=graph_type, norm_start_vec=norm_start_vec)
  Q, Qc, min_val, m, s = graphweave.get_Q_from_df(df, cat_mult=3)
  Q_train, Q_test, Qc_train, Qc_test, all_A_train, all_A_test = graphweave.train_test_split(Q, Qc, all_A, test_frac=test_frac)
  model, num_categories = graphweave.do_train_vals(Q_train, Qc_train, learning_rate=lr, num_epochs=epochs)

  all_generated_graphs = []
  
  for i in range(num_generated_graphs):
    print(f'Generating graph {i}:', end=' ')
    degrees = all_A_test[i].todense().sum(axis=1) if not use_train_degrees else all_A_train[i].todense().sum(axis=1) # degrees can be perturbed
    V_t, V_tplus1 = graphweave.generate_graph_from_degrees(model=model, degrees=degrees, deg_powers=deg_powers, alpha=alpha, m=m, s=s, cat_mult=3, min_val=min_val, num_categories=num_categories, num_steps=Q_train[0][0].shape[0], norm_start_vec=norm_start_vec)

    if do_integer_prog:
      try:
        A_predall2 = reconstruct.reconstruct_G_using_D_integer(V_t, V_tplus1, alpha=alpha, degrees=degrees, max_time=max_time)
      except:
        A_predall2 = None
    else:
      try:
        A_predall = reconstruct.reconstruct_G_using_D(V_t, V_tplus1, alpha=alpha, degrees=degrees, max_time=240)
        A_predall2 = reconstruct.pick_threshold5(A_predall, degrees) if A_predall is not None else None
      except:
        A_predall2 = None

    if A_predall2 is not None:
      all_generated_graphs.append(A_predall2)


  return all_generated_graphs, all_A_train, all_A_test

def funky_save(filename, arr):
  Y = np.empty(len(arr), object)
  if type(arr)==list:
    Y[:] = arr
  else:
    Y[:] = arr.tolist()
  np.savez_compressed(filename, *Y)
  print(f'Saved to {filename}')

def run_and_save_all(methods=['SBM', 'WS', 'BA', 'RGlikeSBM', 'Planar'], num_generated_graphs=40, use_existing_file=True, save_res=True, do_stats=True, use_train_degrees=True, **kwds):
  np.random.seed(0)
  if use_train_degrees is None:
    use_train_degrees = (int((1-kwds['test_frac'])*200) >= num_generated_graphs) if 'test_frac' in kwds else True
  for method in methods:
    fname = f'{method}_{"_".join([f"{k.replace("_","")}{v}" for k,v in kwds.items()])}' if len(kwds)>0 else f'{method}'
    print(fname)
    if use_existing_file and os.path.isfile(fname+'.npz'):
      X = np.load(fname+'.npz', allow_pickle=True)
      try:
        all_generated_graphs, all_A_train, all_A_test = X['all_generated_graphs'], X['all_A_train'], X['all_A_test']
      except:
        try:
          all_generated_graphs = list(np.load(f'Generated_{fname}.npz', allow_pickle=True).values())
          all_A_train = [Z.item() for Z in np.load(f'Train_{method}.npz', allow_pickle=True).values()]
          all_A_test = [Z.item() for Z in np.load(f'Test_{method}.npz', allow_pickle=True).values()]
        except:
          print(f'Need Generated_{fname}.npz or Train_{method}.npz or Test_{method}.npz')
          sys.exit(1)
    else:
      all_generated_graphs, all_A_train, all_A_test = unit_test(graph_type=method, num_generated_graphs=num_generated_graphs, use_train_degrees=use_train_degrees, **kwds)

    if save_res and not use_existing_file:
      funky_save(f'Generated_{fname}', all_generated_graphs)
      funky_save(f'Train_{method}', all_A_train)
      funky_save(f'Test_{method}', all_A_test)

    if do_stats:
      res_df, all_stats_true, all_stats_methods = stats.compare_all(all_A_test, all_generated_graphs, None)

      if save_res:
        np.savez_compressed(fname, res_df=res_df, res_df_columns=res_df.columns.values, res_df_index=res_df.index.values, all_stats_true=all_stats_true, all_stats_methods=all_stats_methods)
        print(f'Saved to {fname}')

