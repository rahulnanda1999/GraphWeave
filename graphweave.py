import torch
import scipy
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import transformer
import matplotlib.pyplot as plt
from scipy.stats import norm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.optimize import minimize
from pandas import Series, DataFrame
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
import sys
import pickle

def my_generate_graph(num_nodes, graph_type='WS', params={}):
  if graph_type=='WS':
    params_to_use = {'k':4, 'p':0.3}
    params_to_use.update(params)
    G = nx.watts_strogatz_graph(num_nodes, **params_to_use)
  elif graph_type=='BA':
    params_to_use = {'m':3}
    params_to_use.update(params)
    G = nx.barabasi_albert_graph(n=num_nodes, **params_to_use)
  elif graph_type in ['SBM', 'SBMPretty']:
    if graph_type == 'SBM':
      params_to_use = {'cluster_fracs':[0.5, 0.3, 0.2], 'p':0.8, 'q':0.3}
    elif graph_type == 'SBMPretty':
      params_to_use = {'cluster_fracs':[0.5, 0.3, 0.2], 'p':0.8, 'q':0.1}
    params_to_use.update(params)
    sizes = (np.array(params_to_use['cluster_fracs']) * num_nodes).astype(int)
    connection_probs = np.ones((len(sizes), len(sizes))) * params_to_use['q']
    np.fill_diagonal(connection_probs, params_to_use['p'])
    G = nx.stochastic_block_model(sizes=sizes, p=connection_probs)
  elif graph_type == 'RGlikeSBM':
    params_to_use = {'cluster_fracs':[0.5, 0.3, 0.2], 'p':0.8, 'q':0.3}
    params_to_use.update(params)
    cf = np.array(params_to_use['cluster_fracs'])
    B = params_to_use['q'] * np.ones((len(cf), len(cf))) + (params_to_use['p']-params_to_use['q']) * np.identity(len(cf))
    expdegs = ((cf[:,None] * B).sum(axis=0) * num_nodes).astype(int)
    d = np.repeat(expdegs, (cf * num_nodes).astype(int))
    G = nx.expected_degree_graph(d, selfloops=False)

  G = G.to_directed()
  
  M = np.zeros((num_nodes, num_nodes))
  for node in G.nodes():
    edges = list(G.out_edges(node))
    if edges:
      for src, tgt in edges:
        prob = 1/len(edges)
        M[src, tgt] = prob
  return G, M

def get_L(G, alpha=0):
#  A = nx.adjacency_matrix(G)#.todense()
  A = (1-alpha)*nx.adjacency_matrix(G) + alpha * scipy.sparse.identity(G.number_of_nodes())
  d_minushalf = 1/np.sqrt(A.sum(axis=1))
  L = A * d_minushalf[:,None] * d_minushalf[None,:]
  return L


def create_graphs_df(num_nodes, num_graphs, num_steps, deg_powers=[0, 1.0], use_norm_lap=True, norm_lap_alpha=0, seed=0, graph_type='WS', params={}, norm_start_vec=True):
  np.random.seed(seed)
  all_M_steps = []
  all_A, all_M, all_L = [], [], []

  other_datasets = {'Cora':'cora_ours',
                    }
  if graph_type in other_datasets:
    with open(f'data/{other_datasets[graph_type]}.pkl', 'rb') as fp:
      all_G = pickle.load(fp)


  for counter in range(num_graphs):
    if graph_type in other_datasets:
      G = all_G[counter]
      G.remove_edges_from(nx.selfloop_edges(G))
    else:
      G, M = my_generate_graph(num_nodes, graph_type=graph_type, params=params)
      all_M.append(M)
    this_A = nx.adjacency_matrix(G)
    all_A.append(this_A)

    if use_norm_lap:
      L = get_L(G, norm_lap_alpha)
      all_L.append(L)

    deg = this_A.sum(axis=0)
    deg_prime = (1-norm_lap_alpha)*deg + norm_lap_alpha
    vT = np.ones((len(deg_powers), len(deg)))
    for i, deg_pwr in enumerate(deg_powers):
      vT[i] = np.power(deg, -deg_pwr)
      if norm_start_vec:
        vT[i] *= 1/np.sum(vT[i]) * len(deg)
      else:
        vT[i] *= np.sqrt(np.sum(deg_prime))/np.sum(vT[i] * np.sqrt(deg_prime))

    graph_steps = {f'all_v_T_M_0':vT}
    for step in range(1, num_steps+1):
      vT = vT @ L #_pwr if use_norm_lap else vT @ M
      graph_steps[f'all_v_T_M_{step}'] = vT #.flatten() #.tolist()
    all_M_steps.append(graph_steps)

  df = pd.DataFrame(all_M_steps)
  return df, all_M, all_L, all_A


def get_Q_from_df(df, cat_mult=3):
  # len(deg_power) * num_graphs * num_steps * num_nodes
  Z = [np.swapaxes(np.stack(df.iloc[i].values), 0, 1) for i in range(len(df))]
  t = np.max([z.shape[2] for z in Z])
  Z2 = [np.pad(z, ((0,0),(0,0),(0, t-z.shape[2])), constant_values=np.nan) for z in Z]
  Q = np.swapaxes(np.stack(Z2), 0, 1)
  mask = np.isnan(Q)
  Qc = ((Q-np.nanmean(Q))/np.nanstd(Q)*cat_mult).round()
  min_val = np.nanmin(Qc)
  Qc -= min_val
  Qc[mask] = 0
  Qc = Qc.astype(int)
  return Q, Qc, min_val, np.nanmean(Q), np.nanstd(Q)

def train_test_split(Q, Qc, all_A, test_frac=0.3):
  num_graphs = Qc.shape[1]
  train_indices = np.random.choice(np.arange(num_graphs), size=max(1, int((1-test_frac)*num_graphs)), replace=False)
  train_mask = np.zeros(num_graphs, dtype=bool)
  train_mask[train_indices] = True
  return Q[:,train_mask,:,:], Q[:,~train_mask,:,:], Qc[:,train_mask,:,:], Qc[:,~train_mask,:,:], [all_A[i] for i in train_indices], [all_A[i] for i in np.where(~train_mask)[0]]

def generate_tensors_vals(Q, Qc):
  Qn_step = np.tile(np.arange(Qc.shape[2]), Qc.shape[0]*Qc.shape[1]).reshape(Qc.shape[:3])
  Qn_start = np.repeat(np.arange(Qc.shape[0]), Qc.shape[1]*Qc.shape[2]).reshape(Qc.shape[:3])
  target_tensor = torch.tensor(Q[:,:,:-1,:].reshape(-1, Q.shape[3]), dtype=torch.float32)
  source_cat_tensor = torch.tensor(Qc[:,:,1:,:].reshape(-1, Qc.shape[3]), dtype=torch.int64)
  source_val_tensor = torch.tensor(Q[:,:,1:,:].reshape(-1, Q.shape[3]), dtype=torch.float32)
  index_step_tensor = torch.tensor(Qn_step[:,:,:-1].reshape(-1), dtype=torch.int64)
  index_start_tensor = torch.tensor(Qn_start[:,:,:-1].reshape(-1), dtype=torch.int64)
  return source_val_tensor, source_cat_tensor, target_tensor, index_step_tensor, index_start_tensor

def do_train_vals(Q, Qc, batch_size=32, embedding_dim=8, num_heads=4, num_layers=2, num_epochs=100, learning_rate=1e-3, dropout=0.1):
  num_categories = np.amax(Qc) + 1
  num_index = Qc.shape[2]
  source_val_tensor, source_cat_tensor, target_tensor, index_step_tensor, index_start_tensor = generate_tensors_vals(Q, Qc)
  dataset = TensorDataset(source_val_tensor, source_cat_tensor, target_tensor, index_step_tensor, index_start_tensor)
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = transformer.GraphTransformerVals(num_categories, num_index, embedding_dim, num_heads, num_layers, dropout).to(device)
  criterion = nn.MSELoss(reduction='none')
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
      
  model.train()
  for epoch in range(num_epochs):
    train_loss = 0
    loss_arr, baseline_loss_arr, idx_step_arr = [], [], []
    tmp_counter = 0
    for src_val, src_cat, tgt, idx_step, idx_start in data_loader:
      this_idx = idx_step.numpy()
      src_val, src_cat, tgt, idx_step, idx_start = src_val.to(device), src_cat.to(device), tgt.to(device), idx_step.to(device), idx_start.to(device)

      optimizer.zero_grad()
      output_pred = model(src_val, src_cat, idx_step, idx_start)

      mask = src_val.isnan()

      all_loss = criterion(output_pred.view(-1), (torch.where(mask, 0, tgt-src_val)).view(-1))
      
      loss = all_loss[(~mask).view(-1)].mean()
      loss.backward()
      optimizer.step()
      train_loss += loss.item()
      
      baseline_loss = criterion(torch.zeros_like(src_val).view(-1), (tgt-src_val).view(-1))
      baseline_loss_arr.extend(np.nanmean(baseline_loss.detach().cpu().numpy().reshape(-1, Q.shape[3]), axis=1))
      loss_arr.extend(np.nanmean(all_loss.detach().cpu().numpy().reshape(-1, Q.shape[3]), axis=1))
      idx_step_arr.extend(this_idx)

    if epoch % 20 == 0 or epoch==num_epochs-1:
      this_df = DataFrame({'idx_step':idx_step_arr, 'loss':loss_arr, 'baseline':baseline_loss_arr})
      z = this_df.groupby('idx_step')[['loss', 'baseline']].mean()
      z['ratio'] = z['loss']/z['baseline']
      z['str'] = z['loss'].map(np.sqrt).round(3).astype(str) + ', ' + z['ratio'].round(2).astype(str)
      print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(data_loader):4.4f}, {dict(z['str'])}")

  return model, num_categories

def create_tensors_helper(vecs, m, s, cat_mult, min_val, num_categories, step):
  # vecs = len(deg_powers) * num_nodes
  Qc_vecs = ((vecs-m)/s*cat_mult).round()
  Qc_vecs -= min_val
  Qc_vecs[Qc_vecs >= num_categories] = num_categories-1  # We keep Q as is
  Qc_vecs[Qc_vecs < 0] = 0
  Qc_vecs = Qc_vecs.astype(int)

  source_cat_tensor = torch.tensor(Qc_vecs, dtype=torch.int64)
  source_val_tensor = torch.tensor(vecs, dtype=torch.float32)
  index_step_tensor = torch.ones(len(vecs), dtype=torch.int64) * step
  index_start_tensor = torch.arange(len(vecs), dtype=torch.int64)
  return source_val_tensor, source_cat_tensor, index_step_tensor, index_start_tensor


def generate_graph_from_degrees(model, degrees, deg_powers, alpha, m, s, cat_mult, min_val, num_categories, num_steps, norm_start_vec=True):
  num_nodes = len(degrees)

  if norm_start_vec:
    dprime = alpha + (1-alpha) * degrees
    fd = np.power(degrees.astype(float)[None,:], np.array(deg_powers)[:,None])
    this_pred = (len(degrees)/np.sum(fd, axis=1) * np.sum(fd * np.sqrt(dprime)[None,:], axis=1)/np.sum(dprime))[:,None] * np.sqrt(dprime)[None,:]
  else:
    start_vec = np.sqrt(alpha + (1-alpha)*degrees)
    start_vec = start_vec / np.sqrt(np.square(start_vec).sum())
    this_pred = np.tile(start_vec, len(deg_powers)).reshape(len(deg_powers),-1)
    # this_pred is len(deg_powers) * len(degrees)

  V_t = np.zeros((len(deg_powers)*num_steps, num_nodes))
  V_tplus1 = np.zeros((len(deg_powers)*num_steps, num_nodes))


  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  criterion = nn.MSELoss()

  model.eval()
  for step in range(num_steps-1, -1, -1):
    V_tplus1[step*len(deg_powers):(step+1)*len(deg_powers)] = this_pred
    source_val_tensor, source_cat_tensor, index_step_tensor, index_start_tensor = \
        create_tensors_helper(vecs=this_pred, m=m, s=s, cat_mult=cat_mult, min_val=min_val,
                              num_categories=num_categories, step=step)

    dataset = TensorDataset(source_val_tensor, source_cat_tensor, index_step_tensor, index_start_tensor)
    data_loader = DataLoader(dataset, batch_size=len(source_val_tensor), shuffle=False)

    for src_val, src_cat, idx_step, idx_start in data_loader:
      src_val, src_cat, idx_step, idx_start = src_val.to(device), src_cat.to(device), idx_step.to(device), idx_start.to(device)
      output_pred = model(src_val, src_cat, idx_step, idx_start)
      this_pred = (src_val + output_pred.squeeze(2)).detach().cpu().numpy()
      this_pred = this_pred / (this_pred.sum(axis=1))[:,None] * num_nodes
      V_t[step*len(deg_powers):(step+1)*len(deg_powers)] = this_pred
      
  return V_t, V_tplus1
