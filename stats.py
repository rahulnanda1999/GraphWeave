import pandas as pd 
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.linalg import pinv, eigvalsh
import scipy.stats
import networkx as nx
import subprocess as sp
import concurrent.futures
import load_helper
import os
import numpy as np
import random


COUNT_START_STR = 'orbit counts: \n'

def compare_all(all_A_true, all_A_ours, dir_path_others):
  print('Loading: ', end='')
  all_A_methods = load_helper.load_all_others(dir_path_others) if dir_path_others is not None else {}
  all_A_methods.update({'GraphWeave':all_A_ours})
  for k, v in all_A_methods.items():
    print(f'{k} ({len(v)} graphs)', end=' ')
  print()

  print('Stats:', end=' ')
  print('GroundTruth', end=' ')
  all_stats_true = compute_all_stats(all_A_true)
  all_stats_methods = compute_all_stats_for_all_methods(all_A_methods)

  print('Distances:', end=' ')
  res_df = wass_dist_all_methods(all_stats_true, all_stats_methods)
  print(res_df)
  return res_df, all_stats_true, all_stats_methods

def find_dist(strue, spred):
  return scipy.stats.wasserstein_distance(strue, spred)

def kernel_parallel_unpacked(x, samples2):
    d = 0
    for s2 in samples2:
        d += find_dist(x, s2)
    return d

def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)

def disc(samples_true, samples_pred, is_parallel=True):
    ''' Discrepancy between 2 samples
    '''
    d = 0
    if not is_parallel:
        for s1 in samples_true:
            for s2 in samples_pred:
                d += find_dist(s1, s2)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(kernel_parallel_worker, 
                    [(s1, samples_pred) for s1 in samples_true]):
                d += dist
    d /= len(samples_true) * len(samples_pred)
    return d

def disc_all(arr_dict_true, arr_dict_pred):
  all_res = {}
  for measure in arr_dict_true[0].keys():
    all_res[measure] = disc([d[measure] for d in arr_dict_true],
                            [d[measure] for d in arr_dict_pred])
  return Series(all_res)

def wass_dist_all_methods(arr_dict_true, dict_all_methods):
  res_df = {}
  print('GroundTruth', end=' ')
  res_df['true'] = disc_all(arr_dict_true, arr_dict_true)
  for method, arr_dict_method in dict_all_methods.items():
    print(method, end=' ')
    res_df[f'inter {method}'] = disc_all(arr_dict_true, arr_dict_method)
    res_df[f'ratio {method}'] = ((1e-10+res_df[f'inter {method}']) / (1e-10+res_df['true'])-1).abs()
  print()
  res_df = DataFrame(res_df).sort_index(axis=1)
  return res_df

def compute_stats(A, counter=0, num_cuts=500, num_pairs=500, seed=0):
  np.random.seed(seed)
  random.seed(seed)
  res = {}
  G = nx.from_numpy_array(A)
  res['pagerank'] = np.array(list(nx.pagerank(G).values()))
  res['closeness'] = np.array(list(nx.closeness_centrality(G).values()))
  res['betweenness'] = np.array(list(nx.betweenness_centrality(G).values()))
  res['harmonic'] = np.array(list(nx.harmonic_centrality(G).values()))
  res['connected'] = np.array([len(c) for c in nx.connected_components(G)])
  res['communities'] = np.array([len(c) for c in nx.algorithms.community.louvain_communities(G)])

  largest_cc = max(nx.connected_components(G), key=len)
  subgraph = G.subgraph(largest_cc)

  res['cut_sizes'], partitions = list(zip(*[nx.approximation.randomized_partitioning(G, seed=i) for i in range(num_cuts)]))

  all_conductances = []
  for p in partitions:
    try:
      this_cond = nx.conductance(G, *p)
      all_conductances.append(this_cond)
    except:
      pass
  res['conductance'] = np.array(all_conductances)
  res['modularity'] = np.array([nx.community.modularity(G, p) for p in partitions])
  res['degree'] = np.array(nx.degree_histogram(G))
  res['clustering'] = np.array(list(nx.clustering(G).values()))
  orbit_counts = orca(G, counter)
  res['orbit'] = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
  
  res['spectral'] = eigvalsh(nx.normalized_laplacian_matrix(G).todense())

  nx.set_edge_attributes(G, 1, name='weight')
  res['maxflow'] = np.array([nx.maximum_flow_value(G, *(np.random.choice(G.number_of_nodes(), 2, replace=False)), capacity='weight') for _ in range(num_pairs)])

  random_pairs = np.random.choice(subgraph.number_of_nodes(), size=(num_pairs,2))
  resistance_mat = DataFrame(nx.resistance_distance(subgraph)).values
  res['resistance'] = resistance_mat[random_pairs[:,0], random_pairs[:,1]]
  
  return res

def compute_stats_packed(t):
  return compute_stats(*t)

def compute_all_stats(arrA, is_parallel=True):
  if not is_parallel:
    all_res = [compute_stats(A) for A in arrA]
  else:
    all_res = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
      for res in executor.map(compute_stats_packed, [(A, i) for i, A in enumerate(arrA)]): 
        all_res.append(res)
  return all_res 

def compute_all_stats_for_all_methods(method_arrGraph_dict):
  res = {}
  for method, arrGraph in method_arrGraph_dict.items():
    print(method, end=' ')
    res[method] = compute_all_stats(arrGraph)
  print()
  return res

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges

def orca(graph, counter):
    tmp_fname = f'orca/tmp{counter}.txt'
    f = open(tmp_fname, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    output = sp.check_output(['./orca/orca', 'node', '4', tmp_fname, 'std'])
    output = output.decode('utf8').strip()
    
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ') ))
          for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts
