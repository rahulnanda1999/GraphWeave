import numpy as np
import networkx as nx
import pickle
import os

def load_digress_graphs(filename):
  all_A = []
  with open(filename) as fp:
    start_adj = False
    arr_edges = []
    for line in fp:
      line = line.rstrip()
      if len(line)==0:
        all_A.append(np.vstack(arr_edges))
        arr_edges = []
        start_adj = False
      elif line[:2] == 'N=':
        n = int(line[2:])
      elif line[0] == 'E':
        start_adj = True
      elif start_adj:
        arr_edges.append(np.array([int(x) for x in line.split()]))
  return all_A

def load_pkl(filename):
  all_A = []
  with open(filename, 'rb') as fp:
    X = pickle.load(fp)
    all_A = [nx.adjacency_matrix(G) for G in X]
  return all_A

def load_all_others(dir_path):
  all_A_others = {}
  try:
    for f in os.listdir(dir_path):
      fname = f'{dir_path}/{f}'
      if f[:7].lower()=='digress':
        all_A_others['DiGress'] = load_digress_graphs(fname)
      elif f[:10].lower() == 'fast_graph' or f[:5].lower() == 'fggsd':
        all_A_others['FGGSD'] = load_pkl(fname)
      elif f[:4].lower() == 'gdss':
        all_A_others['GDSS'] = load_pkl(fname)
      elif f[:8].lower() == 'graphrnn':
        all_A_others['GraphRNN'] = load_pkl(fname)
      elif 'grasp' in f:
        all_A_others['GRASP'] = load_pkl(fname)
  except:
    pass

  return all_A_others

