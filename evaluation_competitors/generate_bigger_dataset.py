import networkx as nx
import numpy as np
import argparse
import os
import random
import pathlib
from scipy.spatial import Delaunay
from torch_geometric.utils import to_networkx
from torch.utils.data import Dataset
import torch.nn.functional as F
import graph_tool.all as gt
import matplotlib.pyplot as plt
import torch
from compute_metrics import is_sbm_graph

class SBMDataset(Dataset):
    def __init__(self, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False, max_comm_size=40):
        filename = f'sbm_{n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            assert False
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample
            for seed in range(n_graphs):
                n_comunities = np.random.random_integers(2, 5)
                comunity_sizes = np.random.random_integers(20, max_comm_size, size=n_comunities)
                probs = np.ones([n_comunities, n_comunities]) * 0.005
                probs[np.arange(n_comunities), np.arange(n_comunities)] = 0.3
                G = nx.stochastic_block_model(comunity_sizes, probs, seed=seed)
                adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                self.adjs.append(adj)
                self.n_nodes.append(len(G.nodes()))
            self.n_max = max(self.n_nodes)
            #torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        # splits
        random.seed(42)
        graphs_len = len(self.adjs)
        idxs = list(range(graphs_len))
        random.shuffle(idxs)
        self.test_idxs = idxs[int(0.8 * graphs_len):]
        self.val_idxs = idxs[0:int(0.2*graphs_len)]
        self.train_idxs = idxs[int(0.2*graphs_len):int(0.8*graphs_len)]

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        return graph
    

class PlanarDataset(Dataset):
    def __init__(self, n_nodes, n_graphs, k, same_sample=False, SON=False, ignore_first_eigv=False):
        filename = f'planar_{n_nodes}_{n_graphs}{"_same_sample" if same_sample else ""}.pt'
        self.k = k
        self.ignore_first_eigv = ignore_first_eigv
        if os.path.isfile(filename):
            assert False
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample , self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.eigvals = []
            self.eigvecs = []
            self.n_nodes = []
            self.max_eigval = 0
            self.min_eigval = 0
            self.same_sample = same_sample
            for i in range(n_graphs):
                # Generate planar graphs using Delauney traingulation
                points = np.random.rand(n_nodes,2)
                tri = Delaunay(points)
                adj = np.zeros([n_nodes,n_nodes])
                for t in tri.simplices:
                    adj[t[0], t[1]] = 1
                    adj[t[1], t[2]] = 1
                    adj[t[2], t[0]] = 1
                    adj[t[1], t[0]] = 1
                    adj[t[2], t[1]] = 1
                    adj[t[0], t[2]] = 1
                G = nx.from_numpy_array(adj)
                adj = torch.from_numpy(adj).float()
                L = nx.normalized_laplacian_matrix(G).toarray()
                L = torch.from_numpy(L).float()
                eigval, eigvec = torch.linalg.eigh(L)

                self.eigvals.append(eigval)
                self.eigvecs.append(eigvec)
                self.adjs.append(adj)
                self.n_nodes.append(len(G.nodes()))
                max_eigval = torch.max(eigval)
                if max_eigval > self.max_eigval:
                    self.max_eigval = max_eigval
                min_eigval = torch.min(eigval)
                if min_eigval < self.min_eigval:
                    self.min_eigval = min_eigval
            self.n_max = n_nodes
            # torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved')

        # splits
        random.seed(42)
        graphs_len = len(self.adjs)
        idxs = list(range(graphs_len))
        random.shuffle(idxs)
        self.test_idxs = idxs[int(0.8 * graphs_len):]
        self.val_idxs = idxs[0:int(0.2*graphs_len)]
        self.train_idxs = idxs[int(0.2*graphs_len):int(0.8*graphs_len)]

        self.max_k_eigval = 0
        for eigv in self.eigvals:
            if eigv[self.k] > self.max_k_eigval:
                self.max_k_eigval = eigv[self.k].item()

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        eigvals = self.eigvals[idx]
        eigvecs = self.eigvecs[idx]
        if self.ignore_first_eigv:
            eigvals = eigvals[1:]
            eigvecs = eigvecs[:,1:]
            size_diff += 1
        graph["eigval"] = F.pad(eigvals, [0, max(0, self.n_max - eigvals.size(0))])
        graph["eigvec"] = F.pad(eigvecs, [0, size_diff, 0, size_diff])
        graph["mask"] = F.pad(torch.ones_like(self.adjs[idx]), [0, size_diff, 0, size_diff]).long()
        return graph


class GridDataset(Dataset):
    def __init__(self, grid_start=10, grid_end=20, same_sample=False):
        filename = f'data/grids_{grid_start}_{grid_end}{"_same_sample" if same_sample else ""}.pt'

        if os.path.isfile(filename):
            assert False
            self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max = torch.load(filename)
            print(f'Dataset {filename} loaded from file')
        else:
            self.adjs = []
            self.n_nodes = []
            self.same_sample = same_sample
            for i in range(grid_start, grid_end):
                for j in range(grid_start, grid_end):
                    G = nx.grid_2d_graph(i, j)
                    adj = torch.from_numpy(nx.adjacency_matrix(G).toarray()).float()
                    self.adjs.append(adj)
                    self.n_nodes.append(len(G.nodes()))
            self.n_max = (grid_end - 1) * (grid_end - 1)
            # torch.save([self.adjs, self.eigvals, self.eigvecs, self.n_nodes, self.max_eigval, self.min_eigval, self.same_sample, self.n_max], filename)
            print(f'Dataset {filename} saved with {len(self.adjs)} graphs')

        # splits
        random.seed(42)
        graphs_len = len(self.adjs)
        idxs = list(range(graphs_len))
        random.shuffle(idxs)
        self.test_idxs = idxs[int(0.8 * graphs_len):]
        self.val_idxs = idxs[0:int(0.2*graphs_len)]
        self.train_idxs = idxs[int(0.2*graphs_len):int(0.8*graphs_len)]

    def __len__(self):
        return len(self.adjs)

    def __getitem__(self, idx):
        if self.same_sample:
            idx = self.__len__() - 1
        graph = {}
        graph["n_nodes"] = self.n_nodes[idx]
        size_diff = self.n_max - graph["n_nodes"]
        graph["adj"] = F.pad(self.adjs[idx], [0, size_diff, 0, size_diff])
        return graph


name = "grid"

if name == "sbm":
    G = SBMDataset(5, k=4, max_comm_size=100)

    print("\n\nEVALUATING")
    print([is_sbm_graph(nx.from_numpy_matrix(G[i]["adj"].numpy()), refinement_steps=1000) for i in range(len(G))])
elif name == "planar":
    G = PlanarDataset(n_nodes=64, n_graphs=10, k=2)

    print("\n\nEVALUATING")
    print([nx.is_planar(nx.from_numpy_matrix(G[i]["adj"].numpy())) for i in range(len(G))])
else:
    G = GridDataset()
    print(G.train_idxs)
    print(len(G.train_idxs), len(G.val_idxs), len(G.test_idxs))
