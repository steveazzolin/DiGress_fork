import random
import networkx as nx
import numpy as np
import argparse
import os
import pickle as pkl
import pathlib
from torch.utils.data import Dataset #, InMemoryDataset, download_url
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
import torch
import graph_tool.all as gt
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# evaluation setup code from GraphRNN
from mmd import compute_mmd
import mmd_digress
from compute_metrics import get_orbit, get_clustering, get_degs, spectral_stats, motif_stats, is_sbm_graph, eval_fraction_unique_non_isomorphic_valid, eval_fraction_isomorphic


from src.datasets.spectre_dataset import SpectreGraphDataset, SpectreGraphDataModule
from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
#from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

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
            self.n_max = max(self.n_nodes)

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
    




# class SpectreGraphDataset(InMemoryDataset):
#     def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
#         self.sbm_file = 'sbm_200.pt'
#         self.planar_file = 'planar_64_200.pt'
#         self.comm20_file = 'community_12_21_100.pt'
#         self.dataset_name = dataset_name
#         self.split = split
#         self.num_graphs = 200
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return ['train.pt', 'val.pt', 'test.pt']

#     @property
#     def processed_file_names(self):
#             return [self.split + '.pt']

#     def download(self):
#         """
#         Download raw qm9 files. Taken from PyG QM9 class
#         """
#         if self.dataset_name == 'sbm':
#             raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt'
#         elif self.dataset_name == 'planar':
#             raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt'
#         elif self.dataset_name == 'comm20':
#             raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt'
#         else:
#             raise ValueError(f'Unknown dataset {self.dataset_name}')
#         file_path = download_url(raw_url, self.raw_dir)

#         adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

#         g_cpu = torch.Generator()
#         g_cpu.manual_seed(0)

#         test_len = int(round(self.num_graphs * 0.2))
#         train_len = int(round((self.num_graphs - test_len) * 0.8))
#         val_len = self.num_graphs - train_len - test_len
#         indices = torch.randperm(self.num_graphs, generator=g_cpu)
#         print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
#         train_indices = indices[:train_len]
#         val_indices = indices[train_len:train_len + val_len]
#         test_indices = indices[train_len + val_len:]

#         train_data = []
#         val_data = []
#         test_data = []

#         for i, adj in enumerate(adjs):
#             if i in train_indices:
#                 train_data.append(adj)
#             elif i in val_indices:
#                 val_data.append(adj)
#             elif i in test_indices:
#                 test_data.append(adj)
#             else:
#                 raise ValueError(f'Index {i} not in any split')

#         torch.save(train_data, self.raw_paths[0])
#         torch.save(val_data, self.raw_paths[1])
#         torch.save(test_data, self.raw_paths[2])


#     def process(self):
#         file_idx = {'train': 0, 'val': 1, 'test': 2}
#         raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

#         data_list = []
#         for adj in raw_dataset:
#             n = adj.shape[-1]
#             X = torch.ones(n, 1, dtype=torch.float)
#             y = torch.zeros([1, 0]).float()
#             edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
#             edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
#             edge_attr[:, 1] = 1
#             num_nodes = n * torch.ones(1, dtype=torch.long)
#             data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
#                                              y=y, n_nodes=num_nodes)
#             data_list.append(data)

#             if self.pre_filter is not None and not self.pre_filter(data):
#                 continue
#             if self.pre_transform is not None:
#                 data = self.pre_transform(data)

#             data_list.append(data)
#         torch.save(self.collate(data_list), self.processed_paths[0])



# class SpectreGraphDataModule(AbstractDataModule):
#     def __init__(self, cfg, n_graphs=200):
#         self.cfg = cfg
#         self.datadir = cfg.dataset.datadir
#         base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
#         root_path = os.path.join(base_path, self.datadir)


#         datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
#                                                  split='train', root=root_path),
#                     'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
#                                         split='val', root=root_path),
#                     'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
#                                         split='test', root=root_path)}
#         # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
#         self.datasets_steve = datasets
#         super().__init__(cfg, datasets)
#         self.inner = self.train_dataset

#     def __getitem__(self, item):
#         return self.inner[item]


# class SpectreDatasetInfos(AbstractDatasetInfos):
#     def __init__(self, datamodule, dataset_config):
#         self.datamodule = datamodule
#         self.name = 'nx_graphs'
#         self.n_nodes = self.datamodule.node_counts()
#         self.node_types = torch.tensor([1])               # There are no node types
#         self.edge_types = self.datamodule.edge_counts()
#         super().complete_infos(self.n_nodes, self.node_types)

# class SBMDataModule(AbstractDataModule):
#     def __init__(self, cfg, n_graphs=200):
#         self.cfg = cfg
#         self.datadir = cfg.dataset.datadir
#         base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
#         root_path = os.path.join(base_path, self.datadir)


#         datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
#                                                  split='train', root=root_path),
#                     'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
#                                         split='val', root=root_path),
#                     'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
#                                         split='test', root=root_path)}
#         # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

#         super().__init__(cfg, datasets)
#         self.inner = self.train_dataset

#     def __getitem__(self, item):
#         return self.inner[item]


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def loader_to_nx(loader):
    networkx_graphs = []
    for i in range(len(loader)):
        data = loader[i]
        # data_list = batch.to_data_list()
        # for j, data in enumerate(data_list):
        networkx_graphs.append(to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True,
                                            remove_self_loops=True))
    return networkx_graphs

def get_dataset_dir(name):
    if name == "planar":
        datadir = "data/planar/"
    elif name == "sbm":
        datadir = "data/sbm/"
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    return os.path.join(base_path, datadir)

def get_reference_graphs(name, path):
    if name in ["sbm", "planar"]:
        # config = dotdict({
        #     "dataset": dotdict({
        #         "name": name,
        #         "datadir": get_dataset_dir(name)
        #     }),
        #     "general": dotdict({
        #         "name": ""
        #     }),
        #     "train": dotdict({
        #         "num_workers": 0
        #     })
        # })
        # dataloaders = SpectreGraphDataModule(config)
        # test_reference_graphs = loader_to_nx(dataloaders.datasets_steve["test"])
        # train_reference_graphs = loader_to_nx(dataloaders.datasets_steve["train"])
        config = dotdict({
            "dataset": dotdict({
                "name": name,
                "datadir": get_dataset_dir(name),
                'remove_h': None,
            }),
            "general": dotdict({
                'name': 'planar_metric_debug',
                'wandb': 'offline',
                'gpus': 1,
                'resume': None,
                'test_only': '/home/steve.azzolin/DiGress_fork/checkpoints/checkpoint_planar.ckpt',
                'sample_bigger_graphs': 0,
                'check_val_every_n_epochs': 5,
                'sample_every_val': 4,
                'val_check_interval': None,
                'samples_to_generate': 512,
                'samples_to_save': 20,
                'chains_to_save': 1,
                'log_every_steps': 50,
                'number_chain_steps': 50,
                'final_model_samples_to_generate': 10000,
                'final_model_samples_to_save': 30,
                'final_model_chains_to_save': 20,
                'evaluate_all_checkpoints': False
            }),
            "train": dotdict({
                "num_workers": 0,
                'n_epochs': 1000,
                'batch_size': 32,
                'lr': 0.0002,
                'clip_grad': None,
                'save_model': True,
                'num_workers': 0,
                'ema_decay': 0,
                'progress_bar': False,
                'weight_decay': 1e-12,
                'optimizer': 'adamw',
                'seed': 0
            }),
            'model': dotdict({
                'type': 'discrete',
                'transition': 'marginal',
                'model': 'graph_tf',
                'diffusion_steps': 500,
                'diffusion_noise_schedule': 'cosine',
                'n_layers': 5,
                'extra_features': 'all',
                'hidden_mlp_dims': {'X': 256, 'E': 128, 'y': 128},
                'hidden_dims': {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128},
                'lambda_train': [5, 0]
            }),
        })
        dataloaders = SpectreGraphDataModule(config)
        if name == "planar":
            digress_metric = PlanarSamplingMetrics(dataloaders)
        elif name == "sbm":
            digress_metric = SBMSamplingMetrics(dataloaders)
        test_reference_graphs = digress_metric.test_graphs
        train_reference_graphs = digress_metric.train_graphs
    elif name == "grid":
        graphs = []
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))
        random.seed(123)
        random.shuffle(graphs)
        graphs_len = len(graphs)
        test_reference_graphs = graphs[int(0.8 * graphs_len):]
        train_reference_graphs = graphs[0:int(0.8*graphs_len)]
    return test_reference_graphs, train_reference_graphs
    # if "planar" in name:
    #     G = PlanarDataset(n_nodes=64 + 0, n_graphs=80, k=2)
    #     test_reference_graphs = [nx.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
    #     train_reference_graphs = [nx.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
    # elif "sbm" in name:
    #     G = SBMDataset(200, k=4, max_comm_size=100)
    #     test_reference_graphs = [nx.convert_matrix.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
    #     train_reference_graphs = [nx.convert_matrix.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
    # return train_reference_graphs, test_reference_graphs


def pick_connected_component_new(G):
    adj_list = list(map(list, iter(G.adj.values())))
    for id,adj in enumerate(adj_list):
        id_min = min(adj)
        if id<id_min and id>=1:
            break
    node_list = list(range(id)) # only include node prior than node "id"

    G = G.subgraph(node_list)
    G = max((G.subgraph(c) for c in nx.connected_components(G)), key=len)
    return G

def main(args):
    with open(args.path, "rb") as f:
        generated = pkl.load(f)

    print(f"Found {len(generated)} generated graphs")
    print("Shape first generated graph: ", len(generated[0].nodes()))

    reference, train = get_reference_graphs(args.dataset_name, args.path) #[:len(generated)]
    print(f"Found {len(reference)} reference graphs")
    print("Shape first generated graph: ", len(reference[0].nodes()))
    print()

    c = 0
    for g in generated:
        if len(g.nodes()) == 0:
            c += 1
    print("Graph w/o nodes", c)
    generated = [g for g in generated if len(g.nodes()) > 0]
    print(len(generated))

    # evaluate metrics
    s1 = get_degs(reference)
    s2 = get_degs(generated)
    degs = mmd_digress.compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False)
    print("Degree stat: ", degs)

    s1 = get_orbit(reference)
    s2 = get_orbit(generated)
    orbit = mmd_digress.compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False, is_hist=False, sigma=30.0)
    print("Orbit stat: ", orbit)

    s1 = get_clustering(reference)
    s2 = get_clustering(generated)
    clust = mmd_digress.compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False, sigma=1.0 / 10)
    print("Clustering stat: ", clust)

    spectre = spectral_stats(reference, generated, is_parallel=False, n_eigvals=-1, compute_emd=False)
    print("Spectre stat: ", spectre)

    motif = motif_stats(reference, generated, motif_type='4cycle', ground_truth_match=None, bins=100, compute_emd=False)
    print("Motif stat: ", motif)

    # frac_unique, frac_unique_non_isomorphic, fraction_unique_non_isomorphic_valid = eval_fraction_unique_non_isomorphic_valid(
    #         generated, reference, is_sbm_graph) # TODO: maybe change test o train!
    frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(generated, train)
    print("Fraction non-iso graphs: ", frac_non_isomorphic)

    acc_ref, acc_gen = 0, 0
    for i in range(max(len(generated), len(reference))):
        if args.dataset_name == "planar":
            if i < len(generated) and nx.is_planar(generated[i]):
                acc_gen += 1
            if i < len(reference) and nx.is_planar(reference[i]):
                acc_ref += 1
        elif args.dataset_name == "sbm":
            if i < len(generated) and is_sbm_graph(generated[i]):
                acc_gen += 1
            if i < len(reference) and is_sbm_graph(reference[i]):
                acc_ref += 1
    print("Validity gen: ", round(acc_gen / len(generated), 3))
    print("Validity ref: ", round(acc_ref / len(reference), 3))

    # plot sample graphs
    for i in range(10):
        nx.draw(reference[i])
        plt.savefig(f"plots/{args.dataset_name}/{i}.png")
        plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)  
    parser.add_argument('--path', type=str, required=True, help="path to generated graphs")
    args = parser.parse_args()
    main(args)
