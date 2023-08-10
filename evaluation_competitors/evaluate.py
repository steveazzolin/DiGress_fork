import networkx as nx
import numpy as np
import argparse
import os
import pathlib
from torch.utils.data import Dataset
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
from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, EgoSamplingMetrics, GridSamplingMetrics


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
    elif name == "grid":
        datadir = "data/grid/"
    elif "grid_small" in name:
        datadir = "data/grid/"
    elif name == "ego":
        datadir = "data/ego/"
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
    return os.path.join(base_path, datadir)

def get_reference_graphs(name, path):
    if "false" in path or "0" in path: # size sampled from train distrib
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
        elif name == "grid" or "grid_small" in name:
            digress_metric = GridSamplingMetrics(dataloaders)
        elif name == "ego":
            digress_metric = EgoSamplingMetrics(dataloaders)
        test_reference_graphs = digress_metric.test_graphs
        train_reference_graphs = digress_metric.train_graphs
        # test_reference_graphs = loader_to_nx(dataloaders.datasets_steve["test"])
        # train_reference_graphs = loader_to_nx(dataloaders.datasets_steve["train"])
    elif "true" in path: # graphs made bigger, so need to generate new bigger samples
        if "planar" in name:
            G = PlanarDataset(n_nodes=64 + 200, n_graphs=80, k=2)
            test_reference_graphs = [nx.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
            train_reference_graphs = [nx.from_numpy_matrix(G[i]["adj"].numpy()) for i in range(len(G))]
        elif "sbm" in name:
            G = SBMDataset(5, k=4, max_comm_size=100)
            exit(1)
    return test_reference_graphs, train_reference_graphs

def main(args):
    generated = np.load(f"{args.path}/generated_adjs.npz")
    print(f"Found {len(generated)} generated graphs")
    print("Shape first generated graph: ", generated["arr_0"].shape)

    # convert to NetworkX
    generated = [nx.from_numpy_array(generated[f"arr_{id}"]) for id in range(len(generated))][:1024]
    reference, train = get_reference_graphs(args.dataset_name, args.path) #[:len(generated)]
    print(f"Found {len(reference)} reference graphs")
    print("Shape first reference graph: ", len(reference[0].nodes()))
    print()


    print("Avg reference: ", np.mean([len(g.nodes()) for g in train]))
    print("Avg generated: ", np.mean([len(g.nodes()) for g in generated]))
    print(reference[0])

    [g.remove_nodes_from(list(nx.isolates(g))) for g in reference] # remove isolated nodes (in GRID we have all the ones  relative to padding)
    [g.remove_nodes_from(list(nx.isolates(g))) for g in train]

    [g.remove_nodes_from(list(nx.isolates(g))) for g in generated]
    generated = [g for g in generated if len(g.nodes()) > 0]

    print("Avg reference no pad: ", np.mean([len(g.nodes()) for g in train]))
    print("Avg generated no pad: ", np.mean([len(g.nodes()) for g in generated]))

    # evaluate metrics
    s1 = get_orbit(reference)
    s2 = get_orbit(generated)
    orbit = compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False, is_hist=False, sigma=30.0)
    print("Orbit stat: ", orbit)

    s1 = get_clustering(reference)
    s2 = get_clustering(generated)
    clust = compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False, sigma=1.0 / 10)
    print("Clustering stat: ", clust)

    s1 = get_degs(reference)
    s2 = get_degs(generated)
    degs = compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False)
    print("Degree stat: ", degs)


    spectre = spectral_stats(reference, generated, is_parallel=False, n_eigvals=-1,
                            compute_emd=False)
    print("Spectre stat: ", spectre)

    # motif = motif_stats(reference, generated, motif_type='4cycle', ground_truth_match=None, 
    #                     bins=100, compute_emd=False)
    # print("Motif stat: ", motif)

    # frac_unique, frac_unique_non_isomorphic, fraction_unique_non_isomorphic_valid = eval_fraction_unique_non_isomorphic_valid(
    #         generated, reference, is_sbm_graph) # TODO: maybe change test o train!
    frac_non_isomorphic = 1.0 - eval_fraction_isomorphic(generated, train)
    print("Fraction non-iso graphs: ", frac_non_isomorphic)
    print("Fraction of iso generated: ", eval_fraction_isomorphic(generated, generated))

    # acc_ref, acc_gen = 0, 0
    # for i in range(max(len(generated), len(reference))):
    #     if args.dataset_name == "planar":
    #         if i < len(generated) and nx.is_planar(generated[i]):
    #             acc_gen += 1
    #         if i < len(reference) and nx.is_planar(reference[i]):
    #             acc_ref += 1
    #     elif args.dataset_name == "sbm":
    #         if i < len(generated) and is_sbm_graph(generated[i]):
    #             acc_gen += 1
    #         if i < len(reference) and is_sbm_graph(reference[i]):
    #             acc_ref += 1
    # print("Validity gen: ", round(acc_gen / len(generated), 3))
    # print("Validity ref: ", round(acc_ref / len(reference), 3))

    # plot sample graphs
    for i in range(10):
        nx.draw(generated[i], node_size=30)
        plt.savefig(f"plots/digress/{args.dataset_name}/{i}.png")
        plt.close()



def main_bigger(args):
    for size in ["1.5", "2.0", "4.0", "8.0"]:
        path = f"../graphs/{args.dataset_name}_{size}_fullprec/generated_adjs.npz"
        generated = np.load(path)
        generated = [nx.from_numpy_array(generated[f"arr_{id}"]) for id in range(len(generated))][:256]

        print(path.upper())
        print(f"Found {len(generated)} generated graphs")
        
        test = np.load(f"../../AHK/dataset/{args.dataset_name}/{args.dataset_name}_large_{size}.npy", allow_pickle=True)
        test = [nx.from_numpy_array(test[id]) for id in range(len(test))]
        print(f"Found {len(test)} test graphs")

        [g.remove_nodes_from(list(nx.isolates(g))) for g in test] # remove isolated nodes
        [g.remove_nodes_from(list(nx.isolates(g))) for g in generated]
        generated = [g for g in generated if len(g.nodes()) > 0]

        print("Avg num_nodes gen/test: ", np.mean([len(g.nodes()) for g in test]), " / ", np.mean([len(g.nodes()) for g in generated]))

        # print("SELFLOOPS: ", [len(list(nx.selfloop_edges(g))) for g in test])
        # print("SELFLOOPS: ", [len(list(nx.selfloop_edges(g))) for g in generated])

        s1 = get_degs(test)
        s2 = get_degs(generated)
        degs = round(compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False), 4)

        s1 = get_orbit(test)
        s2 = get_orbit(generated)
        orbit = round(compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False, is_hist=False, sigma=30.0), 4)

        s1 = get_clustering(test)
        s2 = get_clustering(generated)
        clust = round(compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False, sigma=1.0 / 10), 4)

        spectre = round(spectral_stats(test, generated, is_parallel=False, n_eigvals=-1, compute_emd=False), 4)

        acc_ref, acc_gen = 0, 0
        for i in range(max(len(generated), len(test))):
            if args.dataset_name == "planar":
                if i < len(generated) and nx.is_planar(generated[i]):
                    acc_gen += 1
                if i < len(test) and nx.is_planar(test[i]):
                    acc_ref += 1
            elif args.dataset_name == "sbm":
                if i < len(generated) and is_sbm_graph(generated[i]):
                    acc_gen += 1
                if i < len(test) and is_sbm_graph(test[i]):
                    acc_ref += 1
        acc_gen = round(acc_gen / len(generated), 3)
        acc_ref = round(acc_ref / len(test), 3)

        print(f"Deg:\t{degs}, Clus:\t{clust}, Orbit:\t{orbit}, Spec.:\t{spectre}, Val.:\t{acc_gen}, Val ref.:\t{acc_ref}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)  
    parser.add_argument('--path', type=str, required=False, help="path to generated graphs")
    parser.add_argument('--bigger', action="store_true", required=False, default=None, help="eval bigger splits")
    args = parser.parse_args()

    if args.bigger is None:
        main(args)
    else:
        main_bigger(args)





# EVALUATE WITH DIGRESS
# config = dotdict({
#         "dataset": dotdict({
#             "name": "planar",
#             "datadir": get_dataset_dir("planar"),
#             'remove_h': None,
#         }),
#         "general": dotdict({
#             'name': 'planar_metric_debug',
#             'wandb': 'offline',
#             'gpus': 1,
#             'resume': None,
#             'test_only': '/home/steve.azzolin/DiGress_fork/checkpoints/checkpoint_planar.ckpt',
#             'sample_bigger_graphs': 0,
#             'check_val_every_n_epochs': 5,
#             'sample_every_val': 4,
#             'val_check_interval': None,
#             'samples_to_generate': 512,
#             'samples_to_save': 20,
#             'chains_to_save': 1,
#             'log_every_steps': 50,
#             'number_chain_steps': 50,
#             'final_model_samples_to_generate': 10000,
#             'final_model_samples_to_save': 30,
#             'final_model_chains_to_save': 20,
#             'evaluate_all_checkpoints': False
#         }),
#         "train": dotdict({
#             "num_workers": 0,
#             'n_epochs': 1000,
#             'batch_size': 32,
#             'lr': 0.0002,
#             'clip_grad': None,
#             'save_model': True,
#             'num_workers': 0,
#             'ema_decay': 0,
#             'progress_bar': False,
#             'weight_decay': 1e-12,
#             'optimizer': 'adamw',
#             'seed': 0
#         }),
#         'model': dotdict({
#             'type': 'discrete',
#             'transition': 'marginal',
#             'model': 'graph_tf',
#             'diffusion_steps': 500,
#             'diffusion_noise_schedule': 'cosine',
#             'n_layers': 5,
#             'extra_features': 'all',
#             'hidden_mlp_dims': {'X': 256, 'E': 128, 'y': 128},
#             'hidden_dims': {'dx': 256, 'de': 64, 'dy': 64, 'n_head': 8, 'dim_ffX': 256, 'dim_ffE': 128, 'dim_ffy': 128},
#             'lambda_train': [5, 0]
#         }),
#     })
# dataloaders = SpectreGraphDataModule(config)
# digress_metric = PlanarSamplingMetrics(dataloaders)
# digress_metric(generated, save=False, name=None, current_epoch=None, val_counter=None, local_rank=0, path="/cancella//")
