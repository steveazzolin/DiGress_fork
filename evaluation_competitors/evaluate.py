import networkx as nx
import numpy as np
import argparse
import os
import pathlib
from torch_geometric.utils import to_networkx
import graph_tool.all as gt

# evaluation setup code from GraphRNN
from mmd import compute_mmd, gaussian_emd,gaussian
from compute_metrics import get_orbit, get_clustering, get_degs


from src.datasets.spectre_dataset import SpectreGraphDataset, SpectreGraphDataModule
from src.analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics


def loader_to_nx(loader):
    networkx_graphs = []
    for i, batch in enumerate(loader):
        data_list = batch.to_data_list()
        for j, data in enumerate(data_list):
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

def get_reference_graphs(name):
    config = {
        "dataset":{
            "name": name,
            "dir": get_dataset_dir(name)
        }
    } #batch_size, num_workers
    dataloaders = SpectreGraphDataModule(config)
    reference_graphs = loader_to_nx(dataloaders['test'])
    return reference_graphs

def main(args):
    generated = np.load("outputs/generated_adjs.npz")
    print(f"Found {len(generated)} generated graphs")
    print("Shape first generated graph: ", generated["arr_0"].shape)

    # convert to NetworkX
    generated = [nx.from_numpy_array(generated[f"arr_{id}"]) for id in len(generated)]
    reference = get_reference_graphs(args.dataset_name)
    print(len(reference))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True)  
    parser.add_argument('--path', type=str, required=True, help="path to generated graphs")  
    args = parser.parse_args()
    main(args)