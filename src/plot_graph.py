import json
import argparse
import os
import random
import sys
import logging
from tqdm import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot(args):
    id = int(args.id)
    G = np.load("outputs/generated_adjs.npz")

    assert id < len(G)

    g = nx.from_numpy_array(G[f"arr_{id}"])
    pos = nx.spring_layout(g, iterations=100, seed=0)
    nx.draw(g, pos=pos)
    plt.savefig(f"outputs/plots/{id}.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)  
    args = parser.parse_args()

    plot(args)

if __name__ == '__main__':
    main()
