import os
import random

import numpy as np
import matplotlib as mpl
from typing import List

__all__ = ['set_seed', 'set_mpl', 'double_ended_to_edgelist']

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_mpl():
    # change defaults to be less ugly for matplotlib
    mpl.rc('xtick', labelsize=14, color="#222222")
    mpl.rc('ytick', labelsize=14, color="#222222")
    mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    mpl.rc('font', size=16)
    mpl.rc('xtick.major', size=6, width=1)
    mpl.rc('xtick.minor', size=3, width=1)
    mpl.rc('ytick.major', size=6, width=1)
    mpl.rc('ytick.minor', size=3, width=1)
    mpl.rc('axes', linewidth=1, edgecolor="#222222", labelcolor="#222222")
    mpl.rc('text', usetex=False, color="#222222")

def double_ended_to_edgelist(mat: np.ndarray) -> np.ndarray:
    # Ensure the matrix is in the expected shape (2, N)
    # assert mat.shape[0] == 2, "Matrix should have 2 rows"

    # Convert the matrix rows to lists of tuples
    outbound_nodes = [tuple(row) for row in mat[0]]
    inbound_nodes = [tuple(row) for row in mat[1]]

    # Combine both lists to create a list of all nodes
    all_nodes = outbound_nodes + inbound_nodes

    # Extract unique values from the first entry of the tuples
    unique_values = set(node[0] for node in all_nodes)

    # Append connections (i,1) to (i,2) to the original lists
    for i in unique_values:
        outbound_nodes.append((i, 1))
        inbound_nodes.append((i, 2))
    all_nodes = inbound_nodes + outbound_nodes

    # Convert back to numpy array if needed
    extended_mat = [outbound_nodes, inbound_nodes]

    # convert unique tuples into unique integers in the edge_list
    unique_nodes = set(node for node in all_nodes)
    unique_tuples = {node: i for i, node in enumerate(unique_nodes)}
    indexed_edgelist = [[unique_tuples[node] for node in extended_mat[0]],
                        [unique_tuples[node] for node in extended_mat[1]]]
    indexed_edgelist = list(zip(*indexed_edgelist))

    return indexed_edgelist

