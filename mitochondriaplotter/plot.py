import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import random
from mitochondriaplotter.util import node_tuples_index_to_int
from typing import List, Tuple

__all__ = []
__all__.extend([
    'plot_network_from_edgelist',
    'plot_lattice_from_edgelist',
    'gen_lattice',
])

def plot_network_from_edgelist(edgelist: List[Tuple[int,int]]) -> plt.Figure:
    # Scaling weight of spring interactions
    beta = 400

    # Create the graph from the edgelist
    G = nx.from_edgelist(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
    largest_component = connected_components[0]

    # Create a subgraph for the largest connected component
    largest_subgraph = G.subgraph(largest_component)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}
    pos.update(nx.spring_layout(largest_subgraph, center=(0, 0), weight=beta/len(connected_components[0])))  # Center largest component

    # Position other components in a circular layout around the largest component
    radius = 2  # Radius for positioning other components
    angle_step = 2 * 3.14159 / (len(connected_components) - 1)
    angle = 0

    for component in connected_components[1:]:
        component_subgraph = G.subgraph(component)
        component_pos = nx.spring_layout(component_subgraph, center=(radius * np.cos(angle), radius * np.sin(angle)), weight=beta/len(component))
        pos.update(component_pos)
        angle += angle_step

    # Plot the graph
    fig = plt.figure()
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=500, width=2, edge_color='gray')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, label='Node Degree')

    return fig

def plot_lattice_from_edgelist(edgelist: List[Tuple[int,int]], rows: int, cols: int) -> plt.Figure:
    # Create the graph
    G = nx.Graph()

    # Add all nodes explicitly based on rows and cols
    G.add_nodes_from((i * cols + j + 1 for i in range(rows) for j in range(cols)))

    # Add edges from the edge list
    G.add_edges_from(edgelist)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the grid positions for the nodes
    pos = {}
    nodes = sorted(list(G.nodes()))
    grid_size = int(np.ceil(np.sqrt(len(nodes))))

    for idx, node in enumerate(nodes):
        row = idx // grid_size
        col = idx % grid_size
        pos[node] = (col, -row)  # Using negative row for top-down layout

    # Plot the graph
    fig = plt.figure(figsize=(10, 10))
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=500, edge_color='gray')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, label='Node Degree')

    return fig

def gen_lattice(rows: int, cols: int, p: float, k: int = 4) -> List[Tuple[int,int]]:
    nodes = [(i, j) for i in range(rows) for j in range(cols)]

    # Function to get neighbors for different k values
    def get_neighbors(i, j):
        if k == 4:
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]  # up, down, left, right
        elif k == 3:
            neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        else:
            raise ValueError("Unsupported k value. Use k=4 for conventional lattice or k=3 for triangular lattice.")

        # Filter out neighbors that are out of bounds
        neighbors = [(x, y) for x, y in neighbors if 0 <= x < rows and 0 <= y < cols]
        return neighbors

    edge_list = []
    for i in range(rows):
        for j in range(cols):
            current_node = (i, j)
            neighbors = get_neighbors(i, j)
            for neighbor in neighbors:
                if random.random() < p:
                    edge_list.append((current_node, neighbor))

    el1, el2 = zip(*edge_list)
    return node_tuples_index_to_int([list(el1), list(el2)], cols)


