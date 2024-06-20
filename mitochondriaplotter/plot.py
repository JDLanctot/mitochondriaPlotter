import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Tuple

__all__ = []
__all__.extend([
    'plot_network_from_edgelist',
    'plot_lattice_from_edgelist',
])

def plot_network_from_edgelist(edgelist: List[List[int]]) -> plt.Figure:
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

def plot_lattice_from_edgelist(edgelist: List[List[int]]) -> plt.Figure:
    # Create the graph from the edgelist
    G = nx.from_edgelist(edgelist)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the grid positions for the nodes
    pos = {}
    nodes = list(G.nodes())
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

