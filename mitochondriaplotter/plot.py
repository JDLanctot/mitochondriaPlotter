import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import random
from mitochondriaplotter.util import node_tuples_index_to_int, set_mpl, coalesced_graph
from typing import List, Tuple
import pandas as pd
import seaborn as sns
from os.path import join

__all__ = []
__all__.extend([
    'plot_network_from_edgelist',
    'plot_lattice_from_edgelist',
    'gen_lattice',
    'plot_probability_distribution',
])

def plot_network_from_edgelist(edgelist: List[Tuple[int,int]]) -> plt.Figure:
    # Create the graph from the edgelist
    # G = nx.from_edgelist(edgelist)
    G = coalesced_graph(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}

    # Position components in a grid layout
    grid_size = int(np.ceil(np.sqrt(len(connected_components))))
    max_component_size = len(connected_components[0])

    # Define a base scale for the largest component
    base_scale = 1.0

    for i, component in enumerate(connected_components):
        component_subgraph = G.subgraph(component)

        # Calculate center position for this component
        row = i // grid_size
        col = i % grid_size
        center = (col * 3, -row * 3)  # Adjust the multiplier (3) to change spacing between components

        # Scale factor based on component size
        scale_factor = base_scale * (len(component) / max_component_size) ** 0.5

        # Use kamada_kawai_layout for more uniform edge lengths within each component
        component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)

        # Shift the component to its position in the grid
        component_pos = {node: (x + center[0], y + center[1]) for node, (x, y) in component_pos.items()}

        pos.update(component_pos)

    # Plot the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50, width=1, edge_color='gray', ax=ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # plt.colorbar(sm, label='Node Degree')

    # Remove axis
    ax.set_axis_off()

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

def plot_probability_distribution(data: Tuple[np.ndarray, np.ndarray], bins: int = 10) -> plt.Figure:
    """
    Plots a probability distribution given a tuple of two numpy arrays.

    Parameters:
    - data: tuple (x, y)
        x: numpy array of x values (data points)
        y: numpy array of y values (counts)
    - bins: int
        Number of bins to use for the histogram

    Returns:
    - fig: matplotlib.figure.Figure
        The figure object containing the plot.
    """
    set_mpl()

    x, y = data

    # Compute the histogram of the data
    hist, bin_edges = np.histogram(x, bins=bins, weights=y)

    # Normalize the histogram to get the PDF
    bin_widths = np.diff(bin_edges)
    hist_normalized = hist / (np.sum(hist) * bin_widths)

    # Compute the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the data
    ax.scatter(bin_centers, hist_normalized, label='Probability Distribution')

    # Add labels and title
    ax.set_xlabel('X values')
    ax.set_ylabel('Probability')
    ax.set_title('Probability Distribution')

    # Tight layout for better spacing
    plt.tight_layout()

    return fig

def plot_results(df: pd.DataFrame, save_path: str, file_name: str, a1: float, N_mito: int) -> None:
    # Calculate x-axis values
    df['x_axis'] = a1 * N_mito / df['b1']

    # Set the style for all plots
    sns.set_style("whitegrid")

    # 1. Plot number of components
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='x_axis', y='number_of_components')
    plt.xscale('log')
    plt.title(f'Number of Components vs a1*N_mito/b1 (a1={a1}, N_mito={N_mito})')
    plt.xlabel('a1 * N_mito / b1 (log scale)')
    plt.ylabel('Number of Components')
    plt.savefig(join(save_path, f"{file_name}_number_of_components.png"))
    plt.close()

    # 2. Plot cycle categories
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='x_axis', y='no_cycles', label='No Cycles')
    sns.lineplot(data=df, x='x_axis', y='one_cycle', label='One Cycle')
    sns.lineplot(data=df, x='x_axis', y='many_cycles', label='Many Cycles')
    plt.xscale('log')
    plt.title(f'Cycle Categories vs a1*N_mito/b1 (a1={a1}, N_mito={N_mito})')
    plt.xlabel('a1 * N_mito / b1 (log scale)')
    plt.ylabel('Fraction of Nodes')
    plt.legend()
    plt.savefig(join(save_path, f"{file_name}_cycle_categories.png"))
    plt.close()

    # 3. Plot degree distribution
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='x_axis', y='degree_1', label='Degree 1')
    sns.lineplot(data=df, x='x_axis', y='degree_2', label='Degree 2')
    sns.lineplot(data=df, x='x_axis', y='degree_3', label='Degree 3')
    plt.xscale('log')
    plt.title(f'Degree Distribution vs a1*N_mito/b1 (a1={a1}, N_mito={N_mito})')
    plt.xlabel('a1 * N_mito / b1 (log scale)')
    plt.ylabel('Fraction of Nodes')
    plt.legend()
    plt.savefig(join(save_path, f"{file_name}_degree_distribution.png"))
    plt.close()
