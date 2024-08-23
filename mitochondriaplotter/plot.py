import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import LogLocator, LogFormatterSciNotation, AutoMinorLocator
from matplotlib.font_manager import FontProperties
import numpy as np
import random
from mitochondriaplotter.util import node_tuples_index_to_int, set_mpl, coalesced_graph, optimize_circular_order
from typing import List, Tuple
import pandas as pd
import seaborn as sns
from os.path import join

__all__ = []
__all__.extend([
    'custom_circular_layout',
    'plot_network_from_edgelist',
    'plot_particular_network_from_edgelist',
    'plot_particular_network_from_edgelist_1',
    'plot_particular_network_from_edgelist_2',
    'plot_particular_network_from_edgelist_3',
    'plot_particular_network_from_edgelist_4',
    'plot_particular_network_from_edgelist_5',
    'plot_particular_network_from_edgelist_6',
    'plot_lattice_from_edgelist',
    'gen_lattice',
    'plot_probability_distribution',
])


def custom_circular_layout(G, scale=1, center=None, dim=2, ordering=None):
    if ordering is None:
        ordering = list(G.nodes())

    t = np.linspace(0, 2*np.pi, len(ordering), endpoint=False)
    pos = {node: np.array([np.cos(angle), np.sin(angle)]) for node, angle in zip(ordering, t)}

    # Manual scaling
    if scale != 1:
        pos = {node: scale * coord for node, coord in pos.items()}

    # Manual centering
    if center is not None:
        center = np.asarray(center)
        pos = {node: coord + center for node, coord in pos.items()}

    return pos

# def plot_network_from_edgelist(edgelist: List[Tuple[int,int]], model: str = 'aspatial') -> plt.Figure:
#     # Create the graph from the edgelist
#     if model == 'aspatial':
#         G = coalesced_graph(edgelist)
#     else:
#         G = nx.from_edgelist(edgelist)
#
#     # Find all connected components
#     connected_components = sorted(nx.connected_components(G), key=len, reverse=True)
#
#     # Determine node degrees and apply the viridis colormap
#     degrees = dict(G.degree())
#     cmap = cm.get_cmap('viridis')
#     norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
#     node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]
#
#     # Set up the positions for the nodes
#     pos = {}
#
#     # Position components in a grid layout
#     grid_size = int(np.ceil(np.sqrt(len(connected_components))))
#     max_component_size = len(connected_components[0])
#
#     # Define a base scale for the largest component
#     base_scale = 1.0
#
#     for i, component in enumerate(connected_components):
#         component_subgraph = G.subgraph(component)
#
#         # Calculate center position for this component
#         row = i // grid_size
#         col = i % grid_size
#         center = (col * 3, -row * 3)  # Adjust the multiplier (3) to change spacing between components
#
#         # Scale factor based on component size
#         scale_factor = base_scale * (len(component) / max_component_size) ** 0.5
#
#         # Use kamada_kawai_layout for more uniform edge lengths within each component
#         component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)
#
#         # Shift the component to its position in the grid
#         component_pos = {node: (x + center[0], y + center[1]) for node, (x, y) in component_pos.items()}
#
#         pos.update(component_pos)
#
#     # Plot the graph
#     fig, ax = plt.subplots(figsize=(12, 12))
#     nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50, width=1, edge_color='gray', ax=ax)
#     sm = ScalarMappable(cmap=cmap, norm=norm)
#     # sm.set_array([])
#     # plt.colorbar(sm, label='Node Degree')
#
#     # Remove axis
#     ax.set_axis_off()
#
#     return fig

def plot_network_from_edgelist(edgelist: List[Tuple[int,int]], model: str = 'aspatial') -> plt.Figure:
    # Create the graph from the edgelist
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        G = nx.from_edgelist(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}

    # Calculate component sizes and total size
    component_sizes = [len(component) for component in connected_components]
    total_size = sum(component_sizes)

    # Position components in a grid layout
    grid_size = int(np.ceil(np.sqrt(len(connected_components))))
    max_component_size = len(connected_components[0])

    # Define a base scale for the largest component
    base_scale = 1.5  # Increased to make components larger

    # Calculate total width and height
    total_width = total_height = grid_size * 3  # Increased spacing between components

    for i, component in enumerate(connected_components):
        component_subgraph = G.subgraph(component)

        # Calculate center position for this component
        row = i // grid_size
        col = i % grid_size

        # Calculate cell size based on component size
        cell_size = np.sqrt(component_sizes[i] / total_size) * grid_size

        # Calculate center, adjusting for margins
        center_x = (col + 0.5) * (total_width / grid_size)
        center_y = -(row + 0.5) * (total_height / grid_size)
        center = (center_x, center_y)

        # Scale factor based on component size
        scale_factor = base_scale * (len(component) / max_component_size) ** 0.5 * cell_size

        # Use kamada_kawai_layout for more uniform edge lengths within each component
        component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)

        # Shift and scale the component to its position in the grid
        component_pos = {node: ((x * 0.8 + center[0]), (y * 0.8 + center[1]))
                         for node, (x, y) in component_pos.items()}

        pos.update(component_pos)

    # Plot the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color=node_colors, with_labels=False, node_size=50, width=1, edge_color='gray', ax=ax)

    # Remove axis
    ax.set_axis_off()

    # Set limits to show the entire graph
    ax.set_xlim(-0.1 * total_width, 1.1 * total_width)
    ax.set_ylim(-1.1 * total_height, 0.1 * total_height)

    return fig

def plot_particular_network_from_edgelist(
    edgelist: List[Tuple[int, int]],
    model: str = 'aspatial',
    # scale_factors: List[float] = None,
    # x_translations: List[float] = None,
    # y_translations: List[float] = None
    # rotations: List[float] = None
) -> plt.Figure:
    scale_factors = [0.9, 1.1, 0.8, 0.9, 0.65, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    x_translations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y_translations = [0.5, 0.5, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rotations = [50.0, 0.0, 0.0, 0.0, -20.0, 80.0, 0.0, 0.0, 0.0, -30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Create the graph from the edgelist
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        G = nx.from_edgelist(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}

    # Calculate component sizes and total size
    component_sizes = [len(component) for component in connected_components]
    total_size = sum(component_sizes)

    # Position components in a grid layout
    grid_size = int(np.ceil(np.sqrt(len(connected_components))))
    max_component_size = len(connected_components[0])

    # Define a base scale for the largest component
    base_scale = 1.8  # Increased to make components larger

    # Calculate total width and height
    total_width = total_height = grid_size * 3  # Increased spacing between components

    for i, component in enumerate(connected_components):
        component_subgraph = G.subgraph(component)

        # Calculate center position for this component
        row = i // grid_size
        col = i % grid_size

        # Calculate cell size based on component size
        cell_size = np.sqrt(component_sizes[i] / total_size) * grid_size

        # Calculate center, adjusting for margins
        center_x = (col + 0.5) * (total_width / grid_size)
        center_y = -(row + 0.5) * (total_height / grid_size)
        center = (center_x, center_y)

        # Scale factor based on component size
        scale_factor = base_scale * (len(component) / max_component_size) ** 0.5 * cell_size

        component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)
        # Shift and scale the component to its position in the grid
        component_pos = {node: ((x * 0.8 + center[0]), (y * 0.8 + center[1]))
                         for node, (x, y) in component_pos.items()}

        pos.update(component_pos)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Apply additional scaling, translation, and rotation to each component
    for i, component in enumerate(connected_components):
        scale = scale_factors[i] if scale_factors and i < len(scale_factors) else 1
        x_trans = x_translations[i] if x_translations and i < len(x_translations) else 0
        y_trans = y_translations[i] if y_translations and i < len(y_translations) else 0
        rotation = rotations[i] if rotations and i < len(rotations) else 0

        # Calculate component center
        component_xs, component_ys = zip(*[pos[node] for node in component])
        component_center_x = sum(component_xs) / len(component)
        component_center_y = sum(component_ys) / len(component)

        for node in component:
            x, y = pos[node]

            # Translate to origin
            x -= component_center_x
            y -= component_center_y

            # Scale
            x *= scale
            y *= scale

            # Rotate
            angle = np.radians(rotation)
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)

            # Translate back and apply additional translation
            pos[node] = (x_rot + component_center_x + x_trans,
                         y_rot + component_center_y + y_trans)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Plot the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color='black', with_labels=False, node_size=200, width=5, edge_color='gray', ax=ax)

    # Remove axis
    ax.set_axis_off()

    # Set limits to show the entire graph
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    return fig


# FOR a1 = 0.2, a2 = 0.01, b1 = 0.1, run 8
def plot_particular_network_from_edgelist_1(
    edgelist: List[Tuple[int, int]],
    model: str = 'aspatial',
    # scale_factors: List[float] = None,
    # x_translations: List[float] = None,
    # y_translations: List[float] = None
    # rotations: List[float] = None
) -> plt.Figure:
    scale_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    x_translations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y_translations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rotations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Create the graph from the edgelist
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        G = nx.from_edgelist(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}

    # Calculate component sizes and total size
    component_sizes = [len(component) for component in connected_components]
    total_size = sum(component_sizes)

    # Position components in a grid layout
    grid_size = int(np.ceil(np.sqrt(len(connected_components))))
    max_component_size = len(connected_components[0])

    # Define a base scale for the largest component
    base_scale = 1.8  # Increased to make components larger

    # Calculate total width and height
    total_width = total_height = grid_size * 3  # Increased spacing between components

    for i, component in enumerate(connected_components):
        component_subgraph = G.subgraph(component)

        # Calculate center position for this component
        row = i // grid_size
        col = i % grid_size

        # Calculate cell size based on component size
        cell_size = np.sqrt(component_sizes[i] / total_size) * grid_size

        # Calculate center, adjusting for margins
        center_x = (col + 0.5) * (total_width / grid_size)
        center_y = -(row + 0.5) * (total_height / grid_size)
        center = (center_x, center_y)

        # Scale factor based on component size
        scale_factor = base_scale * (len(component) / max_component_size) ** 0.5 * cell_size

        # Use kamada_kawai_layout for more uniform edge lengths within each component
        component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)

        # Shift and scale the component to its position in the grid
        component_pos = {node: ((x * 0.8 + center[0]), (y * 0.8 + center[1]))
                         for node, (x, y) in component_pos.items()}

        pos.update(component_pos)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Apply additional scaling, translation, and rotation to each component
    for i, component in enumerate(connected_components):
        scale = scale_factors[i] if scale_factors and i < len(scale_factors) else 1
        x_trans = x_translations[i] if x_translations and i < len(x_translations) else 0
        y_trans = y_translations[i] if y_translations and i < len(y_translations) else 0
        rotation = rotations[i] if rotations and i < len(rotations) else 0

        # Calculate component center
        component_xs, component_ys = zip(*[pos[node] for node in component])
        component_center_x = sum(component_xs) / len(component)
        component_center_y = sum(component_ys) / len(component)

        for node in component:
            x, y = pos[node]

            # Translate to origin
            x -= component_center_x
            y -= component_center_y

            # Scale
            x *= scale
            y *= scale

            # Rotate
            angle = np.radians(rotation)
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)

            # Translate back and apply additional translation
            pos[node] = (x_rot + component_center_x + x_trans,
                         y_rot + component_center_y + y_trans)

    # Plot the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color='black', with_labels=False, node_size=200, width=5, edge_color='gray', ax=ax)

    # Remove axis
    ax.set_axis_off()

    # Set limits to show the entire graph
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    return fig


# FOR a1 = 0.2, a2 = 0.01, b1 = 1.0, run 1
def plot_particular_network_from_edgelist_2(
    edgelist: List[Tuple[int, int]],
    model: str = 'aspatial',
    # scale_factors: List[float] = None,
    # x_translations: List[float] = None,
    # y_translations: List[float] = None
    # rotations: List[float] = None
) -> plt.Figure:
    scale_factors = [1.3, 1.8, 1.5, 0.8, 1.0, 1.0]
    x_translations = [1.4, 0.6, 0.0, 0.0, 0.2, 0.0]
    y_translations = [0.0, 0.0, 0.0, -0.3, 0.0, 0.0]
    rotations = [-60.0, -30.0, 0.0, 0.0, 0.0, 0.0]

    # Create the graph from the edgelist
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        G = nx.from_edgelist(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}

    # Calculate component sizes and total size
    component_sizes = [len(component) for component in connected_components]
    total_size = sum(component_sizes)

    # Position components in a grid layout
    grid_size = int(np.ceil(np.sqrt(len(connected_components))))
    max_component_size = len(connected_components[0])

    # Define a base scale for the largest component
    base_scale = 1.8  # Increased to make components larger

    # Calculate total width and height
    total_width = total_height = grid_size * 3  # Increased spacing between components

    for i, component in enumerate(connected_components):
        component_subgraph = G.subgraph(component)

        # Calculate center position for this component
        row = i // grid_size
        col = i % grid_size

        # Calculate cell size based on component size
        cell_size = np.sqrt(component_sizes[i] / total_size) * grid_size

        # Calculate center, adjusting for margins
        center_x = (col + 0.5) * (total_width / grid_size)
        center_y = -(row + 0.5) * (total_height / grid_size)
        center = (center_x, center_y)

        # Scale factor based on component size
        scale_factor = base_scale * (len(component) / max_component_size) ** 0.5 * cell_size

        # Use kamada_kawai_layout for more uniform edge lengths within each component
        component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)

        # Shift and scale the component to its position in the grid
        component_pos = {node: ((x * 0.8 + center[0]), (y * 0.8 + center[1]))
                         for node, (x, y) in component_pos.items()}

        pos.update(component_pos)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y / 2

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Apply additional scaling, translation, and rotation to each component
    for i, component in enumerate(connected_components):
        scale = scale_factors[i] if scale_factors and i < len(scale_factors) else 1
        x_trans = x_translations[i] if x_translations and i < len(x_translations) else 0
        y_trans = y_translations[i] if y_translations and i < len(y_translations) else 0
        rotation = rotations[i] if rotations and i < len(rotations) else 0

        # Calculate component center
        component_xs, component_ys = zip(*[pos[node] for node in component])
        component_center_x = sum(component_xs) / len(component)
        component_center_y = sum(component_ys) / len(component)

        for node in component:
            x, y = pos[node]

            # Translate to origin
            x -= component_center_x
            y -= component_center_y

            # Scale
            x *= scale
            y *= scale

            # Rotate
            angle = np.radians(rotation)
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)

            # Translate back and apply additional translation
            pos[node] = (x_rot + component_center_x + x_trans,
                         y_rot + component_center_y + y_trans)

    # Plot the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color='black', with_labels=False, node_size=200, width=5, edge_color='gray', ax=ax)

    # Remove axis
    ax.set_axis_off()

    # Set limits to show the entire graph
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    return fig


# FOR a1 = 0.2, a2 = 0.0, b1 = 0.1, run 6
def plot_particular_network_from_edgelist_3(
    edgelist: List[Tuple[int, int]],
    model: str = 'aspatial',
    # scale_factors: List[float] = None,
    # x_translations: List[float] = None,
    # y_translations: List[float] = None
    # rotations: List[float] = None
) -> plt.Figure:
    scale_factors = [1.0, 1.0, 0.9, 0.9, 0.35, 0.5, 0.7]
    x_translations = [0.0, 0.0, -0.6, 0.0, 0.0, -2.0, 2.0]
    y_translations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0]
    rotations = [20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Create the graph from the edgelist
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        G = nx.from_edgelist(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}

    # Calculate component sizes and total size
    component_sizes = [len(component) for component in connected_components]
    total_size = sum(component_sizes)

    # Position components in a grid layout
    grid_size = int(np.ceil(np.sqrt(len(connected_components))))
    max_component_size = len(connected_components[0])

    # Define a base scale for the largest component
    base_scale = 1.8  # Increased to make components larger

    # Calculate total width and height
    total_width = total_height = grid_size * 3  # Increased spacing between components

    for i, component in enumerate(connected_components):
        component_subgraph = G.subgraph(component)

        # Calculate center position for this component
        row = i // grid_size
        col = i % grid_size

        # Calculate cell size based on component size
        cell_size = np.sqrt(component_sizes[i] / total_size) * grid_size

        # Calculate center, adjusting for margins
        center_x = (col + 0.5) * (total_width / grid_size)
        center_y = -(row + 0.5) * (total_height / grid_size)
        center = (center_x, center_y)

        # Scale factor based on component size
        scale_factor = base_scale * (len(component) / max_component_size) ** 0.5 * cell_size

        # Use optimized circular layout for the component at row 1, col 1
        if row == 1 and col == 1:
            optimized_order = optimize_circular_order(component_subgraph)
            component_pos = custom_circular_layout(component_subgraph, scale=scale_factor, ordering=optimized_order)
        else:
            component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)
            # Shift and scale the component to its position in the grid
            component_pos = {node: ((x * 0.8 + center[0]), (y * 0.8 + center[1]))
                             for node, (x, y) in component_pos.items()}

        pos.update(component_pos)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Apply additional scaling, translation, and rotation to each component
    for i, component in enumerate(connected_components):
        scale = scale_factors[i] if scale_factors and i < len(scale_factors) else 1
        x_trans = x_translations[i] if x_translations and i < len(x_translations) else 0
        y_trans = y_translations[i] if y_translations and i < len(y_translations) else 0
        rotation = rotations[i] if rotations and i < len(rotations) else 0

        # Calculate component center
        component_xs, component_ys = zip(*[pos[node] for node in component])
        component_center_x = sum(component_xs) / len(component)
        component_center_y = sum(component_ys) / len(component)

        for node in component:
            x, y = pos[node]

            # Translate to origin
            x -= component_center_x
            y -= component_center_y

            # Scale
            x *= scale
            y *= scale

            # Rotate
            angle = np.radians(rotation)
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)

            # Translate back and apply additional translation
            pos[node] = (x_rot + component_center_x + x_trans,
                         y_rot + component_center_y + y_trans)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Plot the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color='black', with_labels=False, node_size=200, width=5, edge_color='gray', ax=ax)

    # Remove axis
    ax.set_axis_off()

    # Set limits to show the entire graph
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    return fig


# FOR a1 = 0.2, a2 = 0.0, b1 = 1.0, run 2
def plot_particular_network_from_edgelist_4(
    edgelist: List[Tuple[int, int]],
    model: str = 'aspatial',
    # scale_factors: List[float] = None,
    # x_translations: List[float] = None,
    # y_translations: List[float] = None
    # rotations: List[float] = None
) -> plt.Figure:
    scale_factors = [1.15, 0.9, 0.95, 0.8, 0.95, 0.9, 0.85, 0.5]
    x_translations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.9]
    y_translations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4]
    rotations = [-30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Create the graph from the edgelist
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        G = nx.from_edgelist(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}

    # Calculate component sizes and total size
    component_sizes = [len(component) for component in connected_components]
    total_size = sum(component_sizes)

    # Position components in a grid layout
    grid_size = int(np.ceil(np.sqrt(len(connected_components))))
    max_component_size = len(connected_components[0])

    # Define a base scale for the largest component
    base_scale = 1.8  # Increased to make components larger

    # Calculate total width and height
    total_width = total_height = grid_size * 3  # Increased spacing between components

    for i, component in enumerate(connected_components):
        component_subgraph = G.subgraph(component)

        # Calculate center position for this component
        row = i // grid_size
        col = i % grid_size

        # Calculate cell size based on component size
        cell_size = np.sqrt(component_sizes[i] / total_size) * grid_size

        # Calculate center, adjusting for margins
        center_x = (col + 0.5) * (total_width / grid_size)
        center_y = -(row + 0.5) * (total_height / grid_size)
        center = (center_x, center_y)

        # Scale factor based on component size
        scale_factor = base_scale * (len(component) / max_component_size) ** 0.5 * cell_size

        component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)
        # Shift and scale the component to its position in the grid
        component_pos = {node: ((x * 0.8 + center[0]), (y * 0.8 + center[1]))
                         for node, (x, y) in component_pos.items()}

        pos.update(component_pos)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Apply additional scaling, translation, and rotation to each component
    for i, component in enumerate(connected_components):
        scale = scale_factors[i] if scale_factors and i < len(scale_factors) else 1
        x_trans = x_translations[i] if x_translations and i < len(x_translations) else 0
        y_trans = y_translations[i] if y_translations and i < len(y_translations) else 0
        rotation = rotations[i] if rotations and i < len(rotations) else 0

        # Calculate component center
        component_xs, component_ys = zip(*[pos[node] for node in component])
        component_center_x = sum(component_xs) / len(component)
        component_center_y = sum(component_ys) / len(component)

        for node in component:
            x, y = pos[node]

            # Translate to origin
            x -= component_center_x
            y -= component_center_y

            # Scale
            x *= scale
            y *= scale

            # Rotate
            angle = np.radians(rotation)
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)

            # Translate back and apply additional translation
            pos[node] = (x_rot + component_center_x + x_trans,
                         y_rot + component_center_y + y_trans)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Plot the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color='black', with_labels=False, node_size=200, width=5, edge_color='gray', ax=ax)

    # Remove axis
    ax.set_axis_off()

    # Set limits to show the entire graph
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    return fig


# FOR a1 = 0.2, a2 = 0.01, b1 = 10, run 5
def plot_particular_network_from_edgelist_5(
    edgelist: List[Tuple[int, int]],
    model: str = 'aspatial',
    # scale_factors: List[float] = None,
    # x_translations: List[float] = None,
    # y_translations: List[float] = None
    # rotations: List[float] = None
) -> plt.Figure:
    scale_factors = [0.9, 1.1, 0.8, 0.9, 0.65, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
    x_translations = [0.0, 0.0, 0.0, -2.8, -7.6, 0.0, 0.0, 0.0, -1.55, -10.6, 0.0, 0.0, 0.0, -10.4, -7.3, 1.2, -1.3, -1.2, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    y_translations = [0.5, 0.5, 0.0, 2.35, 2.35, -1.35, 0.0, 0.0, -1.2, 2.0, 0.0, 0.6, 0.0, 6.35, 5.65, 11.0, 5.0, 4.5, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rotations = [50.0, 0.0, 0.0, -60.0, -50.0, 80.0, 0.0, 0.0, -70.0, -30.0, 0.0, 0.0, 0.0, -30.0, 60.0, 20.0, 90.0, 45.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Create the graph from the edgelist
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        G = nx.from_edgelist(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}

    # Calculate component sizes and total size
    component_sizes = [len(component) for component in connected_components]
    total_size = sum(component_sizes)

    # Position components in a grid layout
    grid_size = int(np.ceil(np.sqrt(len(connected_components))))
    max_component_size = len(connected_components[0])

    # Define a base scale for the largest component
    base_scale = 1.8  # Increased to make components larger

    # Calculate total width and height
    total_width = total_height = grid_size * 3  # Increased spacing between components

    for i, component in enumerate(connected_components):
        component_subgraph = G.subgraph(component)

        # Calculate center position for this component
        row = i // grid_size
        col = i % grid_size

        # Calculate cell size based on component size
        cell_size = np.sqrt(component_sizes[i] / total_size) * grid_size

        # Calculate center, adjusting for margins
        center_x = (col + 0.5) * (total_width / grid_size)
        center_y = -(row + 0.5) * (total_height / grid_size)
        center = (center_x, center_y)

        # Scale factor based on component size
        scale_factor = base_scale * (len(component) / max_component_size) ** 0.5 * cell_size

        component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)
        # Shift and scale the component to its position in the grid
        component_pos = {node: ((x * 0.8 + center[0]), (y * 0.8 + center[1]))
                         for node, (x, y) in component_pos.items()}

        pos.update(component_pos)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Apply additional scaling, translation, and rotation to each component
    for i, component in enumerate(connected_components):
        scale = scale_factors[i] if scale_factors and i < len(scale_factors) else 1
        x_trans = x_translations[i] if x_translations and i < len(x_translations) else 0
        y_trans = y_translations[i] if y_translations and i < len(y_translations) else 0
        rotation = rotations[i] if rotations and i < len(rotations) else 0

        # Calculate component center
        component_xs, component_ys = zip(*[pos[node] for node in component])
        component_center_x = sum(component_xs) / len(component)
        component_center_y = sum(component_ys) / len(component)

        for node in component:
            x, y = pos[node]

            # Translate to origin
            x -= component_center_x
            y -= component_center_y

            # Scale
            x *= scale
            y *= scale

            # Rotate
            angle = np.radians(rotation)
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)

            # Translate back and apply additional translation
            pos[node] = (x_rot + component_center_x + x_trans,
                         y_rot + component_center_y + y_trans)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Plot the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color='black', with_labels=False, node_size=200, width=5, edge_color='gray', ax=ax)

    # Remove axis
    ax.set_axis_off()

    # Set limits to show the entire graph
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    return fig


# FOR a1 = 0.2, a2 = 0.0, b1 = 10, run 1
def plot_particular_network_from_edgelist_6(
    edgelist: List[Tuple[int, int]],
    model: str = 'aspatial',
    # scale_factors: List[float] = None,
    # x_translations: List[float] = None,
    # y_translations: List[float] = None
    # rotations: List[float] = None
) -> plt.Figure:
    scale_factors = [0.7, 0.8, 0.65, 0.6, 0.5, 0.5, 0.4, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    x_translations = [0.0, 0.0, 0.0, 0.0, -10.92, 0.0, 0.0, 0.0, 0.0, -8.3, 0.0, 0.15, 0.0, 0.0, -5.02, 2.0, 2.0, 2.0, 1.4, -5.52, 1.8, 1.8, 1.8, 0.0, 0.0]
    y_translations = [0.0, 0.6, 0.0, 0.0, -5.32, 0.0, 0.0, 0.0, 0.0, 1.75, 0.0, 0.32, 0.0, 0.0, 7.23, 8.0, 8.0, 8.0, 9.2, 5.05, 7.65, 7.65, 7.65, 0.0, 0.0]
    rotations = [0.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 0.0, 37.0, 0.0, 0.0, 15.0, 45.0, -30.0, 15.0, -60.0, 0.0, 25.0, -40.0, 70.0, 0.0, 0.0]

    # Create the graph from the edgelist
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        G = nx.from_edgelist(edgelist)

    # Find all connected components
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
    node_colors = [cmap(norm(degrees[node])) for node in G.nodes()]

    # Set up the positions for the nodes
    pos = {}

    # Calculate component sizes and total size
    component_sizes = [len(component) for component in connected_components]
    total_size = sum(component_sizes)

    # Position components in a grid layout
    grid_size = int(np.ceil(np.sqrt(len(connected_components))))
    max_component_size = len(connected_components[0])

    # Define a base scale for the largest component
    base_scale = 1.8  # Increased to make components larger

    # Calculate total width and height
    total_width = total_height = grid_size * 3  # Increased spacing between components

    for i, component in enumerate(connected_components):
        component_subgraph = G.subgraph(component)

        # Calculate center position for this component
        row = i // grid_size
        col = i % grid_size

        # Calculate cell size based on component size
        cell_size = np.sqrt(component_sizes[i] / total_size) * grid_size

        # Calculate center, adjusting for margins
        center_x = (col + 0.5) * (total_width / grid_size)
        center_y = -(row + 0.5) * (total_height / grid_size)
        center = (center_x, center_y)

        # Scale factor based on component size
        scale_factor = base_scale * (len(component) / max_component_size) ** 0.5 * cell_size

        component_pos = nx.kamada_kawai_layout(component_subgraph, scale=scale_factor)
        # Shift and scale the component to its position in the grid
        component_pos = {node: ((x * 0.8 + center[0]), (y * 0.8 + center[1]))
                         for node, (x, y) in component_pos.items()}

        pos.update(component_pos)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Apply additional scaling, translation, and rotation to each component
    for i, component in enumerate(connected_components):
        scale = scale_factors[i] if scale_factors and i < len(scale_factors) else 1
        x_trans = x_translations[i] if x_translations and i < len(x_translations) else 0
        y_trans = y_translations[i] if y_translations and i < len(y_translations) else 0
        rotation = rotations[i] if rotations and i < len(rotations) else 0

        # Calculate component center
        component_xs, component_ys = zip(*[pos[node] for node in component])
        component_center_x = sum(component_xs) / len(component)
        component_center_y = sum(component_ys) / len(component)

        for node in component:
            x, y = pos[node]

            # Translate to origin
            x -= component_center_x
            y -= component_center_y

            # Scale
            x *= scale
            y *= scale

            # Rotate
            angle = np.radians(rotation)
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)

            # Translate back and apply additional translation
            pos[node] = (x_rot + component_center_x + x_trans,
                         y_rot + component_center_y + y_trans)

    # Recenter the entire layout
    all_xs, all_ys = zip(*pos.values())
    center_x = (max(all_xs) + min(all_xs)) / 2
    center_y = (max(all_ys) + min(all_ys)) / 2

    # Calculate the shift needed to center the layout
    shift_x = total_width / 2 - center_x
    shift_y = total_height / 2 - center_y

    # Apply the shift to all positions
    pos = {node: (x + shift_x, y + shift_y) for node, (x, y) in pos.items()}

    # Plot the graph
    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw(G, pos, node_color='black', with_labels=False, node_size=200, width=5, edge_color='gray', ax=ax)

    # Remove axis
    ax.set_axis_off()

    # Set limits to show the entire graph
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_height)

    return fig


def plot_lattice_from_edgelist(edgelist: List[Tuple[int, int]], rows: int, cols: int) -> plt.Figure:
    # Create the graph
    G = nx.Graph()

    # Add all nodes explicitly based on rows and cols
    G.add_nodes_from((i * cols + j + 1 for i in range(rows) for j in range(cols)))

    # Add edges from the edge list
    G.add_edges_from(edgelist)

    # Determine node degrees and apply the viridis colormap
    degrees = dict(G.degree())
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0.8*min(degrees.values()), vmax=1.25*max(degrees.values()))
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
    sm = ScalarMappable(cmap=cmap, norm=norm)
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


def plot_results(df: pd.DataFrame, save_path: str, file_name: str, model: str) -> None:
    # Calculate x-axis value
    N_mito = df['N_mito'][0]
    if model == 'aspatial':
        a1 = df['a1'][0]
        df['x_axis'] = a1 * N_mito / df['b1']
    else:
        df['x_axis'] = df['p']

    # Set the style for all plots
    plt.style.use('default')  # Reset to default style
    sns.set_style("ticks")  # Use seaborn's ticks style as a base

    # Define markers and get viridis colormap
    markers = ['s', 'o', '^']
    cmap = plt.get_cmap('viridis')

    font_tnr = FontProperties(family='Times New Roman', size=20)

    def setup_plot(ax, is_log_x=False, title=''):
        # Remove grid
        ax.grid(False)

        # Set outer box thickness
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        # Set up ticks for all sides
        ax.tick_params(which='both', width=1.5, direction='in', top=True, right=True, labelsize=20)
        ax.tick_params(which='major', length=6)
        ax.tick_params(which='minor', length=3)

        # Set log ticks if log scale
        if is_log_x:
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(numticks=7))
            ax.xaxis.set_minor_locator(LogLocator(subs='all'))
            ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator())

        ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Apply Times New Roman font to tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_tnr)

        # Increase font size for axis labels
        ax.yaxis.label.set_fontproperties(FontProperties(family='Times New Roman', size=20))
        ax.xaxis.label.set_fontproperties(FontProperties(family='Times New Roman', size=20))

        # Set title with larger font size
        ax.text(0.5, 1.05, title, fontsize=18, fontweight='bold', fontfamily='Times New Roman',
                ha='center', va='bottom', transform=ax.transAxes)

    # 1. Plot number of components
    fig, ax = plt.subplots(figsize=(10, 8))
    color = cmap(0.3)  # Use a color from the darker half of viridis
    sns.lineplot(data=df, x='x_axis', y='number_of_components', marker='s', color=color,
                 markerfacecolor='none', markersize=10, markeredgewidth=2,
                 markeredgecolor=color, ax=ax)
    if model == 'aspatial':
        title = f'Number of Components vs Rate (a1={a1}, N_mito={N_mito})'
        ax.set_xlabel('a1 * N_mito / b1')
    else:
        title = f'Number of Components vs Threshold (N_mito={N_mito})'
        ax.set_xlabel('p')
    ax.set_ylabel('Number of Components')
    setup_plot(ax, is_log_x=(model == 'aspatial'), title=title)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(join(save_path, f"{file_name}_number_of_components.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot cycle categories
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, column in enumerate(['no_cycles', 'one_cycle', 'many_cycles']):
        color = cmap(0.2 + i * 0.2)  # Use colors from the darker half of viridis
        sns.lineplot(data=df, x='x_axis', y=column, label=column.replace('_', ' ').title(),
                     marker=markers[i], color=color, markerfacecolor='none', markersize=10,
                     markeredgewidth=2, markeredgecolor=color, ax=ax)
    if model == 'aspatial':
        title = f'Cycle Categories vs Rate (a1={a1}, N_mito={N_mito})'
        ax.set_xlabel('a1 * N_mito / b1')
    else:
        title = f'Cycle Categories vs Threshold (N_mito={N_mito})'
        ax.set_xlabel('p')
    ax.set_ylabel('Fraction of Nodes')
    setup_plot(ax, is_log_x=(model == 'aspatial'), title=title)
    legend = plt.legend(prop=FontProperties(family='Times New Roman', size=16))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(join(save_path, f"{file_name}_cycle_categories.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Plot degree distribution
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, column in enumerate(['degree_1', 'degree_2', 'degree_3']):
        color = cmap(0.2 + i * 0.2)  # Use colors from the darker half of viridis
        sns.lineplot(data=df, x='x_axis', y=column, label=f'Degree {i+1}',
                     marker=markers[i], color=color, markerfacecolor='none', markersize=10,
                     markeredgewidth=2, markeredgecolor=color, ax=ax)
    if model == 'aspatial':
        title = f'Degree Distribution vs Rate (a1={a1}, N_mito={N_mito})'
        ax.set_xlabel('a1 * N_mito / b1')
    else:
        title = f'Degree Distribution vs Threshold (N_mito={N_mito})'
        ax.set_xlabel('p')
    ax.set_ylabel('Fraction of Nodes')
    setup_plot(ax, is_log_x=(model == 'aspatial'), title=title)
    legend = plt.legend(prop=FontProperties(family='Times New Roman', size=16))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(join(save_path, f"{file_name}_degree_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
