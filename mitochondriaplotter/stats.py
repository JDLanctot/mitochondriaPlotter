import networkx as nx
import numpy as np
from typing import Tuple, Set

__all__ = []
__all__.extend([
    'get_degree_distribution',
    'get_number_of_components',
    'get_relative_component_sizes',
    'get_number_loops',
    'fraction_of_nodes_in_loops',
    'categorize_nodes_by_cycles',
])


def get_degree_distribution(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    degree_sequence = sorted((d for _, d in G.degree()), reverse=True)
    return np.unique(degree_sequence, return_counts=True)


def get_number_of_components(G: nx.Graph) -> int:
    connected_components = sorted(
        nx.connected_components(G), key=len, reverse=True)
    return len(connected_components)


def get_relative_component_sizes(G: nx.Graph) -> Tuple[np.ndarray, np.ndarray]:
    connected_components = sorted(
        nx.connected_components(G), key=len, reverse=True)
    length = len(G)
    sizes = np.array([len(cc)/length for cc in connected_components])
    return sizes, np.ones(len(sizes))


def get_edges_in_component(G: nx.Graph, component: Set) -> int:
    """
    Get the number of edges in a specific connected component.

    :param G: The full NetworkX graph
    :param component: A set of nodes forming a connected component
    :return: Number of edges in the component
    """
    subgraph = G.subgraph(component)
    return subgraph.number_of_edges()


def get_number_loops(G: nx.Graph) -> int:
    cycles = 0
    for cc in sorted(nx.connected_components(G), key=len, reverse=True):
        num_nodes = len(cc)
        num_edges = get_edges_in_component(G, cc)
        cycles += num_edges - num_nodes + 1
    cycles_2 = len(nx.cycle_basis(G))
    assert cycles == cycles_2
    return cycles


def fraction_of_nodes_in_loops(G: nx.Graph) -> float:
    cycles = nx.cycle_basis(G)

    nodes_in_cycles = set()
    for cycle in cycles:
        nodes_in_cycles.update(cycle)

    total_nodes = G.number_of_nodes()
    fraction_in_cycles = len(nodes_in_cycles) / \
        total_nodes if total_nodes > 0 else 0
    return fraction_in_cycles


def categorize_nodes_by_cycles(G: nx.Graph) -> dict:
    # Find all connected components
    components = list(nx.connected_components(G))

    # Initialize counters
    node_counts = {
        'no_cycles': 0,
        'one_cycle': 0,
        'many_cycles': 0
    }

    for component in components:
        subgraph = G.subgraph(component)
        cycles = nx.cycle_basis(subgraph)

        if len(cycles) == 0:
            node_counts['no_cycles'] += len(component)
        elif len(cycles) == 1:
            node_counts['one_cycle'] += len(component)
        else:
            node_counts['many_cycles'] += len(component)

    total_nodes = G.number_of_nodes()

    # Calculate fractions
    fractions = {
        key: count / total_nodes if total_nodes > 0 else 0
        for key, count in node_counts.items()
    }

    # Add raw counts to the result
    fractions.update({f'{key}_count': count for key,
                     count in node_counts.items()})

    return fractions
