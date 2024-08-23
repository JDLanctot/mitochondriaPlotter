from dataclasses import dataclass
from os import makedirs
from os.path import dirname, join
from simple_parsing import field, ArgumentParser
import numpy as np
import networkx as nx
from typing import Tuple, Dict, List, Literal
import pandas as pd

from mitochondriaplotter.plot import plot_probability_distribution, plot_results
from mitochondriaplotter.util import set_seed, coalesced_graph
from mitochondriaplotter.stats import (
    get_degree_distribution, get_relative_component_sizes,
    get_number_loops, fraction_of_nodes_in_loops, categorize_nodes_by_cycles
)


@dataclass
class Options:
    """ Command line options for mitochondria network analysis """
    file_name: str = field(alias='-f', required=True,
                           help="Output file name prefix")
    output_file: str = field(alias='-o', required=True,
                             help="Path to output directory")
    model: Literal['aspatial', 'spatial'] = field(
        alias='-m', required=True, help="Model type: 'aspatial' or 'spatial'")

    # Model parameters
    N_mito: int = field(alias='-N', default=50, help="Number of mitochondria")
    a_s: Tuple[float, float] = field(
        alias='-a', default=(0.002, 0.01), help="Tuple of a1 and a2 values for aspatial model")
    tau: float = field(alias='-t', default=1.0,
                       help="Tau value for lattice model")
    dim: str = field(alias='-d', default='2d',
                     help="Dimension for lattice model: '2d' or 'quasi1d'")

    # General options
    seed: int = field(alias='-s', default=None,
                      help="Random seed for reproducibility")

    def __post_init__(self):
        if self.model == 'aspatial' and self.dim != '2d':
            raise ValueError(
                "Dimension parameter is not used for aspatial model")
        if self.model == 'spatial' and len(self.a_s) != 2:
            raise ValueError("a_s parameter is not used for spatial model")


def process_sample(load_path: str, load_name: str, model: str = 'aspatial', max_nodes: int = 441) -> dict:
    edgelist = np.loadtxt(join(load_path, load_name))
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        # Convert edgelist to a list of tuples with integer node IDs
        edgelist = [(int(edge[0]), int(edge[1])) for edge in edgelist]
        G = nx.from_edgelist(edgelist)

        # Add any missing nodes up to max_nodes
        existing_nodes = set(G.nodes())
        for node in range(max_nodes):
            if node not in existing_nodes:
                G.add_node(node)

    degree_dist = get_degree_distribution(G)
    component_sizes = get_relative_component_sizes(G)
    cycle_categories = categorize_nodes_by_cycles(G)

    # Convert degree distribution to fractions and ensure 3 values
    total_nodes = sum(degree_dist[1])
    degree_fractions = np.zeros(3)
    for degree, count in zip(degree_dist[0], degree_dist[1]):
        if degree <= 3:
            degree_fractions[degree - 1] = count / total_nodes

    return {
        'fraction_in_loops': fraction_of_nodes_in_loops(G),
        'number_of_loops': get_number_loops(G),
        'degree_distribution': degree_fractions,
        'largest_component_size': component_sizes[0][0],
        'number_of_components': len(component_sizes[0]),
        'no_cycles': cycle_categories['no_cycles'],
        'one_cycle': cycle_categories['one_cycle'],
        'many_cycles': cycle_categories['many_cycles']
    }


def extend_results(avg_results: Dict, sample_results: List) -> Dict:
    # Main Metrics
    avg_results['fraction_in_loops'] = np.mean(
        [r['fraction_in_loops'] for r in sample_results])
    avg_results['number_of_loops'] = np.mean(
        [r['number_of_loops'] for r in sample_results])
    avg_results['largest_component_size'] = np.mean(
        [r['largest_component_size'] for r in sample_results])
    avg_results['number_of_components'] = np.mean(
        [r['number_of_components'] for r in sample_results])
    avg_results['no_cycles'] = np.mean(
        [r['no_cycles'] for r in sample_results])
    avg_results['one_cycle'] = np.mean(
        [r['one_cycle'] for r in sample_results])
    avg_results['many_cycles'] = np.mean(
        [r['many_cycles'] for r in sample_results])

    # Average degree distribution
    avg_degree_dist = np.mean([r['degree_distribution']
                              for r in sample_results], axis=0)
    avg_results['degree_1'] = avg_degree_dist[0]
    avg_results['degree_2'] = avg_degree_dist[1]
    avg_results['degree_3'] = avg_degree_dist[2]

    return avg_results


def main(options: Options):
    if options.seed is not None:
        set_seed(options.seed)

    samples = 10
    results = []

    if options.model == 'aspatial':
        a1, a2 = options.a_s
        if a1 == 10:
            a1 = int(a1)
        params_f = f'a1_{a1}_a2_{a2}'
    else:
        params_f = f'dim_{options.dim}_tau_{int(options.tau)}'

    save_path = join(options.output_file, "output", options.model, params_f)
    makedirs(dirname(save_path), exist_ok=True)

    if options.model == 'aspatial':
        # b1_values = [0.001, 0.1, 1, 50, 100, 500, 1000]
        b1_values = [0.001, 0.01, 0.1, 1, 10, 20,
                     50, 80, 100, 150, 200, 500, 800, 1000]

        for b1 in b1_values:
            sample_results = []
            for i in range(1, samples + 1):
                load_name = f"edge_ends_a1_{a1}_b1_{b1}_a2_{a2}_run{i}.out"
                load_path = join(options.output_file, "data",
                                 f"Aspatial_model_data_a1_{a1}_a2_{a2}", f"b1_{b1}")
                sample_results.append(process_sample(
                    load_path, load_name, options.model))

            # Setup the metrics or values that are model specific
            avg_results = {
                'a1': a1,
                'a2': a2,
                'b1': b1,
                'N_mito': options.N_mito
            }

            # Add the metrics shared across models
            avg_results = extend_results(avg_results, sample_results)
            results.append(avg_results)

    else:
        a1 = 1

        if options.dim == 'quasi1d':
            q = 'quasi'
        else:
            q = ''

        p_values = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99]
        for p in p_values:
            sample_results = []
            for i in range(1, samples + 1):
                load_name = f"lattice_node_connection_info_p{p}_run{i}.dat"
                load_path = join(options.output_file, "data",
                                 f"latticeModel_data_{options.dim}_tau{int(options.tau)}", f"p_{p}")
                sample_results.append(process_sample(
                    load_path, f"{q}{load_name}", options.model))

            # Setup the metrics or values that are model specific
            avg_results = {
                'p': p,
                'N_mito': options.N_mito
            }

            # Add the metrics shared across models
            avg_results = extend_results(avg_results, sample_results)
            results.append(avg_results)

    df_results = pd.DataFrame(results)
    plot_results(df_results, save_path, options.file_name, options.model)

    # Save results
    df_results.to_csv(
        join(save_path, f"{options.file_name}_results.csv"), index=False)

    print(df_results)


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(args.options)
