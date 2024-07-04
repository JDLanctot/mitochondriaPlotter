from dataclasses import asdict, dataclass
from os import makedirs, getcwd
from os.path import dirname, join
from simple_parsing import field, ArgumentParser
import numpy as np
import networkx as nx
from typing import Tuple
import pandas as pd

from mitochondriaplotter.plot import plot_probability_distribution, plot_results
from mitochondriaplotter.util import set_seed, coalesced_graph
from mitochondriaplotter.stats import (
    get_degree_distribution, get_relative_component_sizes,
    get_number_loops, fraction_of_nodes_in_loops, categorize_nodes_by_cycles
)

@dataclass
class Options:
    """ options """
    file_name: str = field(alias='-f', required=True)
    output_file: str = field(alias='-o', required=True)
    a_s: Tuple[float, float] = field(alias='-a', required=True)
    N_mito: int = field(alias='-N', default=50, required=False)
    seed: int = field(alias='-s', default=None, required=False)

def process_sample(load_path: str, load_name: str) -> dict:
    edge_list = np.loadtxt(join(load_path, load_name))
    G = coalesced_graph(edge_list)

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

def main(file_name: str, output_file: str, a_s: Tuple[float, float],
         N_mito: int, seed: int = None):
    if seed is not None:
        set_seed(seed)

    a1, a2 = a_s

    save_path = join(output_file, "output")
    makedirs(dirname(save_path), exist_ok=True)

    b1_values = [0.001, 0.1, 1, 50, 100, 500, 1000]
    samples = 10
    results = []

    for b1 in b1_values:
        sample_results = []
        for i in range(1, samples + 1):
            load_name = f"edge_ends_a1_{a1}_b1_{b1}_a2_{a2}_run{i}.out"
            load_path = join(output_file, "data", f"Aspatial_model_data_a1_{a1}_a2_{a2}", f"b1_{b1}")
            sample_results.append(process_sample(load_path, load_name))

        avg_results = {
            'b1': b1,
            'fraction_in_loops': np.mean([r['fraction_in_loops'] for r in sample_results]),
            'number_of_loops': np.mean([r['number_of_loops'] for r in sample_results]),
            'largest_component_size': np.mean([r['largest_component_size'] for r in sample_results]),
            'number_of_components': np.mean([r['number_of_components'] for r in sample_results]),
            'no_cycles': np.mean([r['no_cycles'] for r in sample_results]),
            'one_cycle': np.mean([r['one_cycle'] for r in sample_results]),
            'many_cycles': np.mean([r['many_cycles'] for r in sample_results]),
        }

        # Average degree distribution
        avg_degree_dist = np.mean([r['degree_distribution'] for r in sample_results], axis=0)
        avg_results['degree_1'] = avg_degree_dist[0]
        avg_results['degree_2'] = avg_degree_dist[1]
        avg_results['degree_3'] = avg_degree_dist[2]

        results.append(avg_results)

    df_results = pd.DataFrame(results)
    plot_results(df_results, save_path, file_name, a1, N_mito)
    import ipdb
    ipdb.set_trace()
    # Save results
    df_results.to_csv(join(save_path, f"{file_name}_results.csv"), index=False)

    print(df_results)

if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(**asdict(args.options))
