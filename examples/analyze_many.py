from dataclasses import asdict
from dataclasses import dataclass
from os import makedirs, getcwd
from os.path import dirname, join
from simple_parsing import field, ArgumentParser
from scipy.io import loadmat, savemat
import numpy as np
import networkx as nx
from typing import Tuple

# from mitochondriaplotter.params import HyperParams
from mitochondriaplotter.plot import plot_probability_distribution
from mitochondriaplotter.util import set_seed, double_ended_to_edgelist, coalesced_graph
from mitochondriaplotter.stats import get_degree_distribution, get_relative_component_sizes, get_number_loops, fraction_of_nodes_in_loops, categorize_nodes_by_cycles

import shutil

@dataclass
class Options:
    """ options """
    # save file name
    file_name: str = field(alias='-f', required=True)

    # .yml file containing HyperParams
    # config_file: str = field(alias='-c', required=True)

    # where to save the plot
    output_file: str = field(alias='-o', required=True)

    # what the parameters we are checking
    a_s: Tuple[float, float] = field(alias='-a', required=True)

    # random seed
    seed: int = field(alias='-s', default=None, required=False)

def main(file_name: str, output_file: str, a_s: Tuple[float, float],
         seed: int = None): #config_file: str,
    if seed is not None:
        set_seed(seed)

    a1, a2 = a_s

    # Testing edge_list manually
    # edge_list = double_ended_to_edgelist([
    #     [[1,1], [2,2], [3,1], [5,2], [7,1], [8,1], [9,2]],
    #     [[2,1], [3,1], [4,1], [6,1], [8,1], [9,2], [10,1]]
    # ])
    # save_path = join(output_file, "data")
    # makedirs(dirname(save_path), exist_ok=True)
    # savemat(join(save_path, f"{load_name}.mat"), {"edge_list": edge_list})

    # File locations and parameters
    # hp = HyperParams.load(Path(getcwd() + config_file))
    # config_file_path = join(getcwd(), config_file[1:])
    # shutil.copyfile(config_file_path, f"{dirname(save_path)}/images/{file_name}.yml")

    # Load and Save locations
    save_path = join(output_file, "output")
    makedirs(dirname(save_path), exist_ok=True)
    fs = [] # fraction of nodes in atleast one loop
    Ns = [] # number of loops
    ks = [] # degree distribution
    Lcc_Gs = [] # fractional size of largest Connected component
    Cs = [] # number of connected components
    f_is = [] # fraction of nodes in a cc with no loops, one loop, or many loops
    samples = 10
    for b1 in [0.001, 0.1, 1, 50, 100, 500, 1000]:
        f = 0
        N = 0
        k_vals = [0, 0, 0]
        Lcc_rel = 0
        C = 0
        f_i = [0, 0, 0]
        for i in range(1, samples + 1):
            load_name = f"edge_ends_a1_{a1}_b1_{b1}_a2_{a2}_run{i}.out"
            load_path = join(output_file, "data", f"Aspatial_model_data_a1_{a1}_a2_{a2}", f"b1_{b1}")
            # data = loadmat(join(load_path, f"{load_name}.dat"))
            # edge_list = data['edge_list']
            edge_list = np.loadtxt(join(load_path, load_name))
            # connection_info = np.fromfile(join(load_path, f"{load_name}.dat"), dtype=float)

            # Generate graph
            # G = nx.from_edgelist(edge_list)
            G = coalesced_graph(edge_list)

            # Descriptives
            f += fraction_of_nodes_in_loops(G)
            N += get_number_loops(G)
            print("Fraction of nodes in loops: ", f)
            print("Total number of loops: ", N)

            data = get_degree_distribution(G)
            k_rels = np.divide(data[1], len(G))
            for j,k in enumerate(data[0]):
                k_vals[k - 1] += float(k_rels[j])
            print(data)
            # fig = plot_probability_distribution(data, bins=3)
            # fig.savefig(join(save_path, f"{file_name}_degree.png"))

            data = get_relative_component_sizes(G)
            Lcc_rel += float(data[0][0])
            C += len(data[0])
            print(data)
            # fig = plot_probability_distribution(data)
            # fig.savefig(join(save_path, f"{file_name}_component_sizes.png"))

            result = categorize_nodes_by_cycles(G)
            f_i[0] += result['no_cycles']
            f_i[1] += result ['one_cycle']
            f_i[2] += result['many_cycles']

        f /= samples
        N /= samples
        C /= samples
        fs.append(f)
        f_is.append(np.divide(f_i, samples))
        Ns.append(N)
        Cs.append(C)
        ks.append(np.divide(k_vals, samples))
        Lcc_Gs.append(float(np.divide(Lcc_rel, samples)))
    ks = np.array(ks).T
    f_is = np.array(f_is).T

if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(**asdict(args.options))



