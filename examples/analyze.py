from dataclasses import asdict
from dataclasses import dataclass
from os import makedirs, getcwd
from os.path import dirname, join
from simple_parsing import field, ArgumentParser
from scipy.io import loadmat, savemat
import numpy as np
import networkx as nx

# from mitochondriaplotter.params import HyperParams
from mitochondriaplotter.plot import plot_probability_distribution
from mitochondriaplotter.util import set_seed, double_ended_to_edgelist, coalesced_graph
from mitochondriaplotter.stats import get_degree_distribution, get_relative_component_sizes, get_number_loops, fraction_of_nodes_in_loops

import shutil

@dataclass
class Options:
    """ options """
    # save file name
    file_name: str = field(alias='-f', required=True)

    # data file name
    load_name: str = field(alias='-l', required=True)

    # .yml file containing HyperParams
    # config_file: str = field(alias='-c', required=True)

    # where to save the plot
    output_file: str = field(alias='-o', required=True)

    # random seed
    seed: int = field(alias='-s', default=None, required=False)

def main(file_name: str, load_name: str, output_file: str, seed: int = None): #config_file: str,
    if seed is not None:
        set_seed(seed)

    model = 'aspatial'

    # Testing edgelist manually
    # edgelist = double_ended_to_edgelist([
    #     [[1,1], [2,2], [3,1], [5,2], [7,1], [8,1], [9,2]],
    #     [[2,1], [3,1], [4,1], [6,1], [8,1], [9,2], [10,1]]
    # ])
    # save_path = join(output_file, "data")
    # makedirs(dirname(save_path), exist_ok=True)
    # savemat(join(save_path, f"{load_name}.mat"), {"edgelist": edgelist})

    # File locations and parameters
    # hp = HyperParams.load(Path(getcwd() + config_file))
    # config_file_path = join(getcwd(), config_file[1:])
    # shutil.copyfile(config_file_path, f"{dirname(save_path)}/images/{file_name}.yml")

    # Load and Save locations
    save_path = join(output_file, "images")
    makedirs(dirname(save_path), exist_ok=True)
    load_path = join(output_file, "data")
    # data = loadmat(join(load_path, f"{load_name}.dat"))
    # edgelist = data['edgelist']
    edgelist = np.loadtxt(join(load_path, f"{load_name}.out"))
    # connection_info = np.fromfile(join(load_path, f"{load_name}.dat"), dtype=float)

    # Generate graph
    if model == 'aspatial':
        G = coalesced_graph(edgelist)
    else:
        max_nodes = 500
        # Convert edgelist to a list of tuples with integer node IDs
        edgelist = [(int(edge[0]), int(edge[1])) for edge in edgelist]
        G = nx.from_edgelist(edgelist)

        # Add any missing nodes up to max_nodes
        existing_nodes = set(G.nodes())
        for node in range(max_nodes):
            if node not in existing_nodes:
                G.add_node(node)


    # Descriptives
    f = fraction_of_nodes_in_loops(G)
    n = get_number_loops(G)
    print("Fraction of nodes in loops: ", f)
    print("Total number of loops: ", n)

    # plot and save images
    data = get_degree_distribution(G)
    print(data)
    fig = plot_probability_distribution(data, bins=3)
    fig.savefig(join(save_path, f"{file_name}_degree.png"))

    data = get_relative_component_sizes(G)
    print(data)
    fig = plot_probability_distribution(data)
    fig.savefig(join(save_path, f"{file_name}_component_sizes.png"))

if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(**asdict(args.options))


