from dataclasses import asdict
from dataclasses import dataclass
from os import makedirs, getcwd
from os.path import dirname, join
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from simple_parsing import field, ArgumentParser
from scipy.io import loadmat, savemat

# from mitochondriaplotter.params import HyperParams
from mitochondriaplotter.plot import plot_network_from_edgelist, plot_lattice_from_edgelist, gen_lattice
from mitochondriaplotter.util import set_seed, double_ended_to_edgelist
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

    # whether it is a graph or lattice
    type: str = field(alias='-t', required=False, default='graph')

    # random seed
    seed: int = field(alias='-s', default=None, required=False)

    # Size of a lattice
    lattice_size: Tuple[int,int] = field(alias='-n', required=False, default=(5,5))

    # percolation threshold
    p: float = field(alias='-p', required=False, default=0.4)

    # lattice degree
    k: int = field(alias='-k', required=False, default=4)


def main(file_name: str, load_name: str, output_file: str, type: str = 'graph',
         seed: int = None, lattice_size: Tuple[int,int] = (5,5), p: float = 0.4,
         k: int = 4): #config_file: str,
    if seed is not None:
        set_seed(seed)

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
    save_path = join(output_file, "images")
    makedirs(dirname(save_path), exist_ok=True)
    if type != 'gen_lattice':
        load_path = join(output_file, "data")
        data = loadmat(join(load_path, f"{load_name}.mat"))
        edge_list = data['edge_list']

    # plot and save images
    if type == 'gen_lattice':
        edge_list = gen_lattice(lattice_size[0], lattice_size[1], p, k)
        fig = plot_lattice_from_edgelist(edge_list, lattice_size[0], lattice_size[1])
    elif type == 'lattice':
        fig = plot_lattice_from_edgelist(edge_list, lattice_size[0], lattice_size[1])
    else:
        fig = plot_network_from_edgelist(edge_list)
    fig.savefig(join(save_path, f"{file_name}.png"))

if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(**asdict(args.options))

