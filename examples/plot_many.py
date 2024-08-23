import os
from os import makedirs
from os.path import join, exists
from typing import Tuple, Literal, Union
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from simple_parsing import field, ArgumentParser

from mitochondriaplotter.plot import plot_network_from_edgelist, plot_lattice_from_edgelist, gen_lattice
from mitochondriaplotter.util import set_seed


@dataclass
class Options:
    """ Command line options """
    file_name: str = field(alias='-f', required=True,
                           help="Output file name prefix")
    output_file: str = field(alias='-o', required=True,
                             help="Path to output directory")
    model: Literal['aspatial', 'spatial', 'gen_lattice'] = field(
        alias='-m', required=True, help="Model type: 'aspatial', 'spatial', or 'gen_lattice'")
    seed: int = field(alias='-s', default=None, help="Random seed")

    # Model parameters
    N_mito: int = field(alias='-N', default=50, help="Number of mitochondria")
    a_s: Tuple[float, float] = field(
        alias='-a', default=(0.002, 0.01), help="Tuple of a1 and a2 values for aspatial model")
    tau: float = field(alias='-t', default=1.0,
                       help="Tau value for lattice model")
    dim: str = field(alias='-d', default='2d',
                     help="Dimension for lattice model: '2d' or 'quasi1d'")

    # Options for loading edge list
    load_name: str = field(alias='-l', default=None,
                           help="Input file name (for 'graph' and 'lattice' types)")

    # Options for lattice generation and plotting
    lattice_size: Tuple[int, int] = field(
        alias='-n', default=(5, 5), help="Lattice size (rows, columns)")
    p: float = field(alias='-p', default=0.4, help="Percolation threshold")
    k: int = field(alias='-k', default=4, help="Lattice degree")

    # Number of samples
    samples: int = field(alias='-S', default=10,
                         help="Number of samples to process")


def get_data_path(model: str, a1: float, a2: float, dim: str = None, tau: float = None, b1: float = None) -> str:
    base_dir = os.getcwd()
    if model == 'aspatial':
        return join(base_dir, 'examples', 'data', f'Aspatial_model_data_a1_{a1}_a2_{a2}', f'b1_{b1}')
    elif model == 'spatial':
        return join(base_dir, 'examples', 'data', f'Spatial_model_data_dim_{dim}_tau_{tau}')
    else:
        raise ValueError(f"Unknown model type: {model}")


def get_image_path(model: str, p1: Union[float, str], p2: Union[float, int], var: float) -> str:
    base_dir = os.getcwd()
    if model == 'aspatial':
        return join(base_dir, 'examples', 'images', model, f'a1_{p1}_a2_{p2}', f'b1_{var}')
    elif model == 'spatial':
        return join(base_dir, 'examples', 'images', model, f'dim_{p1}_tau_{p2}', f'p_{var}')
    else:
        raise ValueError(f"Unknown model type: {model}")


def load_edgelist(model: str, a1: float, a2: float, b1: float, run: int, dim: str = None, tau: float = None) -> np.ndarray:
    if model == 'aspatial':
        data_path = get_data_path(model, a1, a2, b1=b1)
        file_name = f'edge_ends_a1_{a1}_b1_{b1}_a2_{a2}_run{run}.out'
    else:  # spatial
        data_path = get_data_path(model, a1, a2, dim=dim, tau=tau)
        file_name = f'edge_ends_dim_{dim}_tau_{tau}_run{run}.dat'

    full_path = join(data_path, file_name)
    if not exists(full_path):
        raise FileNotFoundError(f"Data file not found: {full_path}")
    return np.loadtxt(full_path)


def main(options: Options):
    if options.seed is not None:
        set_seed(options.seed)

    a1, a2 = options.a_s
    if a1 == 10:  # because the file folders given didn't have 10.0 and instead had 10 in this unique case...
        a1 = int(a1)

    if options.model == 'gen_lattice':
        save_path = join(os.getcwd(), 'examples', 'images', 'gen_lattice')
        makedirs(save_path, exist_ok=True)
        edgelist = gen_lattice(
            options.lattice_size[0], options.lattice_size[1], options.p, options.k)
        fig = plot_lattice_from_edgelist(
            edgelist, options.lattice_size[0], options.lattice_size[1])
        fig.savefig(join(save_path, f"{options.file_name}.png"))
        print(f"Plot saved to {join(save_path, options.file_name)}.png")
    else:

        if options.model == 'spatial':
            p_values = [0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99]
            for p in p_values:
                for run in range(1, options.samples + 1):
                    save_path = get_image_path(
                        options.model, options.dim, options.tau, p)
                    makedirs(save_path, exist_ok=True)

                    edgelist = load_edgelist(
                        options.model, a1, a2, p, run, dim=options.dim, tau=options.tau)
                    fig = plot_network_from_edgelist(edgelist, options.model)
                    save_name = f"{options.file_name}_dim_{options.dim}_tau_{options.tau}_run{run}.png"
                    fig.savefig(join(save_path, save_name))
                    print(f"Plot saved to {join(save_path, save_name)}")
                    plt.close()
        elif options.model == 'aspatial':
            b1_values = [0.001, 0.01, 0.1, 1, 10, 20,
                         50, 80, 100, 150, 200, 500, 800, 1000]
            for b1 in b1_values:
                for run in range(1, options.samples + 1):
                    save_path = get_image_path(options.model, a1, a2, b1)
                    makedirs(save_path, exist_ok=True)

                    edgelist = load_edgelist(options.model, a1, a2, b1, run)
                    fig = plot_network_from_edgelist(edgelist, options.model)
                    save_name = f"{options.file_name}_a1_{a1}_a2_{a2}_b1_{b1}_run{run}.png"
                    fig.savefig(join(save_path, save_name))
                    print(f"Plot saved to {join(save_path, save_name)}")
                    plt.close()
        else:
            raise ValueError(f"Unknown model type: {options.model}")


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(args.options)
