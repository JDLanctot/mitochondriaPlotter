from dataclasses import dataclass
from os import makedirs
from os.path import dirname, join
from typing import Tuple, Literal
import numpy as np
from simple_parsing import field, ArgumentParser

from mitochondriaplotter.plot import plot_network_from_edgelist, plot_lattice_from_edgelist, gen_lattice
from mitochondriaplotter.util import set_seed


@dataclass
class Options:
    """ Command line options """
    file_name: str = field(alias='-f', required=True, help="Output file name")
    output_file: str = field(alias='-o', required=True,
                             help="Output directory")
    model: Literal['aspatial', 'spatial', 'gen_lattice'] = field(
        alias='-m', required=True, help="Model type: 'aspatial', 'spatial', or 'gen_lattice'")
    seed: int = field(alias='-s', default=None, help="Random seed")

    # Options for loading edge list
    load_name: str = field(alias='-l', default=None,
                           help="Input file name (for 'graph' and 'lattice' types)")

    # Options for lattice generation and plotting
    lattice_size: Tuple[int, int] = field(
        alias='-n', default=(5, 5), help="Lattice size (rows, columns)")
    p: float = field(alias='-p', default=0.4, help="Percolation threshold")
    k: int = field(alias='-k', default=4, help="Lattice degree")


def load_edgelist(load_path: str, load_name: str, model: str = 'aspatial') -> np.ndarray:
    if model == 'aspatial':
        return np.loadtxt(join(load_path, f"{load_name}.out"))
    else:
        return np.loadtxt(join(load_path, f"{load_name}.dat"))


def main(options: Options):
    if options.seed is not None:
        set_seed(options.seed)

    save_path = join(options.output_file, "images")
    makedirs(dirname(save_path), exist_ok=True)

    if options.model == 'gen_lattice':
        edgelist = gen_lattice(
            options.lattice_size[0], options.lattice_size[1], options.p, options.k)
        fig = plot_lattice_from_edgelist(
            edgelist, options.lattice_size[0], options.lattice_size[1])
    else:
        if options.load_name is None:
            raise ValueError(
                "load_name is required for 'aspatial' and 'spatial' types")
        load_path = join(options.output_file, "data")
        edgelist = load_edgelist(load_path, options.load_name, options.model)

        if options.model == 'spatial':
            fig = plot_network_from_edgelist(edgelist, options.model)
            # fig = plot_lattice_from_edgelist(edgelist, options.lattice_size[0], options.lattice_size[1])
        elif options.model == 'aspatial':
            fig = plot_network_from_edgelist(edgelist, options.model)
        else:
            raise ValueError(f"Unknown type: {options.type}")

    fig.savefig(join(save_path, f"{options.file_name}.png"))
    print(f"Plot saved to {join(save_path, options.file_name)}.png")


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(args.options)
