from dataclasses import dataclass
from simple_parsing import field, ArgumentParser
import pandas as pd
import os
from os.path import join, exists
from typing import Tuple
import numpy as np


@dataclass
class Options:
    """ Command line options for CSV to text conversion """
    model: str = field(alias='-m', required=True,
                       help="Model type: 'aspatial' or 'spatial'")
    data_type: str = field(alias='-t', required=True,
                           help="Data type: 'cycles', 'components', or 'degrees'")

    # Model parameters
    a_s: Tuple[float, float] = field(
        alias='-a', default=(0.002, 0.01), help="Tuple of a1 and a2 values for aspatial model")
    tau: float = field(alias='-T', default=1.0,
                       help="Tau value for lattice model")
    dim: str = field(alias='-d', default='2d',
                     help="Dimension for lattice model: '2d' or 'quasi1d'")

    input_file: str = field(alias='-i', required=False,
                            default='analysis_results', help="Name of input CSV file")


def get_data_path(model: str, a1: float, a2: float, dim: str = None, tau: float = None) -> str:
    base_dir = os.getcwd()
    if model == 'aspatial':
        return join(base_dir, 'examples', 'output', model, f'a1_{a1}_a2_{a2}')
    elif model == 'spatial':
        return join(base_dir, 'examples', 'output', model, f'dim_{dim}_tau_{tau}')
    else:
        raise ValueError(f"Unknown model type: {model}")


def load_dataframe(model: str, a1: float, a2: float, dim: str = None, tau: float = None, input_file: str = 'analysis_results') -> np.ndarray:
    if model == 'aspatial':
        data_path = get_data_path(model, a1, a2)
    else:  # spatial
        data_path = get_data_path(model, a1, a2, dim=dim, tau=tau)
    file_name = f'{input_file}.csv'

    full_path = join(data_path, file_name)
    if not exists(full_path):
        raise FileNotFoundError(f"Data file not found: {full_path}")
    return pd.read_csv(full_path)


def save_dir(model: str, a1: float, a2: float, dim: str = None, tau: float = None) -> np.ndarray:
    if model == 'aspatial':
        data_path = get_data_path(model, a1, a2)
    else:  # spatial
        data_path = get_data_path(model, a1, a2, dim=dim, tau=tau)
    return data_path


def process_csv(options: Options):
    a1, a2 = options.a_s
    if a1 == 10:  # because the file folders given didn't have 10.0 and instead had 10 in this unique case...
        a1 = int(a1)

    # Read the CSV file
    df = load_dataframe(options.model, a1, a2,
                        dim=options.dim, tau=options.tau)

    # Determine the columns to extract based on the model and data type
    if options.model == 'aspatial':
        para_col = 'b1'
        x_col = df['a1'] * df['N_mito'] / df['b1']

        # Extract a1 and a2 values for the file name
        a1 = df['a1'].iloc[0] if 'a1' in df.columns else 'NA'
        a2 = df['a2'].iloc[0] if 'a2' in df.columns else 'NA'
    else:  # spatial
        para_col = 'p'
        x_col = df['p']
        dim = options.dim
        tau = options.tau

    if options.data_type == 'cycles':
        data_cols = ['no_cycles', 'one_cycle', 'many_cycles']
    elif options.data_type == 'components':
        data_cols = ['number_of_components']
    elif options.data_type == 'degrees':
        data_cols = ['degree_1', 'degree_2', 'degree_3']
    else:
        raise ValueError(
            "Invalid data type. Choose 'cycles', 'components', or 'degrees'.")

    # Prepare the output file name
    if options.model == 'aspatial':
        output_file = f"{options.model}_a1_{a1}_a2_{a2}_{options.data_type}.out"
    else:
        output_file = f"{options.model}_dim_{dim}_tau_{tau}_{options.data_type}.out"
    output_dir = save_dir(options.model, a1, a2,
                          dim=options.dim, tau=options.tau)
    output_path = os.path.join(output_dir, output_file)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write the data to the output file
    with open(output_path, 'w') as f:
        for index, row in df.iterrows():
            x_value = x_col.iloc[index] if isinstance(
                x_col, pd.Series) else x_value
            para_value = row[para_col]
            data_values = [row[col] for col in data_cols]
            line = f"{para_value} {x_value} " + \
                " ".join(map(str, data_values)) + "\n"
            f.write(line)

    print(f"Output saved to {output_path}")


def main(options: Options):
    process_csv(options)


if __name__ == "__main__":
    parser = ArgumentParser(add_dest_to_option_strings=False,
                            add_option_string_dash_variants=True)
    parser.add_arguments(Options, "options")
    args = parser.parse_args()

    main(args.options)
