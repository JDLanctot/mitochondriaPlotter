# Mitochondria Plotter

Mitochondria Plotter is a Python-based tool for analyzing and visualizing mitochondrial networks. It provides functionality for both aspatial and spatial models, offering various analysis and visualization options.

## Project Overview
This tool generates network visualizations of mitochondrial connections based on input data. It is designed to help visualize and analyze how mitochondria are connected within a given dataset, supporting both aspatial and spatial models.

## Features

- Support for aspatial and spatial mitochondrial network models
- Network analysis including degree distribution, component sizes, and cycle detection
- Visualization of mitochondrial networks and lattices
- Batch processing and analysis of multiple samples
- Data conversion utilities
- Generation and visualization of lattice models

## Installation

### Prerequisites
- Python 3.x
- Required Python libraries are listed in the `environment.devenv.yml`

To set up the package as a conda environment:
```bash
conda env create --file=environment.devenv.yml
pip install -e .
```

### Getting Started
Clone this repository to your local machine:
```bash
git clone https://github.com/JDLanctot/mitochondriaPlotter.git
cd mitochondriaPlotter
```

## Usage

The project includes several example scripts that demonstrate various functionalities:

### Analysis

The analysis scripts process the input data to generate statistical information about the mitochondrial networks. These scripts create CSV files containing various network metrics, which are then used to create plots for figures in the associated research paper. Specifically, the analyses for figures 3b, 3c, 4b, 4c, 4d, 5c, 5d, and 5e are generated using the statistics yielded by these CSV files, with particular parameter inputs for the models in question.

1. Single Network Analysis:
   ```
   python examples/analyze.py -f <file_name> -l <load_name> -o <output_dir> [-s <seed>]
   ```

2. Batch Network Analysis:
   ```
   python examples/analyze_many.py -f <file_name> -o <output_dir> -m <model> -N <num_mito> -a <a1> <a2> -t <tau> -d <dimension> [-s <seed>]
   ```

### Visualization

The visualization scripts generate graphical representations of the mitochondrial networks based on the input data and analysis results. These scripts are used to create network visualizations and statistical plots. The networks shown in figures 5a and 5b are generated using these visualization scripts. Note that the functions `plot_particular_network_from_edgelist_1` through `plot_particular_network_from_edgelist_6` are modified versions of the `plot_particular_network_from_edgelist` function. These modifications were made to manually adjust the layout of connected components for specific networks, bringing them closer together for clearer visualization.

1. Single Network Plot:
   ```
   python examples/plot.py -f <file_name> -o <output_dir> -m <model> -l <load_name> [-s <seed>]
   ```

2. Batch Network Plotting:
   ```
   python examples/plot_many.py -f <file_name> -o <output_dir> -m <model> -N <num_mito> -a <a1> <a2> -t <tau> -d <dimension> -S <num_samples> [-s <seed>]
   ```

3. Specific Network Plot:
   ```
   python examples/plot_particular.py -f <file_name> -o <output_dir> -m <model> -N <num_mito> -a <a1> <a2> -t <tau> -d <dimension> -r <run> -b <b1> [-s <seed>]
   ```

4. Generate and Plot Lattice:
   ```
   python examples/plot.py -f <file_name> -o <output_dir> -m gen_lattice -n <rows> <columns> -p <percolation_threshold> -k <lattice_degree> [-s <seed>]
   ```

### Data Conversion

Convert analysis results to text format:
```
python examples/to_text.py -m <model> -t <data_type> -a <a1> <a2> -T <tau> -d <dimension> [-i <input_file>]
```

The data conversion utility is designed to transform the output of the analysis scripts into a format that can be easily used by other scripts in the pipeline, particularly those written in other languages like Fortran. This process involves:

1. Reading the CSV file produced by the analysis scripts.
2. Extracting relevant columns based on the specified data type (cycles, components, or degrees).
3. Formatting the data into a simple text-based format.
4. Writing the formatted data to a new text file.

The resulting text file contains space-separated values, with each line representing a different data point. The columns typically include:

1. The parameter value (b1 for aspatial model, p for spatial model)
2. A calculated x-value
3. The relevant data values (e.g., number of cycles, component sizes, or degree distributions)

This format allows for easy parsing and processing by other programs, facilitating integration with existing scientific workflows and analysis pipelines.

## Models

The project supports two main types of models:

1. Aspatial Model: Defined by parameters a1, a2, and b1.
2. Spatial Model: Defined by dimension (2d or quasi1d) and tau.

## Input Data Format

The input data for both aspatial and spatial models should be provided as a text file (.txt or .dat) containing an edgelist. Each line in the file represents a link in the network, with two columns:

1. The first column is the number of the node from which the link originates.
2. The second column is the number of the node to which the link connects.

Example:
```
1 2
3 4
5 6
7 8
...
97 98
99 100
1 80
2 66
...
```

This format represents a network where each line defines a connection between two nodes. For instance, "1 2" means there's a link from node 1 to node 2.

## Output

The scripts generate various outputs including:
- Network visualizations
- Probability distribution plots
- CSV files with analysis results
- Text files with formatted data for further processing

## Dependencies

- numpy
- networkx
- matplotlib
- pandas
- scipy
- simple_parsing

## Contributing

We welcome contributions to this project. Please feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
