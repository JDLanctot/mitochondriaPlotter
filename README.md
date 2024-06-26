# Network Visualization of Mitochondria Connections

## Project Overview
This tool generates a network visualization of mitochondrial connections based on data provided in a .mat file. It is designed to help visualize how each mitochondrion is connected within a given dataset.

## Getting Started

### Prerequisites
- Python 3.x
- Required Python libraries are listed in the `environment.devenv.yml`. 

You can run the following to setup the the package as a conda environment:
```bash
conda env create --file=environment.devenv.yml
pip install -e .
```

### Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/JDLanctot/mitochondriaPlotter.git
cd mitochondriaPlotter
```

### Running the Program
To run the program, navigate to the project directory in your command line interface and execute the following command:

```bash
python examples\plot.py -f image -l edge_list -t gen_lattice -o C:\Users\Jordi\PycharmProjects\mitochondriaPlotter\examples -n 7 7 -p 0.4 -k 4
```

Here you should replace `C:\Users\Jordi\PycharmProjects` with the correction to the path where your repo is located.

Here are the required flags:

_save file name_

file_name: str = field(alias='-f', required=True)


_data file name_

load_name: str = field(alias='-l', required=True)


_where to save the plot_

output_file: str = field(alias='-o', required=True)


_whether it is a graph or lattice or to gen_lattice (generate a lattice)_

type: str = field(alias='-t', required=False, default='graph')


_random seed_

seed: int = field(alias='-s', default=None, required=False)


_Size of a lattice_

lattice_size: Tuple[int,int] = field(alias='-n', required=False, default=(5,5))


_percolation threshold_

p: float = field(alias='-p', required=False, default=0.4)


_lattice degree_

k: int = field(alias='-k', required=False, default=4)


## Data Format
The input data should be a .mat file containing an edgelist. The edgelist is structured as a 2xNx2 matrix, where:
- The first dimension represents outbound and inbound nodes.
- N is the number of connections.
- The second dimension of each N represents:
  - The mitochondria number.
  - A 1 or 2 indicating one of the two ends of the mitochondria.

Each connection links one end of a mitochondrion to another, potentially connecting different mitochondria. It is important to note that connections between the two ends of the same mitochondrion are not included in this list.

## Example
Provide a brief example or a screenshot demonstrating the output of your script.

## Contributing
We welcome contributions to this project. Please feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
