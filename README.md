# pyMRzeroXCAT
Python version of the MRXCAT MATLAB repository.

# Table of Contents

- [Install](#install)
- [Run CINE and PERF models](#run-cine-and-perf-models)
- [Build MRzero Phantom](#build-mrzero-phantom)
- [Example in Jupyter Notebook](#example-in-jupyter-notebook)
- [References](#references)

## Install

1. Download the repository:
   ```bash
   # clone project
   git clone https://github.com/CyprienBouton/pymrzeroxcat
   cd pyMRzeroXCAT
   ```
2. Create a virtual environment (Conda is strongly recommended):
   ```bash
   # create conda environment
   conda create -n mrxcat python=3.10
   conda activate mrxcat
   ```
4. Install the project in editable mode and its dependencies:
   ```bash
   pip install .
   ```

Several new commands will be added to the virtual environment once the installation is completed.
These commands all start with `mrxcat_`.

## Build MRzero Phantom

1. Ask cine and perfusion dataset .zip file from https://www.biomed.ee.ethz.ch/mrxcat,
	After downloading, extract the contents and add them to this repository.

2. Adapt the MRXCAT tissues parameters in [pymrzeroxcat/tissues.json](pymrzeroxcat/tissues.json) to your needs. 
	For a first try, go with the predefined parameters.

3. Create a MRzero [Phantom](https://mrzero-core.readthedocs.io/en/latest/api/phantom.html#voxel-grid-phantom) 
with the command `mrxcat_build_static`

4. Select a XCAT .bin file to build the MRXCAT phantom.

5. Once the simulation is done, you get a *.npz MRzero parameter file to initialize a **VoxelGridPhantom**

![MRzero Phantom](visuals/MRXCAT_Phantom.png)

## Example in Jupyter Notebook
A basic example of a Flash 2D sequence simulation using the MRXCAT phantom in MRzero framework can be found [here](examples/simulate_MRXCAT_Flash2D.ipynb).

## References
If you use this code for research, please cite the following paper 
[Wissmann L, Santelli C, Segars WP, Kozerke S. MRXCAT: Realistic Numerical Phantoms for Cardiovascular Magnetic Resonance. J Cardiovasc Magn Reson 2014;16:63.](https://pubmed.ncbi.nlm.nih.gov/25204441/)