# pyMRXCAT
Python version of the MRXCAT MATLAB repository.

## Install

1. Download the repository:
   ```bash
   # clone project
   git clone https://github.com/CyprienBouton/pyMRXCAT
   cd pyMRXCAT
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

## Run CINE and PERF models 
	
1. 	Ask cine and perfusion dataset .zip file from https://www.biomed.ee.ethz.ch/mrxcat,
	After downloading, extract the contents and add them to this repository.

2. 	Adapt the MRXCAT parameters in pymrxcat/MRXCAT_CMR_CINE/mrxcat_cmr_cine.py and pymrxcat/MRXCAT_CMR_PERF/mrxcat_cmr_perf.py to your needs. 
	For a first try, go with the predefined parameters.

4.	Start cine or perfusion MRXCAT by typing
	`mrxcat_cine` or `mrxcat_perf` into the command line.
    
5. 	Select the first XCAT .bin file from the cine and perfusion datasets
	(cine_act_1.bin for cine, perfusion_act_1.bin for perfusion). 
	Once the simulation is done, you get the following files:
	*.cpx		MRXCAT phantom data
	*.msk		XCAT mask data
	*.sen		MRXCAT coil sensitivity maps
	*.noi		MRXCAT noise only
	*_par.mat	MRXCAT parameters
    
6.	To display the produced phantom, run `mrxcat_display`; and select
	the *.cpx file in the file selection dialog.

![Myocardial perfusion](images/algorithm_animation.gif)
![Cardiac cine](images/algorithm_animation.gif)