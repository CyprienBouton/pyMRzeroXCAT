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
   pip install -e .
   ```

Several new commands will be added to the virtual environment once the installation is completed.
These commands all start with `mrxcat_`.

## Run CINE and PERRF models 
	
1. 	Download the MRXCAT .zip file from https://www.biomed.ee.ethz.ch/mrxcat,
	unpack it and add the folder to your Matlab path (only the MRXCAT 
	folder, not the @MRXCAT_CMR_PERF and @MRXCAT_CMR_CINE folders).
2.	Download the XCAT perfusion and/or cine example .zip files and unpack
	to any working directory. 
3. 	Adapt the MRXCAT parameters in CINEpar.m in @MRXCAT_CMR_CINE or 
	PERFpar.m in @MRXCAT_CMR_PERF to your needs. For a first try, go
	with the predefined parameters.
4.	Start cine or perfusion MRXCAT by typing
	`mrxcat_cine` or `mrxcat_perf` into the command line.
5. 	Select the first XCAT .bin file in the file selection dialog
	(cine_act_1.bin for cine, perfusion_act_1.bin for perfusion). 
	Once the simulation is done, you get the following files:
	*.cpx		MRXCAT phantom data
	*.msk		XCAT mask data
	*.sen		MRXCAT coil sensitivity maps
	*.noi		MRXCAT noise only
	*_par.mat	MRXCAT parameters
6.	To display the produced phantom, run mrxcat_display; and select
	the *.cpx file in the file selection dialog.