# Model-based Deep Learning for Joint RIS Phase Shift Compression and WMMSE Beamforming
Source code for the research paper:
[Model-based Deep Learning for Joint RIS Phase Shift Compression and WMMSE Beamforming](https://doi.org/10.1109/LWC.2026.3683016),
link to preprint: [arxiv](https://arxiv.org/abs/2510.05438).


### Required
Run the following commands in the terminal to install Python libraries to the local Python environment.
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy
pip install pandas
pip install tabulate
pip install tqdm
pip install matplotlib
```

To install MATLAB CVX see [https://cvxr.com/cvx/doc/install.html](https://cvxr.com/cvx/doc/install.html) for instructions (not required but may be useful).

## Description
Descriptions of the Python projects, directories, and content included in this repository.

### MATLAB - system model and RIS beamforming/phaseshift optimization
MATLAB (version 2022a) to generate a dataset for a simulated RIS communication system model.
* [generateHDRISData.m](MATLAB/generateHDRISData.m)
  * run this MATLAB script to generate data for a Half-Duplex RIS model
  * generates the channels / channel errors, solves the beamformer / phase shifts with WMMSE-PI algorithm, then saves the generated data as .csv files located in the [datasets/](MATLAB/datasets/) directory
* [src/](MATLAB/src/)
  * directory containing MATLAB scripts and functions to generate RIS data
  * to change the parameters of the system model, change the contents of: [src/systemModelParameters.m](MATLAB/src/systemModelParameters.m)
* [datasets/](MATLAB/datasets/)
  * directory to store the generated datasets
  * for Python scripts use the .csv files from this directory to load the datasets into a numpy array
  

### Python - deep learning architecture and benchmarks
Python (version 3.11) to run deep learning experiments. 
* [mumiso_autoencoder_quantization.py](mumiso_autoencoder_quantization.py) 
  * written in a single python file for the convenience of locating all models and relevant source code (i.e. proposed AQE-WMMSE, benchmarks, individual deep learning modules, etc.)
  * the script will not run unless modified appropriately based on how the dataset was generated

## Instructions to run project
1. Generate dataset of channels and optimal RIS phase shifts / beamforming
   1. Choose the number of dataset samples by changing the "mc_runs" variable in [src/systemModelParameters.m](MATLAB/src/systemModelParameters.m), along with any other system model parameters.
   2. (Optional) It is recommended to generate the dataset samples in parallel with multiple compute nodes / supercomputer to reduce the computation time of large dataset sizes.
   3. run [generateHDRISData.m](MATLAB/generateHDRISData.m) in MATLAB and the data to be used for the deep learning experiments will be generated and saved to [datasets/](MATLAB/datasets/).
2. Run deep learning experiments
   1. In [mumiso_autoencoder_quantization.py](mumiso_autoencoder_quantization.py), set the dataset/results directory and load the number of dataset directories correspondingly with the "num_dirs" variable (i.e. num_dirs > 1 if the data samples were generated from parallel processes). 
   2. Run [mumiso_autoencoder_quantization.py](mumiso_autoencoder_quantization.py) in Python and the log files of the experiments will be saved to the results directory specified in the script.
