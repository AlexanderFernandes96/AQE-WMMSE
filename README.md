# A-Deep-Learning-Architecture-for-Joint-RIS-Phase-Shift-Compression-and-WMMSE-Beamforming
Source code for the research paper:
*A Deep Learning Architecture for Joint RIS Phase Shift Compression and WMMSE Beamforming.*


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

To install MATLAB CVX see [https://cvxr.com/cvx/doc/install.html](https://cvxr.com/cvx/doc/install.html) for instructions.

## Description
Descriptions of the Python projects, directories, and content included in this repository.

### MATLAB
MATLAB (version 2022a) is used to generate a dataset for a simulated RIS communication system model, i.e. channel matrices, pilots, receive signal, etc.
* [generateHDRISData.m](MATLAB/generateHDRISData.m)
  * Run this MATLAB script to generate data for a Half-Duplex RIS model
* [src/](MATLAB/src/)
  * directory containing MATLAB scripts and functions to generate RIS data
  * To change the parameters of the system model change contents of: [src/systemModelParameters.m](MATLAB/src/systemModelParameters.m)
* [datasets/](MATLAB/datasets/)
  * directory to store the generated datasets.
  * for Python scripts use the .csv files from this directory to easily load the datasets into a numpy array.
  

### Multi-User MISO Neural Network AutoEncoder with Quantization
Auto-Quantization-Encoder WMMSE Beamforming Updater Deep Learning architecture and benchmarks. 
* [mumiso_autoencoder_quantization.py](mumiso_autoencoder_quantization.py)

## Run project
1. Generate dataset of channels and optimal RIS phase shifts / beamforming
   1. Choose the number of dataset samples by changing the "mc_runs" variable in [src/systemModelParameters.m](MATLAB/src/systemModelParameters.m), along with any other system model parameters.
   2. (Optional) It is recommended to generate the dataset samples in parallel with multiple compute nodes / supercomputer due to computation time for computing optimal RIS phase-shift/beamforming.
   3. run [generateHDRISData.m](MATLAB/generateHDRISData.m) in MATLAB and the corresponding .csv files will be generated.
2. Run the deep learning project
   1. In [mumiso_autoencoder_quantization.py](mumiso_autoencoder_quantization.py), set the dataset/results directory to the generated dataset from part 1 and load the number of dataset directories correspondingly with the "num_dirs" variable. 
   2. Run [mumiso_autoencoder_quantization.py](mumiso_autoencoder_quantization.py) in Python 