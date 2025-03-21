# ON-Traffic: An Operator Learning Framework for Online Traffic Flow Estimation and Uncertainty Quantification from Lagrangian Sensors
This repository contains the source code for the paper [ON-Traffic: An Operator Learning Framework for Online Traffic Flow Estimation and Uncertainty Quantification from Lagrangian Sensors](https://arxiv.org/abs/2503.14053)

# Requirements
`environment_dataset.yml` contains the Anaconda environment file that can be used to create the dataset. The environment can be made using `conda env create -f environment_dataset.yml`

# Dataset
We generate both our datasets based on code from [mBarreau](https://github.com/mBarreau/TrafficReconstructionIdentification)
## Numerical Dataset

- First run `godunov_solver/main_godunov_multi.py`, which will in a multiprocessing manner generate `num_simulations` simulations with different initial conditions and boundary control signals.  
- Then run `godunov_solver/combine_simulations.py`, which will group all individually saved simulations into a `.npz` file.
- Finally run the notebook `godunov_solver/process_dataset.ipynb`, which will based on given settings process the dataset making it ready for training.

## SUMO Dataset
- First run `sumo_simulation/simulation_multiprocessing.py`, which will in a multiprocessing manner run SUMO simulations with different initial conditions and boundary control signals. 
- Then run `sumo_simulation/process_dataset_sumo.ipynb`, which will process the dataset making it ready for training.

# training
The given environment file will not install torch with cuda support. To run the training notebook `training.ipynb`, [torch](https://pytorch.org/get-started/locally/) has to be installed with GPU support. 
