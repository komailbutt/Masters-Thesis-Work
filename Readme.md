# Master's Thesis Work

## Autoencoder based characterization and generation of power delivery on PCBs for application with relational databases

This repository contains the complete code, notebooks, trained models, datasets, and result files for my Master's thesis work.

### Project Overview

Power delivery networks are essential in high-speed electronic systems. Their impedance behavior strongly affects power integrity performance. Traditional EM simulations are accurate but computationally expensive when many PCB geometry and material variations need to be evaluated.

My work investigates whether machine learning models can learn compact representations of PCB impedance profiles and predict impedance behavior directly from geometry/material parameters. The final goal is not only forward prediction, but also inverse design: given a desired impedance behavior, find PCB geometry parameters that reproduce a similar response.

### Data Description

The dataset consists of EM-simulated PCB-based power delivery network samples. The dataset (4-Layer PCB based PDN with Two Via Arrays - Central Power Rail) can be requested at https://www.tet.tuhh.de/en/si-pi-database/

Each sample contains:

- Geometry parameters & Material parameters.
- Touchstone-based Z-parameter data.

The main focus is on self-impedance Z11 in this work.

The final selected impedance representation is:

"ln_phase" with two channels:

- channel 0: ln(|Z11|)
- channel 1: phase(Z11)

The final selected latent representation has:

latent_dim = 60

The main dataset contains:

- 20,000 PCB samples
- 17 geometry/material parameters
- 334 frequency points
- 60 latent variables per sample

### Project Objective

The main objective of this thesis is to develop a machine learning framework that can:

- Compress high-dimensional impedance curves into compact latent vectors.
- Reconstruct impedance curves from latent vectors.
- Predict latent impedance representations from PCB geometry/material parameters.
- Analyze which physical parameters influence the impedance behavior.
- Perform inverse design by optimizing geometry parameters toward a desired target impedance.
- Validate generated designs using external EM simulation.

### 1) Stage-1 CNN Autoencoder

#### Main Notebook

Notebooks/1_CNN_AE_Stage_1.ipynb

#### Selected Model

- Model: CNN Autoencoder
- Representation: ln_phase
- Latent dimension: 60
- Loss: MSE
- Training subset: 14,000 samples
- Learning rate: 0.0003
- Epochs: 500

#### Main Checkpoint

checkpoints/cnn_ae_ln_phase_mse_N14000_ld60_lr0.0003_ep500.pt

#### Supporting Scripts

- Sources/dataset.py
- Sources/models.py
- Sources/trainer.py
- Sources/metrics.py
- Sources/utils.py

### 2) LSTM Autoencoders

#### Main Notebooks

- Notebooks/2_LSTM_AE_Stage_1.ipynb
- Notebooks/2a_LSTM_AE_Stage_1_derivativeloss.ipynb

#### Supporting Notebooks

- Sources/dataset_lstm.py
- Sources/models.py
- Sources/trainer_lstm.py
- Sources/trainer_lstm_roi
- Sources/trainer_lstm_derivative_loss.py


#### Main Observation

The LSTM-AE captured global impedance trends but struggled with narrow local resonance spikes. Increasing latent dimension and adding derivative/ROI losses provided limited improvement. The CNN-AE was therefore selected as the main Stage-1 model.

### 3) Latent Database Creation

#### Objective

Build a complete latent database containing following for Stage 2 model training:

- simu_index
- 17 geometry/material parameters
- z_0 ... z_59

#### Main Notebook

Notebooks/Latent_Analysis/build_latent_database_cnn_ln_phase_ld60.ipynb

#### Output Folder

Notebooks/Latent_Analysis/build_latent_database_cnn_ln_phase_ld60.ipynb

#### Main Output Files:

- latent_database_cnn_ln_phase_ld60_full20k.csv
- latent_database_cnn_ln_phase_ld60_full20k.npz

### 4) Latent-Space Analysis

#### Overview

The learned 60D latent space is analyzed to understand how PCB impedance behavior is organized. PCA and geometry-colored plots are used to inspect relationships between physical parameters and latent representations.

#### Objectives

Analyze the latent space to understand:

- Distribution of latent vectors
- PCA structure
- Relation between board dimensions and latent position
- Effect of material parameters
- Resonance-related trends
- Latent-space nearest neighbors

#### Main Notebooks

Analyze the latent space to understand:

- Notebooks/Latent_Analysis/cnn_ln_phase_latent_analysis.ipynb
- Notebooks/Latent_Analysis/pca_analysis.ipynb
- Notebooks/Latent_Analysis/relate_geometry&latent_space.ipynb
- Notebooks/Latent_Analysis/use_latent_database_cnn_ln_phase_ld60.ipynb
- Notebooks/Latent_Analysis/python_tool_resonance_freq_finder.ipynb

#### Output Folders

- Results/cnn_ln_phase_ld60_latent_analysis/
- Results/pca_analysis_latent_space/
- Results/relate_geometry_latent_space/
- Results/use_latent_database_cnn_ln_phase_ld60/

### 5a) Stage-2 Geometry-to-Latent Surrogate Model

#### Overview

Stage-1 provides a compact 60D latent representation of each impedance curve. Stage-2 learns to predict this latent vector directly from PCB geometry/material parameters. This enables fast impedance prediction without running EM simulations.

#### Data Description

##### Input
17 geometry/material parameters
##### Output
60D latent vector

The predicted latent vector is decoded using the frozen CNN-AE decoder to reconstruct: Z11(f)

#### Main Notebook

Notebooks/1_CNN_AE_Stage_2.ipynb

#### Main Checkpoint

checkpoints/stage2/stage2_fnn_ln_phase_ld60_gelu_huber_adam_ep1500_lr0.0002_do0.06_hd(512, 512, 512, 512, 512, 256, 256).pt

#### Custom Feature Experiment (Physically Engineered Input Features)

Although the baseline Stage-2 model already achieved good performance, additional physically motivated features were investigated to determine whether domain knowledge could further improve geometry-to-latent prediction.

##### One Example
The original width parameters XWIDTH and YWIDTH were replaced by Area.

#### Main Notebook
Notebooks/Custom_Features_Stage_2_ANN_CNN_AE.ipynb

#### Output Folder

Results/Custom_Features_Stage_2_FNN_CNN_AE

### 5b) Stage-2 Hyperparameter Tuning

#### Overview

To systematically identify a high-performing model configuration, automated hyperparameter optimization was performed using the Optuna framework.

#### Main Notebook

Notebooks/Stage2_Optuna_FNN_Hyperparameter_Tuning.ipynb

#### Supporting Script

Sources/stage2_optuna.py

### 5c) SHAP Feature Importance

#### Overview

After training the Stage-2 FNN, SHAP analysis is used to understand which PCB geometry/material parameters most influence the predicted latent representation.

#### Main Notebook

Notebooks/Stage2_SHAP_FNN.ipynb

#### Output Folder

Results/stage2_shap_fnn_ln_phase_ld60/

### 6) Inverse Design Problem

#### Overview

Given the target impedance profile as an input, ask the model which geometry parameters could produce this target impedance profile.
Two starting points are considered, meaning their geometry parameters are optimized to get the target impedance profile.

- Nearest- Neighbor Start
- Random Start

#### Nearest-Neighbor Start Inverse Design

##### Main Notebook
Notebooks/Inverse_Design_NN_Start_FNN_CNN_AE.ipynb

##### Output Folder
Results/inverse_design_nn_fnn_cnn_ae/

Each run is saved as:

- Results/inverse_design_nn_fnn_cnn_ae/target_XXXXXX_start_XXXXXX/
  
Saved files include:

- geometry_latent_loss.csv
- geometry_impedance_loss.csv
- metrics_latent_loss.json
- metrics_impedance_loss.json
- loss_comparison.csv
- latent_mse_loss.png
- latent_l2_distance.png
- impedance_loss_total.png
- mag_db_mse_loss.png
- phase_deg_mse_loss.png
- magnitude_latent_loss.png
- phase_latent_loss.png
- latent_vs_impedance_loss_magnitude.png
- latent_vs_impedance_loss_phase.png

##### Optimized Geometry Parameter file for EM Validation

Results/inverse_design_nn_fnn_cnn_ae/geometry_impedance_loss_files

#### Random Start Inverse Design

##### Main Notebook
Notebooks/Inverse_Design_Random_Start_FNN_CNN_AE.ipynb

##### Output Folder
Results/inverse_design_random_fnn_cnn_ae/

Each run is saved as:

- Results/inverse_design_random_fnn_cnn_ae/target_XXXXXX_start_XXXXXX/
  
Saved files include:

- geometry_latent_loss.csv
- geometry_impedance_loss.csv
- metrics_latent_loss.json
- metrics_impedance_loss.json
- loss_comparison.csv
- latent_mse_loss.png
- latent_l2_distance.png
- impedance_loss_total.png
- mag_db_mse_loss.png
- phase_deg_mse_loss.png
- magnitude_latent_loss.png
- phase_latent_loss.png
- latent_vs_impedance_loss_magnitude.png
- latent_vs_impedance_loss_phase.png

##### Optimized Geometry Parameter file for EM Validation

Results/inverse_design_random_fnn_cnn_ae/geometry_impedance_loss_files

#### EM Validations of Nearest - Neighbor & Random Start

##### Main Notebooks

- Notebooks/EM_Validation_NN.ipynb
- Notebooks/EM_Validation_Random.ipynb

##### Output Folders

- Results/EM_Validation_NN/

- Results/EM_Validation_Random/



