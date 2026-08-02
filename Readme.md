# Master's Thesis Work

## Autoencoder based characterization and generation of power delivery on PCBs for application with relational databases

This repository contains the complete code, notebooks, trained models, datasets, and result files for my Master's thesis work.

### Project Overview

Power delivery networks are essential in high-speed electronic systems. Their impedance behavior strongly affects power integrity performance. Traditional EM simulations are accurate but computationally expensive when many PCB geometry and material variations need to be evaluated.

My work investigates whether machine learning models can learn compact representations of PCB impedance profiles and predict impedance behavior directly from geometry/material parameters. The final goal is not only forward prediction, but also inverse design: given a desired impedance behavior, find PCB geometry parameters that reproduce a similar response.

### Data Description

The dataset consists of EM-simulated PCB-based power delivery network samples.

Each sample contains:

Geometry parameters
Material parameters
Frequency-dependent impedance response
Touchstone-based Z-parameter data
Main focus on self-impedance Z11

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

