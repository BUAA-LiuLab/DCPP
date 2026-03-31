# DCPP

DCPP is a deep learning framework for predicting the co-assembly capability of drug molecule pairs. The repository combines multimodal molecular pair modeling, model interpretation, and drug-pathway association prediction in one research codebase.

At a high level, DCPP contains three parts:

- `Mol-CA`: multimodal co-assembly prediction for molecular pairs
- `Co-assembly interpretation`: GCN-based interpretation workflow for the molecular pair model
- `GraphDPA`: graph neural network prediction of drug-pathway associations

## Overview

This repository is organized as a research project rather than a packaged software release. The main components are:

- `DCPP-code/Mol-CA`
  Main code for molecular pair feature extraction, multimodal model training, and interpretation.
- `DCPP-code/GraphDPA-main`
  GraphDPA model code, example data, pretrained models, and example scripts for graph construction, training, and prediction.

## Repository Structure

```text
DCPP/
|-- README.md
`-- DCPP-code/
    |-- Mol-CA/
    |   |-- data/
    |   |-- feature/
    |   |-- interpretation/
    |   |-- lib/
    |   |-- model/
    |   |-- train/
    |   `-- main.py
    |-- GraphDPA-main/
    |   |-- codes/
    |   |-- example_base_data/
    |   |-- example_saved_data/
    |   |-- saved_models/
    |   |-- environment.yml
    |   |-- example-construct-graph.py
    |   |-- example-model-training.py
    |   `-- example-model-prediction.py
    
```

## Main Functions

### 1. Multimodal molecular pair prediction

`Mol-CA` is the core DCPP module for drug pair co-assembly prediction. Based on the current source code, it:

- reads molecular pair data from SMILES files
- extracts graph, sequence, and fingerprint features
- trains GCN-based and multimodal prediction models
- supports model interpretation through a dedicated GCN interpretation workflow

### 2. Drug-pathway association prediction

`GraphDPA-main` provides a graph neural network workflow to model drug-pathway associations. It includes:

- graph construction from example drug and pathway data
- model training on generated graph samples
- ensemble prediction with 15 pretrained GraphDPA models

This part is useful as supporting material, but it is not the main entry point for DCPP.

## Installation

### Option 1: GraphDPA environment

`GraphDPA-main` provides the most complete environment definition in this repository.

```bash
cd DCPP-code/GraphDPA-main
conda env create -f environment.yml
conda activate GraphDPA
```

The environment file includes core dependencies such as:

- Python 3.9
- PyTorch
- PyTorch Geometric
- RDKit
- scikit-learn
- pandas

### Option 2: Mol-CA environment

`Mol-CA` does not include a standalone environment file. Based on the current imports, the workflow depends on packages such as:

- Python 3.x
- torch
- torchvision
- torch-geometric
- torch-scatter
- torch-sparse
- rdkit
- numpy
- scikit-learn
- torchsampler
- pybel or OpenBabel bindings

In practice, the easiest starting point is to create the `GraphDPA` environment first, then install the additional packages required by `Mol-CA`.

## Program Entry Points

This repository does not provide a single unified command-line interface. The main entry points are the scripts below.

### A. Mol-CA main entry

```bash
cd DCPP-code/Mol-CA
python main.py
```

Current workflow in `main.py`:

1. read training and validation SMILES pairs
2. extract multimodal molecular features
3. train the multimodal model
4. optionally run interpretation-related functions

Main related files:

- `DCPP-code/Mol-CA/main.py`
- `DCPP-code/Mol-CA/feature/get_feature.py`
- `DCPP-code/Mol-CA/train/model_train.py`
- `DCPP-code/Mol-CA/interpretation/gcn_interpretation.py`

### B. GraphDPA workflow entry

```bash
cd DCPP-code/GraphDPA-main
python example-construct-graph.py
python example-model-training.py
python example-model-prediction.py
```

Suggested execution order:

1. `example-construct-graph.py`
   Build graph samples from the example drug and pathway data.
2. `example-model-training.py`
   Train a GraphDPA model on the generated graph dataset.
3. `example-model-prediction.py`
   Run prediction and evaluation with the pretrained ensemble models.

Main related files:

- `DCPP-code/GraphDPA-main/environment.yml`
- `DCPP-code/GraphDPA-main/example-construct-graph.py`
- `DCPP-code/GraphDPA-main/example-model-training.py`
- `DCPP-code/GraphDPA-main/example-model-prediction.py`


## Input Data

Important input files used by the main workflows include:

- `DCPP-code/Mol-CA/data/train_smiles.txt`
- `DCPP-code/Mol-CA/data/valid_smiles.txt`
- `DCPP-code/GraphDPA-main/example_base_data/drug2smile.pkl`
- `DCPP-code/GraphDPA-main/example_base_data/pathway2genes.pkl`
- `DCPP-code/GraphDPA-main/example_base_data/extra_val_set.csv`

## Notes

- This repository is best understood as research code and example workflows.
- `GraphDPA-main` is the easiest module to start with because it includes an environment file, example data, and pretrained models.
- `Mol-CA` is the main DCPP prediction module, but it may require local path adjustment and dependency preparation before execution.
- `CoAggregators-master` is best used as supporting reference code rather than the primary DCPP entry point.

## Recommended Start

If you are new to this repository, the recommended order is:

1. create the `GraphDPA` conda environment
2. run the `GraphDPA-main` example scripts
3. review the `Mol-CA` workflow and prepare its dependencies
4. use `CoAggregators-master` as a reference for related experiments
