# Deep_Learning_2_project

## Setup
By following the runnable modeules within this README file, the code can easily be ran on a local machine.
``` Installing and configuring repo
git clone https://github.com/lukegtc/Deep_Learning_2_project.git
cd Deep_Learning_2_project
conda env create -f lspe_lisa.yml
conda activate lspe_lisa.yml
```

## Experiments
Make sure you are in the root directory of the repository when running these scripts.
#### GIN
Basic GIN experiment with no positional embeddings.
``` Running GIN
python -m src.train_GIN
```
Basic GIN experiment with positional embeddings.
``` Running GIN with positional embeddings
python -m src.train --use_pe rw
```
All variations of the cellular complex random walk experiment with different traverse types can be ran through the use of
the 
#### Gated GCN

## Repository structure
In the [`blogpost`](./blogpost.md), background and other details regarding the set up of the experiments and the models can be found.

[`Model`](src/unused/mpgnn.py) contains the implementation of a basic MP-GNN, MP-GNN with positional encoding and MP-GNN with learnable positional encoding.

The [`scripts`](./src/scripts) contains the training scripts for the different models and the modified Random Walk initialization with the inclusion of cycles.
