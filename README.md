# Deep_Learning_2_project

## Setup

``` Installing and configuring repo
git clone https://github.com/lukegtc/Deep_Learning_2_project.git
cd Deep_Learning_2_project
conda env create -f lspe_lisa.yml
conda activate lspe_lisa.yml
```

## Experiments
To run the experiments for basic MP-GNN with PE:
```
python src/train.py 
```

For the MP-GNN-LSPE use:
```
python src/train_LSPE.py 
```

## Repository structure
In the [`blogpost`](./blogpost.md), background and other details regarding the set up of the experiments and the models can be found.
[`Model`](./model.py) contains the implementation of a basic MP-GNN, MP-GNN with positional encoding and MP-GNN with learnable positional encoding.
[`Transform`](./transform.py) contains the modified Random Walk with the inclusion of cycles.