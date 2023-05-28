
# Random Walks Go Cellular

Message Passing Graph Neural Networks are limited in their expressivity.
They can consider non-isomorphic graphs as equivalent and struggle to effectively capture the underlying relationships 
which may result in similar hidden representations for nodes in similar neighborhoods and therefore lead to poor expressive 
power of the network. 
Various solutions have been introduced in order to overcome this shortcoming of MP-GNNs. One approach amounts to including 
structural information in the initial node features, this is done by using positional encodings (PE) to augment the initial 
expressivity of the nodes.
An alternative approach involves integrating topological information from the underlying graph. This is achieved by 
considering the graphs' structure as explicit features. By considering any-dimensional cell (e.g. rings, edges), 
we create a more intricate neighborhood structure with the consequence of being able to distinguish more cases of graph 
isomorphism.

In this work, we explore the effect of more meaningful structural encodings in the MP-GNN GIN and Gated GCN models by 
combining the two aforementioned methods. First, we introduce a novel way to initialize the positional encodings that 
include more topological information. Second, we extend the original GIN and Gated GCN models with learnable positional 
encodings.

These extensions ultimately resulted in several key findings, namely that LSPE improves the performance of GIN architecture, 
something not previously shown in the literature. We were also able to show that using cellular random walks in addition to the 
original Gated GCN model improves the model performance on the ZINC dataset. We also found that when making use of a number of 
different adjacency matrices in the standard GIN and GatedGCN models, the inclusion of boundary/co-boundary adjacency matrices when
encoding the positional features of the graph performs best. Finally, we found that making use of the cellular random walk
with upper, lower and boundary adjacencies results in the best performance when combined with LSPE in the Gated GCN model.

## Setup


``` 
git clone https://github.com/lukegtc/Cellular_RW.git

cd Cellular_RW
``` 


For CPU:
``` 
conda create -n py3.9 python=3.9
conda install pytorch cpuonly -c pytorch
pip install pytorch-lightning  
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```
For GPU
```
conda create -n py3.9 python=3.9
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install pytorch-lightning  
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
  ```

## Experiments

Make sure you are in the root directory of the repository when running these scripts.

#### GIN

Basic GIN experiment with no positional embeddings.

``` Running GIN

python -m src.train.py

```

Basic GIN experiment with traditional Random Walk PE.

``` Running GIN with positional embeddings

python -m src.train.py --use_pe rw

```
GIN with LSPE
``` Running GIN with LSPE

python -m src.train.py --use_pe rw --learnable_pe True

```

To run the experiments for different traverse types, change the value of --traver_type to one of [boundary, upper_adj, lower_adj, upper_lower, upper_lower_boundary]. For instance, the code below is for the RW that includes all traverse types.

``` Running GIN with upper adjacency

python -m src.train.py --use_pe ccrw --traverse_type upper_lower_boundary

```

#### Gated GCN
The same commands apply to the Gated GCN model, but remember to change the model flag. The example below is the command to run the vanilla Gated GCN without any positional encodings.

``` Running GCN

python -m src.train.py --model gated_gcn

```
## Repository structure

[`Blogpost`](./blogpost.md) contains a thorough explanation of our work, with specifics of the experiments and results.

 [`Train`](./src/train.py) contains the training script for all the different models and configurations.
 
[`Models`](./src/models) contains the implementation of GIN and GatedGCN.
  
[`PE`](./src/topology/pe.py) contains the Random Walk implementation, both traditional and cellular.