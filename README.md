# Causal Discovery and Knowledge Injection for Contestable Neural Networks
 This repository provides the code for the paper on Contestable Neural Networks. 
 
 There are three main python scripts:
- `net_inject.py` defines the InjectedNet class: this is used to apply the causal masking to the weights of the input layer of the neural network, enforcing the use of *only* the direct causal relationships provided in the input DAG/graph.

- `main_synth.py` reproduces the experiments on synthetic data. The script iterates over the three main dimensions varying for the generated DAGs: number  of nodes in DAG (|V| in {10, 20, 50}), number of edges (|E| = |V| x e, where e in {1,2,5}), and data size (N=|V| x s, where s in {50,100,200,300,500}). 

- `main_realdata.py` reproduces the experiments on real data. The experiments on real data are on FICO HELOC, Adult, Boston and California Housing datasets. The script iterates over two dimensions: the threshold tau (to retrieve computed DAGs) and the sample size N. All the data is downlowded within the `utils.py` script apart from HELOC (this is instead provided in the fico folder within data directory, both [pre-processed](data/fico/WOE_Rud_data.csv) and [original](data/fico/heloc_dataset_v1.csv)). 

### Example usage
To fit an *injected* network, from python at the root folder, run:
```
# Imports
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from net_inject import InjectedNet
from utils import load_or_gen_data, load_adj_mat

# Load toy data
G, df = load_or_gen_data(name='toy', N=5000, csv='data/synth_nonlinear.csv')
loaded_adj = load_adj_mat(df, DAG_inject='toy', version='test')
X_DAG = df.to_numpy()
y_DAG = np.expand_dims(X_DAG[:,0], -1)
train_X, test_X, train_y, test_y = train_test_split(X_DAG, y_DAG, test_size=0.2, random_state=0)

# Fit toy model
net = InjectedNet(num_inputs = X_DAG.shape[1])
net.fit(X=train_X, y=train_y, num_nodes=X_DAG.shape[1], X_val=test_X, y_val=test_y, inject=True, verbose=True)
```

To rerun the synthetic experiments, from a terminal at the root folder, run:
```
python main_synth.py --version="r_ex_1b_k20_l3_h2_n0" --branchf=1 --known_p=0.2 --noise_p=0.0 --hidden_l=3 --hidden_n_p=2
```

`main_realdata.py` has the same usage. Refer to the scripts for descriptions of the meaning and usage of the parameters. Beware, an end-to-end run of this script takes several hours if run on single cpu/gpu.

### Environment
The code was tested with Python 3.6.9 and 3.9.9. `requirements.txt` provides the necessary packages. Run `pip install -r requirements.txt` from a terminal at the root folder to install all packages in your virtual environment.

### Reference
If you are using this code, please cite our paper
```
@inproceedings{Russo2023contestable,
  author       = {Fabrizio Russo and
                  Francesca Toni},
  title        = {Causal Discovery and Knowledge Injection for Contestable Neural Networks},
  booktitle    = {{ECAI} 2023 - 26th European Conference on Artificial Intelligence,
                  September 30 - October 4, 2023, Krak{\'{o}}w, Poland - Including
                  12th Conference on Prestigious Applications of Intelligent Systems
                  {(PAIS} 2023)},
  series       = {Frontiers in Artificial Intelligence and Applications},
  volume       = {372},
  pages        = {2025--2032},
  publisher    = {{IOS} Press},
  year         = {2023},
  url          = {https://doi.org/10.3233/FAIA230495},
  doi          = {10.3233/FAIA230495},
}
```
