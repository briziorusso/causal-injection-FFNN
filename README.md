# Causal Discovery and Injection for Feed-Forward Neural Networks
 This repository provides the code for the paper on Causal Discovery and Injection. There are three main python scripts:
- `net_inject.py` defines the InjectedNet class: this is used to apply the causal masking to the weights of the input layer of the neural networks, enforcing the use of *only* the direct causal relationships provided in the input graph/DAG.

- `main_realdata.py` reproduces the experiments on synthetic data. The script iterates runs over the three main dimensions varying for the generated DAGs: number  of nodes in DAG (|V| in {10, 20, 50}), number of edges per node (|E| = |V| x e, where e in {1,2,5}), and data size (N=|V| x s, where s in {50,100,200,300,500}). 

- `main_synth.py` reproduces the experiments on real data. The experiments on real data are on FICO HELOC, Adult, Boston and California Housing datasets. The script iterates runs over two dimensions: the threshold tau (to retrieve computed DAGs) and the sample size N.

In the [Notebooks](Notebooks) folder there are two jupyter notebooks (one for the synthetic data `Main_Synth.ipynb` and one for real data `Main_Real.ipynb`) collecting the results and plotting the charts presented in the paper. Each scenario shown in the notebooks also reports the command used to run it (as per below example).

### Example use
From terminal at the root folder, run:
```
python main_synth.py --version="r_ex_1b_k20_l3_h2_n0" --branchf=1 --known_p=0.2 --noise_p=0.0 --hidden_l=3 --hidden_n_p=2
```
