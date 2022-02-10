# Causal Discovery and Injection for Feed-Forward Neural Networks
 This repository provides the code for the paper on Causal Discovery and Injection. There are three main python scripts:
- `net_inject.py` defines the InjectedNet class: this is used to apply the causal masking to the weights of the input layer of the neural network, enforcing the use of *only* the direct causal relationships provided in the input DAG/graph.

- `main_synth.py` reproduces the experiments on synthetic data. The script iterates over the three main dimensions varying for the generated DAGs: number  of nodes in DAG (|V| in {10, 20, 50}), number of edges (|E| = |V| x e, where e in {1,2,5}), and data size (N=|V| x s, where s in {50,100,200,300,500}). 

- `main_realdata.py` reproduces the experiments on real data. The experiments on real data are on FICO HELOC, Adult, Boston and California Housing datasets. The script iterates over two dimensions: the threshold tau (to retrieve computed DAGs) and the sample size N. All the data is downlowded within the `utils.py` script apart from HELOC (this is instead provided in the fico folder within data directory, both [pre-processed](data/fico/WOE_Rud_data.csv) and [original](data/fico/heloc_dataset_v1.csv)). 

In the [Notebooks](Notebooks) folder there are two jupyter notebooks (one for the synthetic data `Main_Synth.ipynb` and one for real data `Main_Real.ipynb`) collecting the results and plotting the charts presented in the paper. Each scenario shown in the notebooks also reports the command used to run it (as per below example).

### Example usage
From terminal at the root folder, run:
```
python main_synth.py --version="r_ex_1b_k20_l3_h2_n0" --branchf=1 --known_p=0.2 --noise_p=0.0 --hidden_l=3 --hidden_n_p=2
```
`main_realdata.py` has the same usage. Refer to the scripts for descriptions of the meaning and usage of the parameters.

### Environment
The code was tested with Python 3.6.9 and 3.9.9. `requirements.txt` provides the necessary packages. Run `pip install -r requirements.txt` from a terminal at the root folder to install all packages in your virtual environment.
