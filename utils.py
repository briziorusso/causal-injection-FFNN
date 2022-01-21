import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx
import random
import pandas as pd
import tensorflow.compat.v1 as tf
import pickle
import argparse
import os
from sys import exit
import seaborn as sns

def load_args():
    """ Load args and run some basic checks.
        Args loaded from:
        - Huggingface transformers training args (defaults for using their model)
        - Manual args from .yaml file
    """
    import sys, yaml 
    assert sys.argv[1] in ['train', 'test', 'chen']
    # Load args from file
    with open(f'config/{sys.argv[1]}.yaml', 'r') as f:
        args = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        for arg in args.__dict__:
            try:
                setattr(args, arg, getattr(args, arg))
            except AttributeError:
                pass
            
            # yaml.dump(args.__dict__, f)

    return args

def random_stability(seed_value=0, deterministic=True, verbose=False):
    '''
        seed_value : int A random seed
        deterministic : negatively effect performance making (parallel) operations deterministic
    '''
    if verbose:
        print('Random seed {} set for:'.format(seed_value))
    try:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        if verbose:
            print(' - PYTHONHASHSEED (env)')
    except:
        pass
    try:
        import random
        random.seed(seed_value)
        if verbose:
            print(' - random')
    except:
        pass
    try:
        import numpy as np
        np.random.seed(seed_value)
        if verbose:
            print(' - NumPy')
    except:
        pass
    try:
        import tensorflow as tf
        try:
            tf.set_random_seed(seed_value)
        except:
            tf.random.set_seed(seed_value)
        if verbose:
            print(' - TensorFlow')
        from keras import backend as K
        if deterministic:
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        else:
            session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        if verbose:
            print(' - Keras')
        if deterministic:
            if verbose:
                print('   -> deterministic')
    except:
        pass

    try:
        import torch
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        if verbose:
            print(' - PyTorch')
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if verbose:
                print('   -> deterministic')
    except:
        pass


## gen_random_dag() and  gen_data_nonlinear() are from https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/alg/castle/CASTLE.py
def gen_random_dag(nodes, edges, seed=0):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""

    random_stability(seed)

    G = nx.DiGraph()
    for i in range(nodes):
        G.add_node(i)
    while edges > 0:
        a = random.randint(0,nodes-1)
        b=a
        while b==a:
            b = random.randint(0,nodes-1)
        G.add_edge(a,b)
        if nx.is_directed_acyclic_graph(G):
            edges -= 1
        else:
            # we closed a loop!
            G.remove_edge(a,b)
    return G


# This function generates data according to a DAG provided in list_vertex and list_edges with mean and variance as input
# It will apply a perturbation at each node provided in perturb.
def gen_data_nonlinear(G, mean = 0, var = 1, SIZE = 10000, perturb = [], sigmoid = True, percent_noise=0, seed=0):
    random_stability(seed)

    list_edges = G.edges()
    list_nodes = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_nodes:
        if v in perturb:
            g.append(np.random.normal(mean,var,SIZE))
            print("perturbing ", v, "with mean var = ", mean, var)
        else:
            g.append(np.random.normal(0,1,SIZE))

    for o in order:
        for edge in list_edges:
            if o == edge[1]: # if there is an edge into this node
                if sigmoid:
                    g[edge[1]] += 1/1+np.exp(-g[edge[0]])
                else:   
                    g[edge[1]] +=np.square(g[edge[0]])
    
    ### Add random noise variables completely disjunt from the other vars
    if percent_noise>0:
        len_noisy = int(len(list_nodes)*percent_noise)
        for n in range(len_noisy):
            g.append(np.random.normal(mean,var,SIZE))

    g = np.swapaxes(g,0,1)
    return pd.DataFrame(g, columns = list(map(str, list_nodes)))


def swap_cols(df, a, b):
    df = df.rename(columns = {a : 'temp'})
    df = df.rename(columns = {b : a})
    return df.rename(columns = {'temp' : b})
def swap_nodes(G, a, b):
    newG = nx.relabel_nodes(G, {a : 'temp'})
    newG = nx.relabel_nodes(newG, {b : a})
    return nx.relabel_nodes(newG, {'temp' : b})

def load_or_gen_data(name, csv = None, num_nodes = None, branchf = None, verbose = False, seed=0):
    random_stability(seed)
    if name == 'toy':
        '''
        Toy DAG
        The node '0' is the target in the Toy DAG
        '''
        G = nx.DiGraph()
        for i in range(10):
            G.add_node(i)
        G.add_edge(1,2)
        G.add_edge(1,3)
        G.add_edge(1,4)
        G.add_edge(2,5)
        G.add_edge(2,0)
        G.add_edge(3,0)
        G.add_edge(3,6)
        G.add_edge(3,7)
        G.add_edge(6,9)
        G.add_edge(0,8)
        G.add_edge(0,9)

        if csv != None:
            df = pd.read_csv(csv)
            # df_test = df.iloc[-1000:]
            # df = df.iloc[:dset_sz]
        else: 
            df = gen_data_nonlinear(G, SIZE = 100000, seed=seed)

    elif name == 'random':
        #Random DAG
        num_edges = int(num_nodes*branchf)
        G = gen_random_dag(num_nodes, num_edges, seed=seed)

        noise = random.uniform(0.3, 1.0)

        df = gen_data_nonlinear(G, SIZE = 100000, var = noise, seed=seed)
        
        for i in range(len(G.edges())):
            if len(list(G.predecessors(i))) > 0:
                df = swap_cols(df, str(0), str(i))
                G = swap_nodes(G, 0, i)
                break      

        df = pd.DataFrame(df)
        if verbose:
            print("Edges = ", len(G.edges()), list(G.edges()))
    
    elif name == 'fico':
        csv = './data/fico/WOE_Rud_data.csv'
        csv_y = './data/fico/y_data.csv'
        label = 'RiskPerformance'
        df = pd.read_csv(csv)
        y = pd.read_csv(os.path.join(csv_y))
        y = pd.get_dummies(y[label])[['Bad']].to_numpy()
        df.insert(loc=0, column=label, value=y)
        G = None

    elif name == 'adult':
        from pmlb import fetch_data
        label = 'Income_50k'

        dataset = fetch_data(name)
        cols = list(dataset.columns)
        cols = [cols[-1]] + cols[:-1]
        df = dataset[cols]
        G = None

    return G, df


def load_adj_mat(DAG_inject, known_perc, num_nodes, verbose=False):

    if DAG_inject == 'toy':
    ##Partial inject experiment
        loaded_adj = np.array((
            [0.0,1.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0],
            [1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
            [1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
            [1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0],
            [1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0],
            [1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0],
            [1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0,1.0],
            [1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0],
            [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0],
            [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0]
            ))

    elif DAG_inject == 'full':
        loaded_adj = np.array((
            [0.0,0.002,0.017,0.021,0.005,0.045,0.035,0.007,0.053,0.055],
            [0.008,0.0,0.072,0.061,0.07,0.005,0.011,0.008,0.004,0.007],
            [0.095,0.036,0.0,0.031,0.027,0.139,0.009,0.005,0.011,0.008],
            [0.071,0.04,0.031,0.0,0.035,0.003,0.108,0.109,0.011,0.007],
            [0.013,0.045,0.022,0.027,0.0,0.008,0.011,0.004,0.007,0.005],
            [0.039,0.003,0.022,0.006,0.005,0.0,0.009,0.007,0.003,0.014],
            [0.038,0.004,0.007,0.011,0.007,0.01,0.0,0.023,0.003,0.163],
            [0.024,0.006,0.004,0.01,0.003,0.022,0.028,0.0,0.009,0.009],
            [0.026,0.004,0.006,0.002,0.009,0.004,0.015,0.005,0.0,0.014],
            [0.026,0.004,0.004,0.005,0.004,0.008,0.073,0.004,0.007,0.0]
            ))

    elif DAG_inject == 'fico_size':
        loaded_adj = pd.read_csv("fico_adj_matrix.csv").to_numpy()

    return loaded_adj

def DAG_retrieve(mat, threshold):
    def max_over_diag(mat):
        if tf.is_tensor(mat):
            import tensorflow
            up_tri = tensorflow.experimental.numpy.triu(mat, k=0)
            low_tri = tensorflow.experimental.numpy.tril(mat, k=0)

            up_mask = tf.cast(tf.math.equal(up_tri , tf.math.maximum(up_tri, tf.transpose(low_tri))),tf.int32)
            low_mask = tf.cast(tf.math.equal(tf.transpose(low_tri) , tf.math.maximum(up_tri, tf.transpose(low_tri))),tf.int32)
            
            up_mask = tf.cast(up_mask, tf.float32)
            low_mask = tf.cast(low_mask, tf.float32)

            maxed_adj = (up_tri * up_mask) + tf.transpose(tf.transpose(low_tri) * low_mask)

        else:
            up_tri = np.triu(mat)
            low_tri = np.tril(mat)

            up_mask = (up_tri == np.maximum(up_tri, low_tri.T)).astype(int)
            low_mask = (low_tri.T == np.maximum(up_tri, low_tri.T)).astype(int)

            maxed_adj = (up_tri * up_mask) + (low_tri.T * low_mask).T

        return maxed_adj

    def zero_under_t(mat, threshold):
        if tf.is_tensor(threshold):
            mat = tf.cast(tf.convert_to_tensor(mat),tf.float32)
            mask = tf.cast(tf.math.greater(mat, threshold), tf.int32)
            mask = tf.cast(mask,tf.float32)
        else:
            mask = (mat > threshold).astype(int)

        thresholded_mat = mat * mask
        return thresholded_mat

    maxed_adj = zero_under_t(max_over_diag(mat), threshold)

    return maxed_adj

def DAG_retreive_np(mat, threshold):

    def max_over_diag(mat):
        up_tri = np.triu(mat)
        low_tri = np.tril(mat)

        up_mask = (up_tri == np.maximum(up_tri, low_tri.T)).astype(int)
        low_mask = (low_tri.T == np.maximum(up_tri, low_tri.T)).astype(int)

        maxed_adj = (up_tri * up_mask) + (low_tri.T * low_mask).T

        return maxed_adj

    def zero_under_t(mat, threshold):
        mask = (mat > threshold).astype(int)
        thresholded_mat = mat * mask
        return thresholded_mat
    
    return zero_under_t(max_over_diag(mat), threshold)


def heat_mat(mat, names=None, colour="#003E74"):
    cm = sns.light_palette(colour, as_cmap=True)
    x=pd.DataFrame(mat).round(3)
    if names is not None:
        x.columns = names
        x.index = names
    x=x.style.background_gradient(cmap=cm, low= np.min(mat.flatten()), high=np.max(mat.flatten())).format("{:.3}")
    return display(x) 


def plot_DAG(mat, graphic_type, debug=False, ori_dag=None, names=None):
    if type(mat) is np.ndarray:
        G1 = nx.from_numpy_matrix(mat, create_using=nx.DiGraph, parallel_edges=False)
    elif isinstance(mat, nx.classes.digraph.DiGraph):
        G1 = mat
    else:
        pass

    print("Total Number of Edges in G:", G1.number_of_edges())
    print("Max in degree:", max([d for n, d in G1.in_degree()]))
    print("DAG:", nx.is_directed_acyclic_graph(G1))

    if graphic_type=="py":
        from pyvis import network as net
        # from IPython.core.display import display, HTML
        g=net.Network(notebook=True, width='100%', height='600px', directed=True)
        if names is None:
            mapping = dict(zip(G1, ["Y"] + ['X{}'.format(str(i)) for i in range(1,len(G1.nodes()))]))
        else:
            mapping = dict(zip(G1, [i for i in names]))
        if debug:
            print(mapping)
        G1 = nx.relabel_nodes(G1, mapping)
        g.from_nx(G1)
        for n in g.nodes:
            n.update({'physics': False})
#         from IPython.display import HTML
#         HTML(g.write_html())  
        return g.show("a.html")
    
        
    elif graphic_type=="nx":
        pos=nx.planar_layout(G1)
    #     scale = 2
    #     pos.update((x, y*scale) for x, y in pos.items())
        labels={}
        color_map = []
        e_color_map = []
        for i in range(len(G1.nodes())):
            if i == 0:
                labels[0] = "Y"
                color="green"
            else:
                labels[i] = "$X_{}$".format(str(i))
                color="#003E74"
            color_map.append(color)
        
        G2 = G1.copy()
        if ori_dag is not None:
            G = ori_dag

            for edge in G.edges():
                if edge not in G1.edges():
                    if debug:
                        print(edge)
                    G2.add_edge(edge[0],edge[1])

            if debug:
                print(G.edges())
                print(G1.edges())    
                print(G2.edges())    

            for edge in G2.edges():
                if edge in G.edges() and edge in G1.edges():
                    color = "black"
                elif edge not in G.edges() and edge in G1.edges():    
                    color = "blue" #Added Edge
                elif edge in G.edges() and edge not in G1.edges():    
                    color = "red" #Missing Edge
                e_color_map.append(color)
            
            
        nx.draw(G2,pos,node_size=400, node_color=color_map, edge_color=e_color_map)
        nx.draw_networkx_labels(G2,pos,labels,font_size=14, font_color="white")
        
        print("\nBlack Arrow = Edge in the original DAG,\nRed Arrow = Missing Edge, Blue Arrow = Additional Edge ")
        return G2



def save_pickle(obj, filename, verbose=True):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=-1)
        if verbose:
            print(f'Dumped PICKLE to {filename}')


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        if verbose:
            print(f'Loaded PICKLE from {filename}')
        return obj




def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.') 