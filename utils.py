import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx
import random
import pandas as pd
import tensorflow.compat.v1 as tf
import pickle

def random_dag(nodes, edges):
    """Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges."""
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
def gen_data_nonlinear(G, mean = 0, var = 1, SIZE = 10000, perturb = [], sigmoid = True):
    list_edges = G.edges()
    list_vertex = G.nodes()

    order = []
    for ts in nx.algorithms.dag.topological_sort(G):
        order.append(ts)

    g = []
    for v in list_vertex:
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
    g = np.swapaxes(g,0,1)
    return pd.DataFrame(g, columns = list(map(str, list_vertex)))


def random_stability(seed_value=1, deterministic=True):
    '''
        seed_value : int A random seed
        deterministic : negatively effect performance making (parallel) operations deterministic
    '''
    print('Random seed {} set for:'.format(seed_value))
    try:
        import os
        os.environ['PYTHONHASHSEED'] = str(seed_value)
        print(' - PYTHONHASHSEED (env)')
    except:
        pass
    try:
        import random
        random.seed(seed_value)
        print(' - random')
    except:
        pass
    try:
        import numpy as np
        np.random.seed(seed_value)
        print(' - NumPy')
    except:
        pass
    try:
        import tensorflow as tf
        try:
            tf.set_random_seed(seed_value)
        except:
            tf.random.set_seed(seed_value)
        print(' - TensorFlow')
        from keras import backend as K
        if deterministic:
            session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        else:
            session_conf = tf.ConfigProto()
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        K.set_session(sess)
        print(' - Keras')
        if deterministic:
            print('   -> deterministic')
    except:
        pass

    try:
        import torch
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        print(' - PyTorch')
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print('   -> deterministic')
    except:
        pass


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


def heat_mat(mat):
    import seaborn as sns
    cm = sns.light_palette("#003E74", as_cmap=True)
    x=pd.DataFrame(mat).round(3)
    x=x.style.background_gradient(cmap=cm, low= np.min(mat.flatten()), high=np.max(mat.flatten())).format("{:.3}")
    return display(x) 


def plot_DAG(mat, ori_dag, graphic_type, debug=False):
    G1 = nx.from_numpy_matrix(mat, create_using=nx.DiGraph, parallel_edges=False)

    print("Total Number of Edges in G:", G1.number_of_edges())
    print("Max in degree:", max([d for n, d in G1.in_degree()]))
    print("DAG:", nx.is_directed_acyclic_graph(G1))

    if graphic_type=="py":
        from pyvis import network as net
        # from IPython.core.display import display, HTML
        g=net.Network(notebook=True, width='100%', height='600px', directed=True)
        mapping = dict(zip(G1, ["Y"] + ['X{}'.format(str(i)) for i in range(1,len(G1.nodes()))]))
        print(mapping)
        G1 = nx.relabel_nodes(G1, mapping)
        g.from_nx(G1)
        for n in g.nodes:
            n.update({'physics': False})

        return g.show("a.html")
        # display(HTML("a.html"))
        
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
        return nx.draw_networkx_labels(G2,pos,labels,font_size=14, font_color="white")
        
        print("\nBlack Arrow = Edge in the original DAG,\nRed Arrow = Missing Edge, Blue Arrow = Additional Edge ")
  



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