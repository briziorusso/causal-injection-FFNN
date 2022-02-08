import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx
import random
import pandas as pd
import sklearn
import tensorflow.compat.v1 as tf
import pickle
import argparse
import os
from sys import exit
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    return pd.DataFrame(g, columns = list(map(str, range(g.shape[1]))))


def swap_cols(df, a, b):
    df = df.rename(columns = {a : 'temp'})
    df = df.rename(columns = {b : a})
    return df.rename(columns = {'temp' : b})
def swap_nodes(G, a, b):
    newG = nx.relabel_nodes(G, {a : 'temp'})
    newG = nx.relabel_nodes(newG, {b : a})
    return nx.relabel_nodes(newG, {'temp' : b})

def load_or_gen_data(name, csv = None, num_nodes = None, branchf = None, verbose = False, seed=0):
    global G
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
        G = nx.DiGraph()
        csv = './data/fico/WOE_Rud_data.csv'
        csv_y = './data/fico/y_data.csv'
        label = 'RiskPerformance'
        df = pd.read_csv(csv)
        y = pd.read_csv(os.path.join(csv_y))
        y = pd.get_dummies(y[label])[['Bad']].to_numpy()
        df.insert(loc=0, column=label, value=y)

    elif name == 'adult':
        G = nx.DiGraph()
        from pmlb import fetch_data
        label = 'Income_50k'

        dataset = fetch_data(name)
        cols = list(dataset.columns)
        cols = [cols[-1]] + cols[:-1]
        df = dataset[cols]

        ##Rebalance data 
        sample1 = df.iloc[pd.Index(np.random.choice(df[df['target']==1].index, df[df['target']==0].shape[0]))]
        all0 = df[df['target']==0]
        df = pd.concat([sample1,all0],ignore_index=True).sample(frac=1).reset_index(drop=True)


    elif name == 'boston':
        G = nx.DiGraph()
        # from keras.datasets import boston_housing
        # (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
        # if verbose:
        #     print(f'Training data : {train_data.shape}')
        #     print(f'Test data : {test_data.shape}')
        #     print(f'Training sample : {train_data[0]}')
        #     print(f'Training target sample : {train_targets[0]}')
        from sklearn.datasets import load_boston
        label = 'PRICE'

        df = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)
        y = load_boston().target
        df.insert(loc=0, column=label, value=y)


    elif name == 'cali':
        G = nx.DiGraph()
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        label = housing.target_names[0]

        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        y = np.array(housing.target)
        df.insert(loc=0, column=label, value=y)

    return G, df


def load_adj_mat(df, DAG_inject, version, verbose=False, debug=False):

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
    elif all(i in version for i in ['partial', 'adult']):
        partial_mat = 1 - np.identity(df.shape[1])
        if debug:  
            print(partial_mat)
            print(df.columns)

        ## Define what cannot be caused by other variables
        prime_causes = ['age', 'sex', 'race', 'native-country']
        only_effects = ['target','capital-gain', 'capital-loss']
        not_caused_by = [('education',['occupation','hours-per-week'])
                        ,('education-num',['occupation','hours-per-week'])
                        ,('fnlwgt',['occupation','hours-per-week'])
                        ,('relationship',['occupation','hours-per-week'])
                        ,('marital-status',['occupation','hours-per-week'])
                        ]

        for i,col in enumerate(df.columns.values):
            if col in prime_causes:
                ## the whole column is set to 0
                partial_mat[:,i] = 0
            elif col in only_effects:
                ## the whole row, apart from target, is set to 0
                partial_mat[i,1:] = 0    
            elif col in [c[0] for c in not_caused_by]:
                not_causes = [c[1] for c in not_caused_by if c[0]==col][0]
                not_causes_idxs = [i for i in range(len(df.columns)) if list(df.columns)[i] in not_causes]
                if debug:
                    print("0ing:", not_causes_idxs)
                partial_mat[not_causes_idxs,i] = 0  
        loaded_adj = partial_mat

    return loaded_adj

def refine_mat(df, mat, version, debug=False):
    partial_mat = mat
    if debug:  
        print(partial_mat)
        print(df.columns)    
    
    if all(i in version for i in ['refine', 'adult']):

        ## Define what cannot be caused by other variables
        prime_causes = ['age', 'sex', 'race', 'native-country']
        only_effects = ['target','capital-gain', 'capital-loss']
        not_caused_by = [('education',['occupation','hours-per-week'])
                        ,('education-num',['occupation','hours-per-week'])
                        ,('fnlwgt',['occupation','hours-per-week'])
                        ,('relationship',['occupation','hours-per-week'])
                        ,('marital-status',['occupation','hours-per-week'])
                        ]

        for i,col in enumerate(df.columns.values):
            if col in prime_causes:
                ## the whole column is set to 0
                partial_mat[:,i] = 0
            elif col in only_effects:
                ## the whole row, apart from target, is set to 0
                partial_mat[i,1:] = 0    
            elif col in [c[0] for c in not_caused_by]:
                not_causes = [c[1] for c in not_caused_by if c[0]==col][0]
                not_causes_idxs = [i for i in range(len(df.columns)) if list(df.columns)[i] in not_causes]
                if debug:
                    print("0ing:", not_causes_idxs)
                partial_mat[not_causes_idxs,i] = 0  
        
    return partial_mat


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
    # x.style.set_properties(**{
    # 'background-color': 'white',
    # 'font-size': '20pt',
    # })
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
        # from pyvis.network import Network
        # from IPython.core.display import display, HTML
        # net = Network(notebook=True)
        # net.from_nx(G)
        # net.toggle_drag_nodes(False)


        g=net.Network(notebook=True, width='100%', height='600px', directed=True)
        if names is None:
            mapping = dict(zip(G1, ["Y"] + ['X{}'.format(str(i)) for i in range(1,len(G1.nodes()))]))
        else:
            mapping = dict(zip(G1, [i for i in names]))
        if debug:
            print(mapping)
        G1 = nx.relabel_nodes(G1, mapping)
        
        g.from_nx(G1)
        g.prep_notebook()

        for n in g.nodes:
            n.update({'physics': False})
        from IPython.display import HTML
        HTML(g.write_html("a.html"))  
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


def plot_ly_by(df, 
        x, x_desc, 
        y1='right', y1_desc="Reconstruction Accuracy", 
        y1_range=[-0.8, 1], y1_ticks=[0,0.2,0.4,0.6,0.8,1], y1_vis=True,
        y2='MSEmean', y2_desc="Mean Squared Error",
        margin_list = [10, 10, 0, 10, 0],
        y2_range= [0, 1.8], y2_ticks=[0,0.2,0.5,0.8], y2_vis=True,
        showleg = True,
        legend_cord = [0.8,1.05],
        save=False, name='',version='', xwidth = 600,
        main_gray = '#262626',
        sec_gray = '#595959',
        main_blue = '#005383',
        sec_blue = '#0085CA'):
    df = df.sort_values(by=[x],axis=0)
    # df=df[['Type', 'N_nodes','alpha', 'MSE', 'MAE','right']]
    mses = df.groupby(['Type',x], as_index=False).agg([ 'count','mean','std']).round(2).reset_index()
    mses.columns = list(map(''.join, mses.columns.values))
    mses['text'] = mses[['MSEmean','MSEstd']].apply(lambda x : '{} ({})'.format(x[0],x[1]), axis=1)
    mses[x] = mses[x].astype(str)
    display(mses)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    msesC = mses[mses['Type']==-1]
    fig.add_trace(
        go.Bar(x=msesC[x], y=msesC[y2], name="CASTLEMSE",
        marker_color=main_gray,#'#FF4136', 
        marker_line_color=main_gray,#rgb(8,48,107)',
                    marker_line_width=2, opacity=0.6,
        # error_y=dict(
        #         type='data', # value of error bar given in data coordinates
        #         array=msesC['MSEstd'],
        #         visible=True,
        #         color='Black',
        #         thickness=1.5,
        #         width=3),
        showlegend=False,
            text=msesC['text'],
            textposition='auto',
                    ),
        secondary_y=True,
    )
    msesI = mses[mses['Type']==0.05]
    fig.add_trace(
        go.Bar(x=msesI[x], y=msesI[y2], name="InjectedMSE",
        marker_color=main_blue,#'#0085CA',#'#3D9970', 
        marker_line_color=main_blue,#'#0085CA',#'White',
                    marker_line_width=2, opacity=0.6,
        # error_y=dict(
        #         type='data', # value of error bar given in data coordinates
        #         array=msesI['MSEstd'],
        #         visible=True,
        #         color='Black',
        #         thickness=1.5,
        #         width=3),
        showlegend=False,
            text=msesI['text'],
            textposition='inside',
                    ),
        secondary_y=True,
    )

    fig.add_trace(go.Box(
        y=df[df['Type']==-1][y1],
        x=df[df['Type']==-1][x].astype(str),
        boxmean='sd', # represent mean
        boxpoints=False,#'outliers',
        name='CASTLE+',
        marker_color=main_gray#'#FF4136'
    ),
        secondary_y=False,)

    if y1=='right':
        d = msesC['rebasedmean']#[i*1.2 for i in msesC['rightmean']]

        fig.add_trace(
            go.Scatter(x=msesC[x], y=d, name="CASTLE+20%" , mode='markers', marker_symbol='line-ew',
                    marker=dict(
                    color='Black',
                    size=50,
                    line=dict(
                        color='Black',
                        width=2)
                        ),
                    showlegend=False),
            secondary_y=False, 
        )

    fig.add_trace(go.Box(
        y=df[df['Type']==0.05][y1],
        x=df[df['Type']==0.05][x].astype(str),
        boxmean='sd', # represent mean
        boxpoints=False,#'outliers',
        name='Injected',
        marker_color=main_blue#'#0085CA'#'#3D9970'
    ),
        secondary_y=False,)
    fig.update_layout(
        # yaxis_title='normalized moisture',
        boxmode='group' # group together boxes of the different traces for each value of x
    )

    if xwidth == 600:
        x1shift=-330
        x2shift=270
    elif xwidth == 1100:
        x1shift=-580
        x2shift=480
    else:
        x1shift = -xwidth*0.53       
        x2shift = xwidth*0.42       

    if not y1_vis:
        y1_desc = ''
    else:
        fig.add_annotation(
                # Don't specify y position, because yanchor="middle" should do it
                yshift=150,
                xshift=x1shift,
                align="left",
                valign="top",
                text=y1_desc,
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                # Parameter textangle allow you to rotate annotation how you want
                textangle=-90
            )
    if not y2_vis:
        y2_desc = ''
    else:
        fig.add_annotation(
            # Don't specify y position, because yanchor="middle" should do it
            yshift=10,
            xshift=x2shift,
            align="left",
            valign="top",
            text=y2_desc,
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            # Parameter textangle allow you to rotate annotation how you want
            textangle=-90
        )
    # Set x-axis title
    fig.update_xaxes(showgrid=True,
    title={'text':x_desc#,'font':{'size':18}
    })
    # Set y-axes titles
    fig.update_yaxes(showgrid=True,nticks=10,zeroline=True, title={'text':""}, 
    range=y1_range,
    tickvals=y1_ticks,
    tickformat=".0%",
    secondary_y=False,
    showticklabels=y1_vis
    )
    fig.update_yaxes(showgrid=True,nticks=10,zeroline=True, title={'text':""},
    range=y2_range,
    tickvals=y2_ticks,
    secondary_y=True,
    showticklabels=y2_vis)

    # Add figure title
    fig.update_layout(
        showlegend=showleg,
        title='',
        legend={
            'y':legend_cord[1],
            'x':legend_cord[0],
            # 'y':-0.08,
            # 'x':0.92,
            'orientation':"h",
            'xanchor': 'center',
            'yanchor': 'top'},
        template='plotly_white',
        autosize=True,
        width=xwidth, height=350, 
        margin=dict(
            l=margin_list[0],
            r=margin_list[1],
            b=margin_list[2],
            t=margin_list[3],
            pad=margin_list[4],
        ),
        font=dict(
            family='Serif',#"Courier New, monospace",
            size=18,
            color="Black"
        )    
    )

    if save:
        output_folder = "figures"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        out_path = os.path.join(output_folder,f"plot_{name}_{version}.png")

        import kaleido
        fig.write_image(out_path)

    fig.show()

def plot_ly_by_compare(df, 
        x, x_desc, 
        y1='right', y1_desc="Reconstruction Accuracy", 
        margin_list = [10, 10, 0, 10, 0],
        y1_range=[-0.8, 1], y1_ticks=[0,0.2,0.4,0.6,0.8,1], y1_vis=True,
        y2='MSEmean', y2_desc="Mean Squared Error",
        y2_range= [0, 1.8], y2_ticks=[0,0.2,0.4,0.6,0.8], y2_vis=True,
        showleg = True,
        legend_cord = [0.8,1.05],
        save=False, name='',version='', xwidth = 1100,
        main_gray = '#262626',
        sec_gray = '#595959',
        main_blue = '#005383',
        sec_blue = '#0085CA',
        names_list = ['CASTLE+ w/noise','CASTLE+','Injected','Injected w/noise']
       ):
    colors_list = [sec_gray,main_gray,main_blue,sec_blue]

    df = df.sort_values(by=[x],axis=0)
    # df=df[['Type', 'N_nodes','alpha', 'MSE', 'MAE','right']]
    mses = df.groupby(['Type','V',x], as_index=False).agg([ 'count','mean','std']).round(2).reset_index()
    mses.columns = list(map(''.join, mses.columns.values))
    mses['text'] = mses[['MSEmean','MSEstd']].apply(lambda x : '{} ({})'.format(x[0],x[1]), axis=1)
    mses[x] = mses[x].astype(str)

    print(np.unique(df['V']))
    display(mses)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    ########## BarCharts
    msesC = mses[(mses['Type']==-1) & (mses['V']==np.unique(mses['V'])[1])]
    fig.add_trace(
        go.Bar(x=msesC[x], y=msesC[y2], name="CASTLEMSE",
        marker_color=colors_list[0],#'#FF4136', 
        marker_line_color=colors_list[0],#rgb(8,48,107)',
                    marker_line_width=2, opacity=0.6,
        # error_y=dict(
        #         type='data', # value of error bar given in data coordinates
        #         array=msesC['MSEstd'],
        #         visible=True,
        #         color='Black',
        #         thickness=1.5,
        #         width=3),
        showlegend=False,
            text=msesC['text'],
            textposition='auto',
                    ),
        secondary_y=True,
    )
    msesC = mses[(mses['Type']==-1) & (mses['V']==np.unique(mses['V'])[0])]
    fig.add_trace(
        go.Bar(x=msesC[x], y=msesC[y2], name="CASTLEMSE",
        marker_color=colors_list[1],#'#FF4136', 
        marker_line_color=colors_list[1],#rgb(8,48,107)',
                    marker_line_width=2, opacity=0.6,
        # error_y=dict(
        #         type='data', # value of error bar given in data coordinates
        #         array=msesC['MSEstd'],
        #         visible=True,
        #         color='Black',
        #         thickness=1.5,
        #         width=3),
        showlegend=False,
            text=msesC['text'],
            textposition='auto',
                    ),
        secondary_y=True,
    )


    msesI = mses[(mses['Type']==0.05) & (mses['V']==np.unique(mses['V'])[0])]
    fig.add_trace(
        go.Bar(x=msesI[x], y=msesI[y2], name="InjectedMSE",
        marker_color=colors_list[2],#'#0085CA',#'#3D9970', 
        marker_line_color=colors_list[2],#'#0085CA',#'White',
                    marker_line_width=2, opacity=0.6,
        # error_y=dict(
        #         type='data', # value of error bar given in data coordinates
        #         array=msesI['MSEstd'],
        #         visible=True,
        #         color='Black',
        #         thickness=1.5,
        #         width=3),
        showlegend=False,
            text=msesI['text'],
            textposition='inside',
                    ),
        secondary_y=True,
    )
    msesI = mses[(mses['Type']==0.05) & (mses['V']==np.unique(mses['V'])[1])]
    fig.add_trace(
        go.Bar(x=msesI[x], y=msesI[y2], name="InjectedMSE",
        marker_color=colors_list[3],#'#0085CA',#'#3D9970', 
        marker_line_color=colors_list[3],#'#0085CA',#'White',
                    marker_line_width=2, opacity=0.6,
        # error_y=dict(
        #         type='data', # value of error bar given in data coordinates
        #         array=msesI['MSEstd'],
        #         visible=True,
        #         color='Black',
        #         thickness=1.5,
        #         width=3),
        showlegend=False,
            text=msesI['text'],
            textposition='inside',
                    ),
        secondary_y=True,
    )


    ########## Boxplots
    fig.add_trace(go.Box(
        y=df[(df['Type']==-1) & (df['V']==np.unique(df['V'])[1])][y1],
        x=df[(df['Type']==-1) & (df['V']==np.unique(df['V'])[1])][x].astype(str),
        boxmean='sd', # represent mean
        boxpoints=False,#'outliers',
        name=names_list[0],
        marker_color= colors_list[0]
    ),
    secondary_y=False,)
    fig.add_trace(go.Box(
        y=df[(df['Type']==-1) & (df['V']==np.unique(df['V'])[0])][y1],
        x=df[(df['Type']==-1) & (df['V']==np.unique(df['V'])[0])][x].astype(str),
        boxmean='sd', # represent mean
        boxpoints=False,#'outliers',
        name=names_list[1],
        marker_color=colors_list[1]#'#FF4136'
    ),
        secondary_y=False,)


    if y1=='right':

        msesC = mses[(mses['Type']==-1) & (mses['V']==np.unique(mses['V'])[0])]
        d = msesC['rebasedmean']#[i*1.2 for i in msesC['rightmean']]

        fig.add_trace(
            go.Scatter(x=msesC[x], y=d, name="CASTLE+20%" , mode='markers', marker_symbol='line-ew',
                    marker=dict(
                    color='Black',
                    size=50,
                    line=dict(
                        color='Black',
                        width=2)
                        ),
                    showlegend=False),
            secondary_y=False, 
        )


        msesC = mses[(mses['Type']==-1) & (mses['V']==np.unique(mses['V'])[1])]
        d = msesC['rebasedmean']#[i*1.2 for i in msesC['rightmean']]

        fig.add_trace(
            go.Scatter(x=msesC[x], y=d, name="CASTLE+20%" , mode='markers', marker_symbol='line-ew',
                    marker=dict(
                    color='Black',
                    size=100,
                    line=dict(
                        color='Black',
                        width=2)
                        ),
                    showlegend=False),
            secondary_y=False, 
        )

    fig.add_trace(go.Box(
        y=df[(df['Type']==0.05) & (df['V']==np.unique(df['V'])[0])][y1],
        x=df[(df['Type']==0.05) & (df['V']==np.unique(df['V'])[0])][x].astype(str),
        boxmean='sd', # represent mean
        boxpoints=False,#'outliers',
        name=names_list[2],
        marker_color=colors_list[2]#'#0085CA'#'#3D9970'
    ),
        secondary_y=False,)
    fig.add_trace(go.Box(
        y=df[(df['Type']==0.05) & (df['V']==np.unique(df['V'])[1])][y1],
        x=df[(df['Type']==0.05) & (df['V']==np.unique(df['V'])[1])][x].astype(str),
        boxmean='sd', # represent mean
        boxpoints=False,#'outliers',
        name=names_list[3],
        marker_color= colors_list[3]
    ),
        secondary_y=False,)


    if y1=='right':
        d = msesC['rebasedmean']#[i*1.2 for i in msesC['rightmean']]

        fig.add_trace(
            go.Scatter(x=msesC[x], y=d, name="CASTLE+20%" , mode='markers', marker_symbol='line-ew',
                    marker=dict(
                    color='Black',
                    size=50,
                    line=dict(
                        color='Black',
                        width=2)
                        ),
                    showlegend=False),
            secondary_y=False, 
        )


    fig.update_layout(
        # yaxis_title='normalized moisture',
        boxmode='group' # group together boxes of the different traces for each value of x
        ,bargap=0.1
        ,bargroupgap=0.1
    )

    if xwidth == 600:
        x1shift=-330
        x2shift=270
    elif xwidth == 1100:
        x1shift=-580
        x2shift=480
    else:
        x1shift = -xwidth*0.53       
        x2shift = xwidth*0.42            

    if not y1_vis:
        y1_desc = ''
    else:
        fig.add_annotation(
                # Don't specify y position, because yanchor="middle" should do it
                yshift=150,
                xshift=x1shift,
                align="left",
                valign="top",
                text=y1_desc,
                showarrow=False,
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                # Parameter textangle allow you to rotate annotation how you want
                textangle=-90
            )
    if not y2_vis:
        y2_desc = ''
    else:
        fig.add_annotation(
            # Don't specify y position, because yanchor="middle" should do it
            yshift=10,
            xshift=x2shift,
            align="left",
            valign="top",
            text=y2_desc,
            showarrow=False,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            # Parameter textangle allow you to rotate annotation how you want
            textangle=-90
        )
    # Set x-axis title
    fig.update_xaxes(showgrid=True,
    title={'text':x_desc#,'font':{'size':18}
    })
    # Set y-axes titles
    fig.update_yaxes(showgrid=True,nticks=10,zeroline=True, title={'text':""#,'font':{'size':18}
    }, 
    range=y1_range,
    tickvals=y1_ticks,
    tickformat=".0%",
    secondary_y=False,
    showticklabels=y1_vis
    )
    fig.update_yaxes(showgrid=True,nticks=10,zeroline=True, title={'text':""#,'font':{'size':18}
    },
    range=y2_range,
    tickvals=y2_ticks,
    secondary_y=True,
    showticklabels=y2_vis)

    # Add figure title
    fig.update_layout(
        showlegend=showleg,
        title='',
        legend={
            'y':legend_cord[1],
            'x':legend_cord[0],
            # 'y':-0.08,
            # 'x':0.92,
            'orientation':"h",
            'xanchor': 'center',
            'yanchor': 'top'},
        template='plotly_white',
        autosize=True,
        width=xwidth, height=350, 
        margin=dict(
            l=margin_list[0],
            r=margin_list[1],
            b=margin_list[2],
            t=margin_list[3],
            pad=margin_list[4],
        ),
        font=dict(
            family='Serif',#"Courier New, monospace",
            size=18,
            color="Black"
        )    
    )

    if save:
        output_folder = "figures"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        out_path = os.path.join(output_folder,f"plot_{name}_{version}.png")

        import kaleido
        fig.write_image(out_path)

    fig.show()


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


def check_run_status(version, folder='./results/'):
    import re
    import os
    import pandas as pd
    import numpy as np

    nodes = '([\w]+)'
    seeds = '([\w]+)'
    dsets = '([\w.]+)'
    outfs = str(1)
    infs = str(1)
    thetas = '([-+]?([0-9]+)(\.[0-9]+)?)'

    res = []
    for filename in os.listdir(folder):
        match = re.search(f'Nested1FoldCASTLE.Reg.Synth.{nodes}.{dsets}.{version}.pkl$', filename)
        if match != None:
            report = load_pickle(os.path.join(folder, filename), verbose=False)
            for key, value in report.items():
                t = [[version],[value[i] for i in ['theta', 'n_nodes', 'N_edges', 'seed', 'data_size']]]
                res.append([i for sublist in t for i in sublist])

    df = pd.DataFrame(res)
    if len(df)>0:
        df.columns = ['V','Type', 'N_nodes', 'N_edges', 'seed', 'Size']
        df['alpha'] = ((df['Size'].astype(int)*0.8)/df['N_nodes'].astype(int)).astype(int)
        runs_list = df[df['V']==version].groupby(by=['seed','N_nodes','alpha','Type'], as_index=False).size()

        # comb_list1 =  [''.join(i) for i in zip(runs_list['seed'].map(str),runs_list['N_nodes'].map(str),runs_list['alpha'].map(str))]   
        comb_list =  [''.join(i) for i in zip(runs_list['seed'].map(str),runs_list['N_nodes'].map(str),runs_list['alpha'].map(str),runs_list['Type'].map(str))]   
    else:
        comb_list = []

    return comb_list

