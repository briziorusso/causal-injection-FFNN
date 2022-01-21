import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
np.set_printoptions(suppress=True)
from numpy.core.numeric import NaN
import networkx as nx
import random
import pandas as pd
from signal import signal, SIGINT
import argparse
import datetime
from tqdm import tqdm


from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, f1_score, mean_absolute_error

from CASTLE2 import CASTLE ##TODO change name
from utils import *

#import tensorflow as tf
#Disable TensorFlow 2 behaviour
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


# tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices())
# tf.debugging.set_log_device_placement(True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', default='test')
    parser.add_argument('--csv')
    parser.add_argument('--output_log', type = str, default = 'castle_recon.log')
    parser.add_argument('--ckpt_file', type = str, default = 'recon.ckpt')

    parser.add_argument("--random_dag", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Activate nice mode.")
    parser.add_argument('--num_nodes', type = int, default = 50) #20
    parser.add_argument('--branchf', type = float, default = 2) #4
    parser.add_argument('--known_p', type = float, default = 0.2) #0.2
    parser.add_argument('--noise_p', type = float, default = 0.2) #0.2
    parser.add_argument('--dataset_sz', type = int, default = 5000)

    parser.add_argument('--out_folds', type = int, default = 1)
    parser.add_argument('--in_folds', type = int, default = 1)
    parser.add_argument('--reg_lambda', type = float, default = 1)
    parser.add_argument('--reg_beta', type = float, default = 5)

    parser.add_argument('--gpu', type = str, default = '')
    parser.add_argument('--extension', type = str, default = '')
       
    args = parser.parse_args()
    signal(SIGINT, handler)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # if args.gpu == '':
    #     args.gpu = '/CPU:0'

    # with tf.device(args.gpu):

    version = args.version#"r_ex_5b", 'r_ex_1b_50'
    folder = "./results/"
    args.ckpt_file = os.path.join("./models/",args.ckpt_file)
    # args.csv = 'synth_nonlinear.csv'
    DAG_inject = 'partial' ## One of 'full', 'toy', 'partial'
    force_refit = True
    verbose = False
    standardise = True

    known_perc = args.known_p
    noise_perc = args.noise_p
    thetas = [-1, 0.05]
    seeds_list = [0, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    nodes_list = [10,20,50]
    alphas = [50, 100, 200, 300, 500]#r_ex2[20, 50, 100, 150, 200]

    pbar = tqdm(seeds_list, leave=None)
    for seed in pbar:
        pbar.set_description("Seed %s" % seed)
        pbar.refresh()

        random_stability(seed)

        if seed == 0:
            continue

        pbar2 = tqdm(nodes_list, leave=None)
        for num_nodes in pbar2:
            pbar2.set_description("Nodes %s" % num_nodes)
            pbar2.refresh()

            # if num_nodes in [10,20]:
            #     continue
            
            hidden_n = int(num_nodes*3.2)
            
            dataset_szs = [int(num_nodes*a*1.25) for a in alphas]

            if args.random_dag:
                def swap_cols(df, a, b):
                    df = df.rename(columns = {a : 'temp'})
                    df = df.rename(columns = {b : a})
                    return df.rename(columns = {'temp' : b})
                def swap_nodes(G, a, b):
                    newG = nx.relabel_nodes(G, {a : 'temp'})
                    newG = nx.relabel_nodes(newG, {b : a})
                    return nx.relabel_nodes(newG, {'temp' : b})
                
                #Random DAG
                num_edges = int(num_nodes*args.branchf)
                G = gen_random_dag(num_nodes, num_edges, seed=seed)

                noise = random.uniform(0.3, 1.0)
                if verbose:
                    print("Setting noise to ", noise)
                
                df = gen_data_nonlinear(G, SIZE = 100000, var = noise, percent_noise=noise_perc, seed=seed)
                # df_test =  gen_data_nonlinear(G, SIZE = int(dset_sz*0.25), var = noise)
                
                for i in range(len(G.edges())):
                    if len(list(G.predecessors(i))) > 0:
                        df = swap_cols(df, str(0), str(i))
                        # df_test = swap_cols(df_test, str(0), str(i))
                        G = swap_nodes(G, 0, i)
                        break      

                df = pd.DataFrame(df)
                # df_test = pd.DataFrame(df_test)
                        
                #print("Number of parents of G", len(list(G.predecessors(i))))
                if verbose:
                    print("Edges = ", len(G.edges()), list(G.edges()))
                
            else:
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

                if args.csv:
                    df = pd.read_csv(args.csv)
                    # df_test = df.iloc[-1000:]
                    # df = df.iloc[:dset_sz]
                else: 
                    df = gen_data_nonlinear(G, SIZE = 100000)
                    df_test = gen_data_nonlinear(G, SIZE = int(5000))

            pbar3 = tqdm(dataset_szs, leave=None)
            for dset_sz in pbar3:
                random_stability(seed)
                alpha = dset_sz*0.8/num_nodes

                pbar3.set_description("alpha %s" % alpha)
                pbar3.refresh()

                # if alpha not in [500]:
                #     continue             
                
                if args.out_folds >= 2:
                    kf_out = KFold(n_splits = args.out_folds, random_state = seed, shuffle = True)
                    kf_out_splits = kf_out.split(df)
                else:
                    train, test = train_test_split(df, test_size=0.2, random_state=seed)
                    kf_out_splits = [tuple([train.index.to_numpy(), test.index.to_numpy()])]

                out_fold = 0
                out_file = os.path.join(folder,f"Nested{args.out_folds}FoldCASTLE.Reg.Synth.{num_nodes}.{dset_sz}.{version}.pkl")
                if os.path.exists(out_file):
                    result_metrics_dict = load_pickle(out_file, verbose=False)
                else:
                    result_metrics_dict = {}

                for train_idx, val_idx in  kf_out_splits:#tqdm(kf_out_splits, desc="outer", leave=None):
                    out_fold += 1
                    X_DAG = df.loc[train_idx]
                    X_DAG = X_DAG.iloc[:dset_sz]

                    if standardise:
                        scaler = StandardScaler()
                        if args.random_dag:
                            X_DAG = scaler.fit_transform(X_DAG)
                        else:
                            if args.csv:
                                scaler.fit(pd.read_csv(args.csv))
                                X_DAG = scaler.transform(X_DAG)
                            else:
                                X_DAG = scaler.fit_transform(X_DAG)
                        df_test = scaler.transform(df.loc[val_idx])
                    else:
                        X_DAG = X_DAG.to_numpy()
                        df_test = df.loc[val_idx].to_numpy()

                    y_DAG = np.expand_dims(X_DAG[:,0], -1)
                
                    X_test = df_test
                    y_test = df_test[:,0]

                    ckpt_file = args.ckpt_file + "_ds" + str(dset_sz)

                    ## create baseline for tuning
                    castle = CASTLE(num_train = X_DAG.shape[0], num_inputs = X_DAG.shape[1], n_hidden=hidden_n, 
                                    reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                    w_threshold = 0, ckpt_file = ckpt_file, seed = seed)
                    castle.fit(X=X_DAG, y=y_DAG, num_nodes=np.shape(X_DAG)[1], X_val=X_test, y_val=y_test,
                            overwrite=force_refit, tune=False, maxed_adj=None, tuned=False, verbose=verbose)

                    W_est = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
                    save_pickle(W_est, os.path.join(folder,f"adjmats/W_est.{num_nodes}.{seed}.{dset_sz}.{out_fold}.{version}.pkl"), verbose=False)

                    MSE_base = mean_squared_error(castle.pred(X_test), y_test)
                    with open(args.output_log, "a") as logfile:
                        logfile.write(str(dset_sz) + ",  baseliine" ",  "+
                                "${0:.6f}$".format(round(MSE_base,6)) + 
                                '\n')

                    if DAG_inject == 'full':
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
                    elif DAG_inject == 'toy':
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
                    elif DAG_inject == 'partial':
                        n = int(len(G.edges())*known_perc)
                        known_edges = random.sample(list(G.edges()), n)
                        if verbose:
                            print(len(known_edges), known_edges)
                        loaded_adj = 1 - np.identity(num_nodes)

                        for item in known_edges:
                            loaded_adj[item[1],item[0]]=0
                    
                    if args.in_folds >= 2:
                        kf = KFold(n_splits = args.in_folds, random_state = seed, shuffle = True)
                        kf_splits = kf.split(X_DAG)
                    elif args.in_folds == 1:
                        train_idx = random.sample(range(len(X_DAG)), k=int(len(X_DAG)*0.8))
                        test_idx = [i for i in range(len(X_DAG)) if i not in train_idx]
                        kf_splits = [tuple([train_idx, test_idx])]
                    elif args.in_folds == 0:
                        kf_splits = kf_out_splits
                                    
                    REG_castle = []
                    for theta in tqdm(thetas, desc="thetas", leave=None):
                        fold = 0

                        # if theta == -1: #########################################################
                        #     continue    

                        # print("Dataset limits are", np.ptp(X_DAG), np.ptp(X_test), np.ptp(y_test))
                        for train_idx, val_idx in kf_splits:#tqdm(kf_splits, desc="inner", leave=None):
                            if castle:
                                castle.__del__()

                            score = {}
                            fold += 1

                            if verbose:    
                                print(f"out_fold ={out_fold}, fold={fold}")
                                print("******* Doing dataset size T= ", len(train_idx),", V=", len(val_idx) , "****************")

                            X_train = X_DAG[train_idx]
                            y_train = np.expand_dims(X_DAG[train_idx][:,0], -1)
                            X_val = X_DAG[val_idx]
                            y_val = X_DAG[val_idx][:,0]


                            start = datetime.datetime.now()
                            if theta >= 0:
                                castle = CASTLE(num_train = X_train.shape[0], num_inputs = X_train.shape[1], n_hidden=hidden_n,#X_train.shape[1], 
                                                reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                                    w_threshold = theta, ckpt_file = ckpt_file, tune = True, hypertrain = True, adj_mat=loaded_adj)
                                castle.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                        overwrite=False, tune=True, maxed_adj=None, tuned=False, verbose=verbose)
                            else:
                                castle = CASTLE(num_train = X_train.shape[0], num_inputs = X_train.shape[1], n_hidden=hidden_n,#X_train.shape[1],
                                                reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                                    w_threshold = theta, ckpt_file = args.ckpt_file)
                                castle.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                        overwrite=force_refit, tune=False, maxed_adj=None, tuned=False, verbose=verbose)
                            ct = datetime.datetime.now() - start

                            W_est = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
                            save_pickle(W_est, os.path.join(folder, f"adjmats/W_est.{num_nodes}.{seed}.{dset_sz}.{out_fold}.{fold}.{str(theta)}.{version}.pkl"), verbose=False)
                            # heat_mat(W_est)

                            REG_castle.append(mean_squared_error(castle.pred(X_test), y_test))
                            # print("MSE = ", mean_squared_error(castle.pred(X_test), y_test))

                            # if fold > 1:
                                # print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle))

                            score["timestamp"] = ct

                            score["theta"] = theta
                            score["fold"] = fold
                            score["n_nodes"] = len(G.nodes)
                            score['N_edges'] = len(G.edges)
                            score["data_size"] = dset_sz
                            score["seed"] = seed

                            score["MSE"] = mean_squared_error(castle.pred(X_test), y_test)
                            score["MAE"] = mean_absolute_error(castle.pred(X_test), y_test)
                            score["Test_size"] = X_test.shape

                            for std_val in [1,2,3]:
                                X_test_sub = X_test[np.where((X_test[:,0] < -std_val) | (X_test[:,0] > std_val))]
                                y_test_sub = y_test[np.where((y_test < -std_val) | (y_test >std_val))]
                                if len(y_test_sub) > 0 :
                                    score[f"MSE_std_{std_val}"] = mean_squared_error(castle.pred(X_test_sub), y_test_sub)
                                else:
                                    score[f"MSE_std_{std_val}"] = NaN
                                score[f"Test_size_{std_val}"] = X_test_sub.shape[0]

                            tau_list = list(np.arange(min(W_est.flatten()), max(W_est.flatten()), 0.005))
                            scores = []
                            all_taus = []

                            for tau in tau_list:

                                maxed_adj = DAG_retreive_np(W_est, tau)

                                G1 = nx.from_numpy_matrix(maxed_adj, create_using=nx.DiGraph, parallel_edges=False)

                                a = set(G.edges())
                                b = set(G1.edges())

                                import math
                                penalty_mis = -1
                                penalty_add = -1

                                n=G.number_of_nodes()
                                r=2
                                n_perm = math.factorial(n)/math.factorial(n-r)

                                if verbose:
                                    print("Equal sets:",a==b)
                                    print("intersection:",a.intersection(b)   )
                                    print("union:",a.union(b)        )
                                    print("unequal:",a.union(b)  - a.intersection(b))   
                                    print("Missing edges", len(a-a.intersection(b)))
                                    print("Added edges", len(b-a.intersection(b)))

                                DAG_score = (n_perm + len(a-a.intersection(b))*penalty_mis + len(b-a.intersection(b))*penalty_add)/n_perm
                            #     print("DAG score", DAG_score)

                                scores.append((tau,DAG_score, len(G1.edges()), len(a-a.intersection(b)), len(b-a.intersection(b)), len(a.intersection(b)), len(a.intersection(b))/len(G.edges()) ))
                    #         print(scores)
                            c_tau = max([a[1] for a in scores])   
                            max_tau =  [a[0] for a in scores if a[1] ==c_tau][0]
                            Dscore =    [a[1] for a in scores if a[1] ==c_tau][0]
                            total =    [a[2] for a in scores if a[1] ==c_tau][0]
                            missing =  [a[3] for a in scores if a[1] ==c_tau][0]
                            added =    [a[4] for a in scores if a[1] ==c_tau][0]
                            matching = [a[5] for a in scores if a[1] ==c_tau][0]
                            perfect =    round(len([a[1] for a in scores if a[1] ==c_tau and a[1] ==1])/len([a[1] for a in scores if a[1] ==c_tau]),2)
                            # print('perfect:' ,len([a[1] for a in scores if a[1] ==c_tau and a[1] ==1]))
                            right = [a[6] for a in scores if a[1] ==c_tau][0]
                            wrong = 1-right

                            if verbose:
                                print("Matching:",matching  )
                                print("Missing edges:", missing)
                                print("Added edges:", added)

                            all_taus.append((theta , seed, int(dset_sz), max_tau, int(num_nodes), len(G.edges()), total, matching, missing, added, perfect, right, wrong))

                            score['max_tau'] = max_tau
                            score['matching'] = matching
                            score['missing'] = missing
                            score['added'] = added
                            score['perfect'] = perfect
                            score['DAG_score'] = Dscore
                            score['right'] = right
                            score['wrong'] = wrong

                            if percent_noise>0:
                                len_noisy = int(len(list_nodes)*percent_noise)
                                ## find if there are edges from or to random nodes
                                
                            result_metrics_dict[ "theta="+str(theta)+", num_nodes="+str(num_nodes) + ", data_sz=" + str(dset_sz) +", seed="+str(seed)+", out_fold="+str(out_fold)+", fold="+str(fold)] = score
                
                            # print("theta:",theta, ", fold:",fold, ", MSE:",score['MSE'])

                            save_pickle(result_metrics_dict, out_file, verbose=False)

                        def format_str(mean, std):
                            return "${0:.6f}".format(round(mean,6)) + " \pm {0:.6f}$    ".format(round(std,6))
                        with open(args.output_log, "a") as logfile:
                            logfile.write(str(num_nodes) + ",  " + str(seed) + ",  " + str(dset_sz) + ",  " + str(theta)  + ",  "+
                                        format_str(np.mean(REG_castle), np.std(REG_castle)) + 
                                        '\n')
                        if verbose:    
                            print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle)) 