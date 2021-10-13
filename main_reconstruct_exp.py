import numpy as np
np.set_printoptions(suppress=True)
import networkx as nx
import pickle
import random
import pandas as pd
#import tensorflow as tf
#Disable TensorFlow 2 behaviour
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler  
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# tf.config.list_physical_devices('GPU')
import os
from sklearn.metrics import mean_squared_error
from CASTLE2 import CASTLE
from utils import *
from signal import signal, SIGINT
from sys import exit
import argparse
import datetime
from tqdm import tqdm
  
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
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv')

    parser.add_argument("--random_dag", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")

    parser.add_argument('--num_nodes', type = int, default = 10)
    
    parser.add_argument('--dataset_sz', type = int, default = 5000)
    parser.add_argument('--output_log', type = str, default = 'castle_nested_v2.log')
    parser.add_argument('--out_folds', type = int, default = 1)
    parser.add_argument('--in_folds', type = int, default = 10)
    parser.add_argument('--reg_lambda', type = float, default = 1)
    parser.add_argument('--reg_beta', type = float, default = 5)
    parser.add_argument('--gpu', type = str, default = '')
    parser.add_argument('--ckpt_file', type = str, default = 'tmp.ckpt')
    parser.add_argument('--extension', type = str, default = '')
    parser.add_argument('--branchf', type = float, default = 4)
       
    args = parser.parse_args()
    signal(SIGINT, handler)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    version = "p1"
    full_DAG_inject = False
    force_refit = False

    #[50,100,200,500,1000,2000,3000,4000,5000]
    dataset_szs = [100,1000,5000]
    dataset_szs = [int(e*1.25) for e in dataset_szs]

    for dset_sz in tqdm(dataset_szs, desc="Dset"):
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
            num_edges = int(args.num_nodes*args.branchf)
            G = random_dag(args.num_nodes, num_edges)

            noise = random.uniform(0.3, 1.0)
            print("Setting noise to ", noise)
            
            df = gen_data_nonlinear(G, SIZE = dset_sz, var = noise).iloc[:dset_sz]
            df_test =  gen_data_nonlinear(G, SIZE = int(dset_sz*0.25), var = noise)
            
            for i in range(len(G.edges())):
                if len(list(G.predecessors(i))) > 0:
                    df = swap_cols(df, str(0), str(i))
                    df_test = swap_cols(df_test, str(0), str(i))
                    G = swap_nodes(G, 0, i)
                    break      
                    
            #print("Number of parents of G", len(list(G.predecessors(i))))
            print("Edges = ", list(G.edges()))
            
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
                df = gen_data_nonlinear(G, SIZE = dset_sz)
                df_test = gen_data_nonlinear(G, SIZE = int(dset_sz*0.2))
        
        kf_out = KFold(n_splits = args.out_folds, random_state = 0, shuffle = True)
        out_fold = 0
        result_metrics_dict = {}

        for train_idx, val_idx in  tqdm(kf_out.split(df), desc="outer"):
            out_fold += 1
            X_DAG = df.loc[train_idx]
            X_DAG = X_DAG.iloc[:dset_sz]

            scaler = StandardScaler()
            if args.random_dag:
                df = scaler.fit_transform(X_DAG)
            else:
                if args.csv:
                    scaler.fit(pd.read_csv(args.csv))
                    X_DAG = scaler.transform(X_DAG)
                else:
                    X_DAG = scaler.fit_transform(X_DAG)
    
            y_DAG = np.expand_dims(X_DAG[:,0], -1)
            df_test = scaler.transform(df.loc[val_idx])
        
            X_test = df_test
            y_test = df_test[:,0]

            ckpt_file = args.ckpt_file + "_ds" + str(dset_sz)

            ## create baseline for tuning
            castle = CASTLE(num_train = X_DAG.shape[0], num_inputs = X_DAG.shape[1], reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                w_threshold = 0, ckpt_file = ckpt_file)
            castle.fit(X=X_DAG, y=y_DAG, num_nodes=np.shape(X_DAG)[1], X_val=X_test, y_val=y_test,
                    overwrite=force_refit, tune=False, maxed_adj=None, tuned=False, verbose=False)

            MSE_base = mean_squared_error(castle.pred(X_test), y_test)
            with open(args.output_log, "a") as logfile:
                logfile.write(str(dset_sz) + ",  baseliine" ",  "+
                        "${0:.6f}$".format(round(MSE_base,6)) + 
                        '\n')

            if full_DAG_inject:
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
            else:
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


            kf = KFold(n_splits = args.in_folds, random_state = 0, shuffle = True)

            # if args.thetas_exp:
            #     thetas = [-1, 0, 0.01,  0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11 ]
            # else:
            thetas = [-1, 0.05]

            REG_castle = []
            for theta in tqdm(thetas, desc="thetas"):
                fold = 0
                # print("Dataset limits are", np.ptp(X_DAG), np.ptp(X_test), np.ptp(y_test))
                for train_idx, val_idx in tqdm(kf.split(X_DAG), desc="inner"):
                    # castle.__del__()
                    
                    score = {}
                    fold += 1

                    print(f"out_fold ={out_fold}, fold={fold}")
                    print("******* Doing dataset size T= ", len(train_idx),", V=", len(val_idx) , "****************")

                    X_train = X_DAG[train_idx]
                    y_train = np.expand_dims(X_DAG[train_idx][:,0], -1)
                    X_val = X_DAG[val_idx]
                    y_val = X_DAG[val_idx][:,0]

                    # ckpt_file = args.ckpt_file + '.t' + str(theta) + '.F' + str(fold)

                    start = datetime.datetime.now()
                    if theta >= 0:
                        castle = CASTLE(num_train = X_train.shape[0], num_inputs = X_train.shape[1], reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                            w_threshold = theta, ckpt_file = ckpt_file, tune = True, hypertrain = True, adj_mat=loaded_adj)
                        castle.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                overwrite=False, tune=True, maxed_adj=None, tuned=False, verbose=False)
                    else:
                        castle = CASTLE(num_train = X_train.shape[0], num_inputs = X_train.shape[1], reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                            w_threshold = theta, ckpt_file = args.ckpt_file)
                        castle.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                overwrite=force_refit, tune=False, maxed_adj=None, tuned=False, verbose=False)
                    ct = datetime.datetime.now() - start

                    W_est = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
                    # heat_mat(W_est)

                    G1 = nx.from_numpy_matrix(W_est, create_using=nx.DiGraph, parallel_edges=False)

                    REG_castle.append(mean_squared_error(castle.pred(X_test), y_test))
                    print("MSE = ", mean_squared_error(castle.pred(X_test), y_test))

                    if fold > 1:
                        print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle))

                    score["timestamp"] = ct

                    score["MSE"] = mean_squared_error(castle.pred(X_test), y_test)
                    score["Test_size"] = X_test.shape

                    for std_val in [1,2,3]:
                        X_test_sub = X_test[np.where((X_test[:,0] < -std_val) | (X_test[:,0] > std_val))]
                        y_test_sub = y_test[np.where((y_test < -std_val) | (y_test >std_val))]

                        score[f"MSE_std_{std_val}"] = mean_squared_error(castle.pred(X_test_sub), y_test_sub)
                        score[f"Test_size_{std_val}"] = X_test_sub.shape[0]

                    # score["dset"] = dset_sz
                    score["theta"] = theta
                    score["fold"] = fold

                    a = set(G.edges())
                    b = set(G1.edges())

                    import math
                    penalty_mis = -1
                    penalty_add = -1
                    penalty_dir = -1

                    n=G.number_of_nodes()
                    r=2
                    n_perm = math.factorial(n)/math.factorial(n-r)

                    print("Equal sets:",a==b)
                    # print("intersection:",a.intersection(b)   )
                    # print("union:",a.union(b)        )
                    # print("unequal:",a.union(b)  - a.intersection(b))   

                    print("Missing:",len(a-a.intersection(b)))
                    print("Added:",len(b-a.intersection(b)))

                    score["Missing edges"] = len(a-a.intersection(b))
                    score["Added edges"] = len(b-a.intersection(b))

                    score["DAG score"] = (n_perm + len(a-a.intersection(b))*penalty_mis + len(b-a.intersection(b))*penalty_add)/n_perm

                    # result_metrics_dict["dset=",dset_sz,"theta="+str(theta)+", fold="+str(fold)] = score
                    result_metrics_dict["out_fold="+str(out_fold)+"theta="+str(theta)+", fold="+str(fold)] = score

                    print("theta:",theta, ", fold:",fold, ", MSE:",score['MSE'])

                    save_pickle(result_metrics_dict, os.path.join(f"results/Nested{args.out_folds}FoldCASTLE.Reg.Synth.{dset_sz}.{version}.pkl"))



                def format_str(mean, std):
                    return "${0:.6f}".format(round(mean,6)) + " \pm {0:.6f}$    ".format(round(std,6))
                with open(args.output_log, "a") as logfile:
                    logfile.write(str(dset_sz) + ",  " + str(theta)  + ",  "+
                                format_str(np.mean(REG_castle), np.std(REG_castle)) + 
                                '\n')
                print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle)) 