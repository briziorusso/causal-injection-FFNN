import numpy as np
from numpy.core.numeric import NaN
np.set_printoptions(suppress=True)
import networkx as nx
import pickle
import random
import pandas as pd
#import tensorflow as tf
#Disable TensorFlow 2 behaviour
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, classification_report
from CASTLE2 import CASTLE
from utils import *
from signal import signal, SIGINT
from sys import exit
import argparse
import datetime
from tqdm import tqdm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


# tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# tf.config.list_physical_devices('GPU')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv')

    parser.add_argument("--random_dag", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Activate nice mode.")

    parser.add_argument('--num_nodes', type = int, default = 20)
    parser.add_argument('--branchf', type = float, default = 4)
    
    parser.add_argument('--dataset_sz', type = int, default = 5000)
    parser.add_argument('--output_log', type = str, default = 'fico2.log')
    parser.add_argument('--ckpt_file', type = str, default = 'fico2.ckpt')

    parser.add_argument('--out_folds', type = int, default = 5)
    parser.add_argument('--in_folds', type = int, default = 5)
    parser.add_argument('--reg_lambda', type = float, default = 1)
    parser.add_argument('--reg_beta', type = float, default = 5)
    parser.add_argument('--gpu', type = str, default = '')
    parser.add_argument('--extension', type = str, default = '')
       
    args = parser.parse_args()
    signal(SIGINT, handler)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    dag_type = 'fico' ##'fico','toy','random'
    version = "fico2"
    DAG_inject = 'fico' ## One of 'full', 'toy', 'partial', 'fico', 'fico_size'
    folder = './results/'
    csv_y = './data/fico/y_data.csv'
    force_refit = True
    verbose = False
    known_perc = 0.2
    dataset_szs = [100000]
    thetas = [] ### Leave empty for automatic binning of the adj_mat

    ## fico size
    # DAG_inject = 'fico_size' ## One of 'full', 'toy', 'partial', 'fico', 'fico_size'
    # thetas = [-1,0.006] ### Leave empty for automatic binning of the adj_mat
    # dataset_szs = [100,500,1000,2000,3000,4000,5000,6000,7000,8000]
    ##
    # dataset_szs = [args.dataset_sz]
    #[50,100,200,500,1000,2000,3000,4000,5000]
    # dataset_szs = [int(e*1.25) for e in dataset_szs]
    random_stability(0)

    if dag_type == 'random' :
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
        G = gen_random_dag(args.num_nodes, num_edges)

        noise = random.uniform(0.3, 1.0)
        if verbose:
            print("Setting noise to ", noise)
        
        df = gen_data_nonlinear(G, SIZE = 100000, var = noise)
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
        print("Edges = ", len(G.edges()), list(G.edges()))
        
    elif dag_type == 'toy':
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
            df_test = gen_data_nonlinear(G, SIZE = int(100000*0.2))
    elif dag_type == 'fico':
        df = pd.read_csv(args.csv)
        label = 'RiskPerformance'
        y = pd.read_csv(os.path.join(csv_y))
        y = pd.get_dummies(y[label])[['Bad']].to_numpy()
        df.insert(loc=0, column=label, value=y)

    for dset_sz in tqdm(dataset_szs, desc="Dset", leave=None):
        
        kf_out = KFold(n_splits = args.out_folds, random_state = 0, shuffle = True)
        out_fold = 0

        out_file = os.path.join(folder,f"Nested{args.out_folds}FoldCASTLE.Reg.Synth.{dset_sz}.{version}.pkl")
        if os.path.exists(out_file):
            result_metrics_dict = load_pickle(out_file, verbose=False)
        else:
            result_metrics_dict = {}


        for train_idx, val_idx in  tqdm(kf_out.split(df), desc="outer", leave=None):
            out_fold += 1
            X_DAG = df.loc[train_idx]
            X_DAG = X_DAG.iloc[:dset_sz]
            y_DAG = np.expand_dims(X_DAG.iloc[:,0], -1)

            scaler = StandardScaler()
            if dag_type == 'random' :
                X_DAG = scaler.fit_transform(X_DAG)
                X_test = scaler.transform(df.loc[val_idx])
            elif dag_type == 'toy' :
                if args.csv:
                    scaler.fit(pd.read_csv(args.csv))
                    X_DAG = scaler.transform(X_DAG)
                    X_test = scaler.transform(df.loc[val_idx])
                else:
                    X_DAG = scaler.fit_transform(X_DAG)
                    X_test = scaler.transform(df.loc[val_idx])
            if dag_type == 'fico' :
                    X_DAG = X_DAG.to_numpy()
                    X_test = df.loc[val_idx].to_numpy()

            y_test = X_test[:,0]

            ckpt_file = args.ckpt_file + "_ds" + str(dset_sz)

            ## create baseline for tuning
            castle = CASTLE(num_train = X_DAG.shape[0], num_inputs = X_DAG.shape[1], n_hidden=X_DAG.shape[1], 
                            reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                            w_threshold = 0, ckpt_file = ckpt_file)
            castle.fit(X=X_DAG, y=y_DAG, num_nodes=np.shape(X_DAG)[1], X_val=X_test, y_val=y_test,
                    overwrite=force_refit, tune=False, maxed_adj=None, tuned=False, verbose=False)

            W_est = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
            save_pickle(W_est, os.path.join(folder,f"adjmats/W_est.{dset_sz}.{out_fold}.{version}.pkl"), verbose=False)

            if dag_type == 'fico':
                preds = np.where(castle.pred(X_test) > 0.5, 1, 0).flatten() 
                if verbose:
                    print(classification_report(y_test, preds, digits = 3))
                    print('Accuracy =',round(classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy'],3))
                MSE_base = classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy']
            else:
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
                loaded_adj = 1 - np.identity(args.num_nodes)

                for item in known_edges:
                    loaded_adj[item[1],item[0]]=0
            elif DAG_inject == 'fico_size':
                loaded_adj = pd.read_csv("fico_adj_matrix.csv").to_numpy()
            elif DAG_inject == 'fico':
                loaded_adj = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
            
            kf = KFold(n_splits = args.in_folds, random_state = 0, shuffle = True)
            if len(thetas) == 0:
                min_mat = min(loaded_adj.flatten())
                max_mat = max(loaded_adj.flatten())
                # thetas =  list(np.arange(min_mat, max_mat, round((max_mat-min_mat)/20,3)))
                thetas =  list(np.arange(0, 0.05, 0.001))

            REG_castle = []
            for theta in tqdm(thetas, desc="thetas", leave=None):
                fold = 0
                if sum((loaded_adj).flatten()>theta)>0:
                # print("Dataset limits are", np.ptp(X_DAG), np.ptp(X_test), np.ptp(y_test))
                    for train_idx, val_idx in tqdm(kf.split(X_DAG), desc="inner", leave=None):
                        # castle.__del__()
                        
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
                            castle.__del__()
                            castle = CASTLE(num_train = X_train.shape[0], num_inputs = X_train.shape[1], n_hidden=X_train.shape[1], 
                                            reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                                w_threshold = theta, ckpt_file = ckpt_file, tune = True, hypertrain = True, adj_mat=loaded_adj)
                            castle.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                    overwrite=False, tune=True, maxed_adj=None, tuned=False, verbose=False)
                        else:
                            castle = CASTLE(num_train = X_train.shape[0], num_inputs = X_train.shape[1], n_hidden=X_train.shape[1], 
                                            reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                                w_threshold = theta, ckpt_file = args.ckpt_file)
                            castle.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                    overwrite=force_refit, tune=False, maxed_adj=None, tuned=False, verbose=False)
                        ct = datetime.datetime.now() - start

                        W_est = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
                        save_pickle(W_est, os.path.join(folder, f"adjmats/W_est.{dset_sz}.{out_fold}.{fold}.{str(theta)}.{version}.pkl"), verbose=False)
                        # heat_mat(W_est)

                        G1 = nx.from_numpy_matrix(W_est, create_using=nx.DiGraph, parallel_edges=False)

                        if dag_type == 'fico':
                            preds = np.where(castle.pred(X_test) > 0.5, 1, 0).flatten() 
                            if verbose:
                                print(classification_report(y_test, preds, digits = 3))
                                print('Accuracy =',round(classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy'],3))
                            MSE_base = classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy']
                            REG_castle.append(MSE_base)

                        else:                    
                            REG_castle.append(mean_squared_error(castle.pred(X_test), y_test))
                            if verbose:
                                print("MSE = ", mean_squared_error(castle.pred(X_test), y_test))
                                if fold > 1:
                                    print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle))

                        score["timestamp"] = ct

                        if dag_type == 'fico':
                            preds = np.where(castle.pred(X_test) > 0.5, 1, 0).flatten() 
                            if verbose:
                                print(classification_report(y_test, preds, digits = 3))
                                print('Accuracy =',round(classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy'],3))
                            MSE_base = classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy']
                            score["accuracy"] = MSE_base

                            for std_val in [1,2,3]:
                                X_test_sub = X_test[np.where((X_test[:,0] < -std_val) | (X_test[:,0] > std_val))]
                                y_test_sub = y_test[np.where((y_test < -std_val) | (y_test >std_val))]
                                preds = np.where(castle.pred(X_test_sub) > 0.5, 1, 0).flatten() 

                                if len(y_test_sub) > 0 :
                                    score[f"accuracy_std_{std_val}"] = classification_report(y_test_sub, preds, digits = 3,output_dict=True)['accuracy']
                                else:
                                    score[f"accuracy_std_{std_val}"] = NaN
                                score[f"Test_size_{std_val}"] = X_test_sub.shape[0]
                        else:
                            score["MSE"] = mean_squared_error(castle.pred(X_test), y_test)

                            for std_val in [1,2,3]:
                                X_test_sub = X_test[np.where((X_test[:,0] < -std_val) | (X_test[:,0] > std_val))]
                                y_test_sub = y_test[np.where((y_test < -std_val) | (y_test >std_val))]
                                if len(y_test_sub) > 0 :
                                    score[f"MSE_std_{std_val}"] = mean_squared_error(castle.pred(X_test_sub), y_test_sub)
                                else:
                                    score[f"MSE_std_{std_val}"] = NaN
                                score[f"Test_size_{std_val}"] = X_test_sub.shape[0]


                        # score["dset"] = dset_sz
                        score["Test_size"] = X_test.shape
                        score["theta"] = theta
                        score["fold"] = fold

                        if dag_type != 'fico':
                            a = set(G.edges())
                            b = set(G1.edges())

                            import math
                            penalty_mis = -1
                            penalty_add = -1
                            penalty_dir = -1

                            n=G.number_of_nodes()
                            r=2
                            n_perm = math.factorial(n)/math.factorial(n-r)
                            if verbose:    
                                print("Equal sets:",a==b)
                                print("intersection:",a.intersection(b)   )
                                print("union:",a.union(b)        )
                                print("unequal:",a.union(b)  - a.intersection(b))   
                                print("Missing:",len(a-a.intersection(b)))
                                print("Added:",len(b-a.intersection(b)))

                            score["Missing edges"] = len(a-a.intersection(b))
                            score["Added edges"] = len(b-a.intersection(b))

                            score["DAG score"] = (n_perm + len(a-a.intersection(b))*penalty_mis + len(b-a.intersection(b))*penalty_add)/n_perm

                        # result_metrics_dict["dset=",dset_sz,"theta="+str(theta)+", fold="+str(fold)] = score
                        result_metrics_dict["out_fold="+str(out_fold)+"theta="+str(theta)+", fold="+str(fold)] = score

                        # print("theta:",theta, ", fold:",fold, ", MSE:",score['MSE'])

                        save_pickle(result_metrics_dict, out_file, verbose=False)

                def format_str(mean, std):
                    return "${0:.6f}".format(round(mean,6)) + " \pm {0:.6f}$    ".format(round(std,6))
                with open(args.output_log, "a") as logfile:
                    logfile.write(str(dset_sz) + ",  " + str(theta)  + ",  "+
                                format_str(np.mean(REG_castle), np.std(REG_castle)) + 
                                '\n')
                if verbose:    
                    print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle)) 