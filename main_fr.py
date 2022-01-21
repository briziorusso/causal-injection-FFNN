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

import warnings
warnings.filterwarnings('always') ## to avoid classification report to complain

# tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# tf.config.list_physical_devices('GPU')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--version',default='test')
    parser.add_argument('--output_log', type = str, default = 'test.log')
    parser.add_argument('--ckpt_file', type = str, default = 'test.ckpt')
    parser.add_argument('--csv')

    parser.add_argument("--random_dag", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Activate nice mode.")

    parser.add_argument('--num_nodes', type = int, default = 20)
    parser.add_argument('--branchf', type = float, default = 4)
    parser.add_argument('--noise_p', type = float, default = 4)
    parser.add_argument('--dataset_sz', type = int, default = 5000)


    parser.add_argument('--out_folds', type = int, default = 5)
    parser.add_argument('--in_folds', type = int, default = 5)
    parser.add_argument('--reg_lambda', type = float, default = 1)
    parser.add_argument('--reg_beta', type = float, default = 5)
    parser.add_argument('--gpu', type = str, default = '')
    parser.add_argument('--extension', type = str, default = '')
       
    args = parser.parse_args()
    signal(SIGINT, handler)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    version = args.version

    prob_type = 'class' ## 'reg' or 'class'
    
    dag_type = version.split("_")[0] ## 'toy','random', 'fico', 'adult', 'boston'
    DAG_inject = version.split("_")[0] ## One of 'full', 'toy', 'partial', 'fico', 'fico_size'
    args.output_log = version+'.log'
    args.ckpt_file = os.path.join("./models/",version,version+'.ckpt')
    folder = './results/'
    
    force_refit = True
    verbose = False
    
    seed = 0
    known_perc = 0.2    
    dataset_szs = [100000]
    thetas = [] ### Leave empty for automatic binning of the adj_mat
    theta_interval = True ### True: binning is based on deciles, False: fixed intervales 0.001

    ####### fico2 ######
    # seed = 0
    # dataset_szs = [100000]
    # thetas = [] ### Leave empty for automatic binning of the adj_mat
    # theta_interval = False ### True: binning is based on deciles, False: fixed intervales 0.001
    # parser.add_argument('--dataset_sz', type = int, default = 5000)
    # parser.add_argument('--output_log', type = str, default = 'fico2.log')
    # parser.add_argument('--ckpt_file', type = str, default = 'fico2.ckpt')

    # parser.add_argument('--out_folds', type = int, default = 5)
    # parser.add_argument('--in_folds', type = int, default = 5)
    # parser.add_argument('--reg_lambda', type = float, default = 1)
    # parser.add_argument('--reg_beta', type = float, default = 5)

    ####### fico size #####
    # seed = 0
    # DAG_inject = 'fico_size' ## One of 'full', 'toy', 'partial', 'fico', 'fico_size'
    # thetas = [-1,0.006] ### Leave empty for automatic binning of the adj_mat
    # dataset_szs = [100,500,1000,2000,3000,4000,5000,6000,7000,8000]
    ##
    # dataset_szs = [args.dataset_sz]
    #[50,100,200,500,1000,2000,3000,4000,5000]
    # dataset_szs = [int(e*1.25) for e in dataset_szs]
    # parser.add_argument('--dataset_sz', type = int, default = 5000)
    # parser.add_argument('--output_log', type = str, default = 'fico_size.log')
    # parser.add_argument('--ckpt_file', type = str, default = 'fico_size.ckpt')

    # parser.add_argument('--out_folds', type = int, default = 5)
    # parser.add_argument('--in_folds', type = int, default = 5)
    # parser.add_argument('--reg_lambda', type = float, default = 1)
    # parser.add_argument('--reg_beta', type = float, default = 5)

    random_stability(seed)

    G, df = load_or_gen_data(name=dag_type, csv=None)

    if len(dataset_szs) > 1:
        pbar1 = tqdm(dataset_szs, leave=None)
    else:
        pbar1 = dataset_szs

    for dset_sz in pbar1:
        if len(dataset_szs) > 1:
            pbar1.set_description("Dset %s" % dset_sz)
            pbar1.refresh()
        
        out_file = os.path.join(folder,f"Nested{args.out_folds}FoldCASTLE.Reg.Synth.{dset_sz}.{version}.pkl")
        if os.path.exists(out_file):
            result_metrics_dict = load_pickle(out_file, verbose=False)
        else:
            result_metrics_dict = {}

        kf_out = KFold(n_splits = args.out_folds, random_state = seed, shuffle = True)
        out_fold = 0
        pbar2 = tqdm(kf_out.split(df), leave=None)

        for train_idx, val_idx in  pbar2:
            out_fold += 1

            pbar2.set_description("out_fold %s" % out_fold)
            pbar2.refresh()

            ## Data pre-processing
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
            if dag_type not in ['toy','random'] :
                    X_DAG = X_DAG.to_numpy()
                    X_test = df.loc[val_idx].to_numpy()

            y_test = X_test[:,0]

            ckpt_file = args.ckpt_file + "_ds" + str(dset_sz)

            ## create baseline for tuning
            castle = CASTLE(num_train = X_DAG.shape[0], num_inputs = X_DAG.shape[1], n_hidden=int(X_DAG.shape[1]*3.2), 
                            reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                            w_threshold = 0, ckpt_file = ckpt_file, seed = seed)
            castle.fit(X=X_DAG, y=y_DAG, num_nodes=np.shape(X_DAG)[1], X_val=X_test, y_val=y_test,
                    overwrite=force_refit, tune=False, maxed_adj=None, tuned=False, verbose=False)

            W_est = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
            save_pickle(W_est, os.path.join(folder,f"adjmats/W_est.{dset_sz}.{out_fold}.{version}.pkl"), verbose=False)

            ## Evaluate results
            if prob_type == 'class':
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

            ## Load or create Adjacency matric
            if DAG_inject in ['toy','full','fico_size']:
                loaded_adj = load_adj_mat(DAG_inject)
            elif DAG_inject == 'partial':
                n = int(len(G.edges())*known_perc)
                known_edges = random.sample(list(G.edges()), n)
                if verbose:
                    print(len(known_edges), known_edges)
                loaded_adj = 1 - np.identity(args.num_nodes)
                for item in known_edges:
                    loaded_adj[item[1],item[0]]=0            
            elif DAG_inject not in ['toy','random']:
                loaded_adj = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))

            
            ## Create list of thetas by binning, if not provided
            if len(thetas) == 0:
                min_mat = min(loaded_adj.flatten())
                max_mat = max(loaded_adj.flatten())                
                if theta_interval:
                    t =  [[-1],list(np.arange(min_mat, max_mat, round((max_mat-min_mat)/20,3)))]
                    thetas =  [i for sublist in t for i in sublist]
                else:
                    thetas =  list(np.arange(0.026, max_mat, 0.001)) ####################################remember to change 0.026 back to min_mat
            # print('thetas to do:', len(thetas))

            kf = KFold(n_splits = args.in_folds, random_state = seed, shuffle = True)

            pbar3 = tqdm(enumerate(thetas), leave=None)

            REG_castle = []
            for n_t, theta in pbar3:
                fold = 0
                # print(n_t, theta)
                pbar3.set_description("theta %s" % theta)
                pbar3.refresh()

                total_edg = sum((loaded_adj).flatten()>theta)
                ## Make sure there are edges
                if total_edg==0:
                    continue                
                if n_t>0:
                    last_theta = thetas[n_t-1]

                    total_edg_last = sum((loaded_adj).flatten()>last_theta)

                    ## Make sure changing theta had an effect
                    if total_edg == total_edg_last:
                        continue

                pbar4 = tqdm(kf.split(X_DAG), leave=None)

                for train_idx, val_idx in pbar4:
                    # castle.__del__()
                    
                    score = {}
                    fold += 1

                    pbar4.set_description("in_fold %s" % fold)
                    pbar4.refresh()

                    if verbose:    
                        print(f"out_fold ={out_fold}, fold={fold}")
                        print("******* Doing dataset size T= ", len(train_idx),", V=", len(val_idx) , "****************")

                    ## Split data again
                    X_train = X_DAG[train_idx]
                    y_train = np.expand_dims(X_DAG[train_idx][:,0], -1)
                    X_val = X_DAG[val_idx]
                    y_val = X_DAG[val_idx][:,0]

                    ## Fit Network, with or without injection
                    start = datetime.datetime.now()
                    if theta >= 0:
                        castle.__del__()
                        castle = CASTLE(num_train = X_train.shape[0], num_inputs = X_train.shape[1], n_hidden=int(X_train.shape[1]*3.2), 
                                        reg_lambda = args.reg_lambda, reg_beta = args.reg_beta, seed = seed,
                                            w_threshold = theta, ckpt_file = ckpt_file, tune = True, hypertrain = True, adj_mat=loaded_adj)
                        castle.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                overwrite=force_refit, tuned=False, tune=True, maxed_adj=loaded_adj, verbose=verbose)
                    else:
                        castle = CASTLE(num_train = X_train.shape[0], num_inputs = X_train.shape[1], n_hidden=int(X_train.shape[1]*3.2), 
                                        reg_lambda = args.reg_lambda, reg_beta = args.reg_beta, seed = seed,
                                            w_threshold = theta, ckpt_file = args.ckpt_file)
                        castle.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                overwrite=force_refit, tune=False, maxed_adj=None, tuned=False, verbose=verbose)
                    ct = datetime.datetime.now() - start

                    W_est = castle.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
                    save_pickle(W_est, os.path.join(folder, f"adjmats/W_est.{dset_sz}.{out_fold}.{fold}.{str(theta)}.{version}.pkl"), verbose=False)
                    # heat_mat(W_est)

                    if prob_type == 'class':
                        preds = np.where(castle.pred(X_test) > 0.5, 1, 0).flatten() 
                        if verbose:
                            print(classification_report(y_test, preds, digits = 3))
                            print('Accuracy =',round(classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy'],3))
                        MSE_base = classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy']
                        REG_castle.append(MSE_base)
                        score["accuracy"] = MSE_base

                        # for std_val in [1,2,3]:
                        #     X_test_sub = X_test[np.where((X_test[:,0] < -std_val) | (X_test[:,0] > std_val))]
                        #     y_test_sub = y_test[np.where((y_test < -std_val) | (y_test >std_val))]
                        #     preds = np.where(castle.pred(X_test_sub) > 0.5, 1, 0).flatten() 

                        #     if len(y_test_sub) > 0 :
                        #         score[f"accuracy_std_{std_val}"] = classification_report(y_test_sub, preds, digits = 3,output_dict=True)['accuracy']
                        #     else:
                        #         score[f"accuracy_std_{std_val}"] = NaN
                        #     score[f"Test_size_{std_val}"] = X_test_sub.shape[0]
                    else:                    
                        REG_castle.append(mean_squared_error(castle.pred(X_test), y_test))
                        if verbose:
                            print("MSE = ", mean_squared_error(castle.pred(X_test), y_test))
                            if fold > 1:
                                print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle))
                        score["MSE"] = mean_squared_error(castle.pred(X_test), y_test)

                        for std_val in [1,2,3]:
                            X_test_sub = X_test[np.where((X_test[:,0] < -std_val) | (X_test[:,0] > std_val))]
                            y_test_sub = y_test[np.where((y_test < -std_val) | (y_test >std_val))]
                            if len(y_test_sub) > 0 :
                                score[f"MSE_std_{std_val}"] = mean_squared_error(castle.pred(X_test_sub), y_test_sub)
                            else:
                                score[f"MSE_std_{std_val}"] = NaN
                            score[f"Test_size_{std_val}"] = X_test_sub.shape[0]

                    score["timestamp"] = ct
                    # score["dset"] = dset_sz
                    score["Test_size"] = X_test.shape
                    score["theta"] = theta
                    score["fold"] = fold

                    ## Evaluate concordance to known DAG
                    if dag_type in ['toy', 'random']:
                        G1 = nx.from_numpy_matrix(W_est, create_using=nx.DiGraph, parallel_edges=False)
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
                    result_metrics_dict["out_fold="+str(out_fold)+", theta="+str(theta)+", fold="+str(fold)] = score

                    # print("theta:",theta, ", fold:",fold, ", MSE:",score['MSE'])

                    save_pickle(result_metrics_dict, out_file, verbose=False)

                ## Log Results
                def format_str(mean, std):
                    return "${0:.6f}".format(round(mean,6)) + " \pm {0:.6f}$    ".format(round(std,6))
                with open(args.output_log, "a") as logfile:
                    logfile.write(str(dset_sz) + ",  " + str(theta)  + ",  "+
                                format_str(np.mean(REG_castle), np.std(REG_castle)) + 
                                '\n')
                if verbose:    
                    print("MEAN =", np.mean(REG_castle), "STD =", np.std(REG_castle)) 