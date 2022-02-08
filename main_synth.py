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
from sklearn.metrics import mean_squared_error, mean_absolute_error

from net_inject import InjectedNet
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

    ## Run arguments
    parser.add_argument('--gpu', type = str, default = '')
    parser.add_argument('--extension', type = str, default = '')    
    parser.add_argument('--version', default='test')
    parser.add_argument('--csv')
    parser.add_argument('--output_log', type = str, default = 'InjectedNet_recon.log')
    parser.add_argument('--ckpt_file', type = str, default = 'recon.ckpt')
    parser.add_argument('--force_refit', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")
    parser.add_argument('--overwrite_res', type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Activate nice mode.")
    ## Data and model
    parser.add_argument("--random_dag", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Activate nice mode.")
    parser.add_argument('--num_nodes', type = int, default = 10) #20
    parser.add_argument('--branchf', type = float, default = 2) #4
    parser.add_argument('--known_p', type = float, default = 0.2) #0.2
    parser.add_argument('--noise_p', type = float, default = 0.0) #0.2
    parser.add_argument('--dataset_sz', type = int, default = 5000)
    parser.add_argument('--hidden_l', type = int, default = 1) #20
    parser.add_argument('--hidden_n_p', type = float, default = 3.2) #20
    parser.add_argument('--reg_lambda', type = float, default = 1)
    parser.add_argument('--reg_beta', type = float, default = 5)

    ## Experiment strategy
    parser.add_argument('--out_folds', type = int, default = 1)
    parser.add_argument('--in_folds', type = int, default = 1)

    args = parser.parse_args()
    signal(SIGINT, handler)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # if args.gpu == '':
    #     args.gpu = '/CPU:0'

    # with tf.device(args.gpu):

    runs_list_theta = check_run_status(args.version)

    folder = "./results/"
    if not os.path.exists(folder):
        os.mkdir(folder)    
        
    args.output_log = os.path.join("./logs/", args.version, args.output_log)
    if not os.path.exists(os.path.join("./logs/", args.version)):
        os.mkdir(os.path.join("./logs/", args.version))

    # args.csv = 'synth_nonlinear.csv' ## toy
    DAG_inject = 'partial' ## One of 'full', 'toy', 'partial'
    verbose = False
    standardise = True

    thetas = [-1, 0.05]
    seeds_list = [0, 10, 100, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
    nodes_list = [10,20,50]
    alphas = [50, 100, 200, 300, 500]

    pbar = tqdm(seeds_list, leave=None)
    for seed in pbar:
        pbar.set_description("Seed %s" % seed)
        pbar.refresh()

        random_stability(seed)

        pbar2 = tqdm(nodes_list, leave=None)
        for num_nodes in pbar2:
            pbar2.set_description("Nodes %s" % num_nodes)
            pbar2.refresh()

            hidden_n = int(num_nodes*args.hidden_n_p)
            hidden_l = args.hidden_l
            
            dataset_szs = [int(num_nodes*a*1.25) for a in alphas]

            if args.random_dag:
                dag_type = 'random'
            else:
                dag_type = 'toy'

            G, df = load_or_gen_data(name=dag_type, num_nodes=num_nodes, branchf=args.branchf, csv=args.csv)

            pbar3 = tqdm(dataset_szs, leave=None)
            for dset_sz in pbar3:
                random_stability(seed)
                alpha = dset_sz*0.8/num_nodes

                pbar3.set_description("alpha %s" % alpha)
                pbar3.refresh()

                ## Check if the run is already in store
                current_run_code = str(seed)+str(num_nodes)+str(int(alpha))
                if all([i in runs_list_theta for i in [current_run_code+"-1.0",current_run_code+"0.05"]]) and args.overwrite_res==False:
                    continue

                if args.out_folds >= 2:
                    kf_out = KFold(n_splits = args.out_folds, random_state = seed, shuffle = True)
                    kf_out_splits = kf_out.split(df)
                else:
                    train, test = train_test_split(df, test_size=0.2, random_state=seed)
                    kf_out_splits = [tuple([train.index.to_numpy(), test.index.to_numpy()])]

                out_fold = 0
                out_file = os.path.join(folder,f"Nested{args.out_folds}FoldCASTLE.Reg.Synth.{num_nodes}.{dset_sz}.{args.version}.pkl")
                if os.path.exists(out_file):
                    result_metrics_dict = load_pickle(out_file, verbose=False)
                else:
                    result_metrics_dict = {}

                for train_idx, val_idx in  kf_out_splits:
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

                    net = None
                    if current_run_code+"0.05" not in runs_list_theta:

                        ## Do not refit existing model for baseline castle if seed, folds and model structure is the same
                        ckpt_file = os.path.join("./models/common_seed_size", args.ckpt_file + 
                                                        "_seed" + str(seed) + "_of" + str(out_fold) + 
                                                        "_nodes" + str(num_nodes) + "_b" + str(args.branchf) + 
                                                        "_ds" + str(dset_sz) + "_np" + str(args.noise_p) +
                                                        "_hl" + str(args.hidden_l) + "_hn" + str(args.hidden_n_p)
                                                        )
                        ## Create baseline for injection
                        net = InjectedNet(num_inputs = X_DAG.shape[1], n_hidden=hidden_n, hidden_layers=hidden_l,
                                        reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                        w_threshold = 0, ckpt_file = ckpt_file, seed = seed)
                        net.fit(X=X_DAG, y=y_DAG, num_nodes=np.shape(X_DAG)[1], X_val=X_test, y_val=y_test,
                                overwrite=False, inject=False, injected=False, verbose=verbose)

                        out_file_mat = os.path.join(folder,f"adjmats/baselines/W_est.{seed}.{out_fold}.{num_nodes}.{args.branchf}.{dset_sz}.{args.noise_p}.{args.hidden_l}.{args.hidden_n_p}.pkl")
                        if os.path.exists(out_file_mat):
                            W_est = load_pickle(out_file_mat, verbose=False)
                        else:
                            if not os.path.exists(os.path.join(folder,"adjmats/baselines/")):
                                os.mkdir(os.path.join(folder,"adjmats/baselines/"))
                            W_est = net.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
                            save_pickle(W_est, os.path.join(out_file_mat), verbose=False)

                            MSE_base = mean_squared_error(net.pred(X_test), y_test)
                            with open(args.output_log, "a") as logfile:
                                logfile.write("baseliine" + ",  seed"+ str(seed)+ ",  out_fold" + str(out_fold) + 
                                                ",  num_nodes" + str(num_nodes) + ",  p_edges" + str(args.branchf) +
                                                ",  dset_sz"+ str(dset_sz) + ",  noise_p" + str(args.noise_p) + 
                                                ",  hidden_l"  + str(args.hidden_l) + ",  hidden_n_p" + str(args.hidden_n_p) +
                                        ",  performance" + "${0:.6f}$".format(round(MSE_base,6)) + 
                                        '\n')

                    ## Load or create Adjacency matric
                    if DAG_inject in ['toy','full'] or 'size' in DAG_inject:
                        loaded_adj = load_adj_mat(DAG_inject)
                    elif DAG_inject == 'partial':
                        n = int(len(G.edges())*args.known_p)
                        known_edges = random.sample(list(G.edges()), n)
                        if verbose:
                            print(len(known_edges), known_edges)
                        loaded_adj = 1 - np.identity(num_nodes)
                        for item in known_edges:
                            loaded_adj[item[1],item[0]]=0            
                    elif DAG_inject not in ['toy','random']:
                        loaded_adj = net.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
                    
                    if args.in_folds >= 2:
                        kf = KFold(n_splits = args.in_folds, random_state = seed, shuffle = True)
                        kf_splits = kf.split(X_DAG)
                    elif args.in_folds == 1:
                        train_idx = random.sample(range(len(X_DAG)), k=int(len(X_DAG)*0.8))
                        test_idx = [i for i in range(len(X_DAG)) if i not in train_idx]
                        kf_splits = [tuple([train_idx, test_idx])]
                    elif args.in_folds == 0:
                        kf_splits = kf_out_splits
                                    
                    REG_net = []
                    for theta in tqdm(thetas, desc="thetas", leave=None):
                        fold = 0

                        ## Check if the run is already in store
                        current_run_code = str(seed)+str(num_nodes)+str(int(alpha))+str(theta)
                        if current_run_code in runs_list_theta and args.overwrite_res==False:
                            continue

                        for train_idx, val_idx in kf_splits:
                            if net:
                                net.__del__()

                            score = {}
                            fold += 1

                            if verbose:    
                                print(f"out_fold ={out_fold}, fold={fold}")
                                print("******* Doing dataset size T= ", len(train_idx),", V=", len(val_idx) , "****************")

                            X_train = X_DAG[train_idx]
                            y_train = np.expand_dims(X_DAG[train_idx][:,0], -1)
                            X_val = X_DAG[val_idx]
                            y_val = X_DAG[val_idx][:,0]

                            ## Do not refit existing model for normal castle if seed, folds and model structure is the same
                            ckpt_file1 = os.path.join("./models/", "castle", args.ckpt_file + 
                                                        "_seed" + str(seed) + "_of" + str(out_fold) + "_if" + str(fold) + 
                                                        "_nodes" + str(num_nodes) + "_b" + str(args.branchf) + 
                                                        "_ds" + str(dset_sz) + "_np" + str(args.noise_p) +
                                                        "_hl" + str(args.hidden_l) + "_hn" + str(args.hidden_n_p)
                                                        )

                            start = datetime.datetime.now()
                            if theta >= 0:
                                net = InjectedNet(num_inputs = X_train.shape[1], n_hidden=hidden_n, hidden_layers=hidden_l,
                                                reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                                    w_threshold = theta, ckpt_file = ckpt_file, inject = True, adj_mat=loaded_adj)
                                net.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                        overwrite=True, injected=False, inject=True, verbose=verbose)
                            else:
                                net = InjectedNet(num_inputs = X_train.shape[1], n_hidden=hidden_n, hidden_layers=hidden_l,
                                                reg_lambda = args.reg_lambda, reg_beta = args.reg_beta,
                                                    w_threshold = theta, ckpt_file = ckpt_file1)
                                net.fit(X=X_train, y=y_train, num_nodes=np.shape(X_train)[1], X_val=X_val, y_val=y_val,
                                        overwrite=args.force_refit, injected=False, inject=False, verbose=verbose)
                            ct = datetime.datetime.now() - start

                            W_est = net.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
                            save_pickle(W_est, os.path.join(folder, f"adjmats/W_est.{num_nodes}.{seed}.{dset_sz}.{out_fold}.{fold}.{str(theta)}.{args.version}.pkl"), verbose=False)

                            REG_net.append(mean_squared_error(net.pred(X_test), y_test))
                            if verbose:
                                print("MSE = ", mean_squared_error(net.pred(X_test), y_test))
                                if fold > 1:
                                    print("MEAN =", np.mean(REG_net), "STD =", np.std(REG_net))

                            score["timestamp"] = ct

                            score["theta"] = theta
                            score["fold"] = fold
                            score["n_nodes"] = len(G.nodes)
                            score['N_edges'] = len(G.edges)
                            score["data_size"] = dset_sz
                            score["seed"] = seed

                            score["MSE"] = mean_squared_error(net.pred(X_test), y_test)
                            score["MAE"] = mean_absolute_error(net.pred(X_test), y_test)
                            score["Test_size"] = X_test.shape

                            for std_val in [1,2,3]:
                                X_test_sub = X_test[np.where((X_test[:,0] < -std_val) | (X_test[:,0] > std_val))]
                                y_test_sub = y_test[np.where((y_test < -std_val) | (y_test >std_val))]
                                if len(y_test_sub) > 0 :
                                    score[f"MSE_std_{std_val}"] = mean_squared_error(net.pred(X_test_sub), y_test_sub)
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

                                fooled_p = 0
                                fooled_edges = 0
                                causenoise = 0
                                if args.noise_p>0:
                                    len_noisy = int(len(G.nodes)*args.noise_p)
                                    ## find if there are edges from or to random nodes
                                    causenoise = np.sum(maxed_adj[ :, -len_noisy:] > 0)
                                    causedbynoise = np.sum(maxed_adj[-len_noisy: , :] > 0)
                                    fooled_p = (causenoise+causedbynoise)/((df.shape[1]*len_noisy-2)*2)
                                    fooled_edges = causenoise+causedbynoise

                                scores.append((tau,DAG_score, len(G1.edges()), len(a-a.intersection(b)), len(b-a.intersection(b)), len(a.intersection(b)), len(a.intersection(b))/len(G.edges()) , fooled_p, fooled_edges, causenoise))

                            c_tau = max([a[1] for a in scores])   
                            max_tau =  [a[0] for a in scores if a[1] ==c_tau][0]
                            Dscore =    [a[1] for a in scores if a[1] ==c_tau][0]
                            total =    [a[2] for a in scores if a[1] ==c_tau][0]
                            missing =  [a[3] for a in scores if a[1] ==c_tau][0]
                            added =    [a[4] for a in scores if a[1] ==c_tau][0]
                            matching = [a[5] for a in scores if a[1] ==c_tau][0]
                            perfect =    round(len([a[1] for a in scores if a[1] ==c_tau and a[1] ==1])/len([a[1] for a in scores if a[1] ==c_tau]),2)
                            right = [a[6] for a in scores if a[1] ==c_tau][0]
                            wrong = 1-right
                            fooled = [a[7] for a in scores if a[1] ==c_tau][0]
                            fooled_e = [a[8] for a in scores if a[1] ==c_tau][0]
                            causenoise = [a[9] for a in scores if a[1] ==c_tau][0]

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
                            score['fooled'] = fooled
                            score['fooled_e'] = fooled_e
                            score['causenoise'] = causenoise
                               
                            result_metrics_dict[ "theta="+str(theta)+", num_nodes="+str(num_nodes) + ", data_sz=" + str(dset_sz) +", seed="+str(seed)+", out_fold="+str(out_fold)+", fold="+str(fold)] = score
                
                            save_pickle(result_metrics_dict, out_file, verbose=False)

                        def format_str(mean, std):
                            return "${0:.6f}".format(round(mean,6)) + " ({0:.6f})$ ".format(round(std,6))
                        with open(args.output_log, "a") as logfile:
                                    logfile.write("theta" +str(theta) + ",  seed"+ str(seed)+ ",  out_fold" + str(out_fold) + 
                                        ",  num_nodes" + str(num_nodes) + ",  p_edges" + str(args.branchf) +
                                        ",  dset_sz"+ str(dset_sz) + ",  noise_p" + str(args.noise_p) + 
                                        ",  hidden_l"  + str(args.hidden_l) + ",  hidden_n_p" + str(args.hidden_n_p) +
                                        ",  performance" + format_str(np.mean(REG_net), np.std(REG_net)) + 
                                        '\n')
                        if verbose:    
                            print("MEAN =", np.mean(REG_net), "STD =", np.std(REG_net)) 

