import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from numpy.core.numeric import NaN
np.set_printoptions(suppress=True)
import networkx as nx
import random
import pandas as pd
from signal import signal, SIGINT
import argparse
import datetime
from tqdm import tqdm

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score

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
    parser.add_argument('--gpu', help='Name of GPU to use',type = str, type = str, default = '')
    parser.add_argument('--csv', help='Name of the input csv, if available', type = str )
    parser.add_argument('--version', help='Tags all results',type = str, default='test')
    parser.add_argument('--output_log', help='Name of the .log file', type = str, default = 'test.log')
    parser.add_argument('--ckpt_file', help='Name of the .ckpt file', type = str, default = 'test.ckpt')
    parser.add_argument('--force_refit', help='Overwrite models in ckpt_file location', type=str2bool, nargs='?', const=True, default=True)    
    parser.add_argument('--overwrite_res', help='Overwrite summary results in results location', type=str2bool, nargs='?',const=True, default=False)    

    ## Data and model
    parser.add_argument("--random_dag", help='Bool, if True DAG is generated', type=str2bool, nargs='?',const=True, default=True)
    parser.add_argument('--num_nodes', help='Number of nodes |V| in synthetic DAG ', type = int, default = 20)
    parser.add_argument('--branchf', help='Proportion of edges over nodes in synthetic DAG e=|E|/|V|', type = float, default = 4)
    parser.add_argument('--noise_p', help='Proportion of noise variables (over |V|) to add to the data', type = float, default = 0.2)
    parser.add_argument('--known_p', help='Proportion of edges |E|) known in synthetic DAG', type = float, default = 0.2) #0.2
    parser.add_argument('--dataset_sz', help='Sample size |N|', type = int, default = 5000)
    parser.add_argument("--dataset_szs", help='Comma delimited list of sample sizes N', type=str, default='100000')
    parser.add_argument('--hidden_l', help='Number of hidden layers', type = int, default = 1) 
    parser.add_argument('--hidden_n_p', help='Multiplier (to |V|) for number of hidden neurons', type = float, default = 3.2) 
    parser.add_argument('--reg_lambda', help='Coefficient for R_DAG loss component', type = float, default = 1)
    parser.add_argument('--reg_beta', help='Coefficient for L1 component of R_DAG loss', type = float, default = 5)
    parser.add_argument('--lr', help='Learning Rate', type=float, default = 0.001)

    parser.add_argument('--seed', help='Seed to reproduce', type = int, default = 0)
    parser.add_argument('--out_folds', help='Number of outer folds in outer folds in nested xval', type = int, default = 5)
    parser.add_argument('--in_folds', help='Number of inner folds in outer folds in nested xval', type = int, default = 5)
    parser.add_argument("--thetas", help='Comma delimited list of thetas (taus), leave empty for autobinning of adjmat range', type=str, default='')
    parser.add_argument("--theta_range", help='Comma delimited list. Provide min, max, step of thetas (taus)', type=str, default='')
    parser.add_argument("--theta_auto", help='Bool, if True DAG theta interval is created linearly', type=str2bool, nargs='?',const=True, default=True)
    parser.add_argument("--verbose", help='Bool, if True prints debugging info', type=str2bool, nargs='?',const=True, default=False)

    args = parser.parse_args()
    signal(SIGINT, handler)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    verbose = args.verbose
    seed = args.seed
    version = args.version
    dag_type = version.split("_")[0] ## 'toy','random', 'fico', 'adult', 'boston'. If _partial/refine are provided different settings will apply
    DAG_inject = version.split("_")[0] ## One of 'full', 'toy', 'partial' or dataname

    if any(x in version for x in ['boston','cali']): 
        problem_type = 'reg' ## 'reg' or 'class' 
    elif any(x in version for x in ['fico','adult']): 
        problem_type = 'class' ## 'reg' or 'class' 

    folder = './results/'
    args.ckpt_file = os.path.join("./models/",version,version+'.ckpt')
    args.output_log = os.path.join("./logs/", version.split("_")[0], version+'.log')
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(os.path.join("./logs/", version.split("_")[0])):
        os.mkdir(os.path.join("./logs/", version.split("_")[0])) 

    dataset_szs = [int(item) for item in args.dataset_szs.split(',')]
    thetas= [float(item) for item in args.thetas.split(',') if item] 
    theta_interval = args.theta_auto 
    theta_min_max= [float(item) for item in args.theta_range.split(',') if item]

    random_stability(seed)
    if verbose:
        print(dag_type)
        print(args.lr)
    G, df = load_or_gen_data(name=dag_type, csv=args.csv)

    if len(dataset_szs) > 1:
        pbar1 = tqdm(dataset_szs, leave=None)
    else:
        pbar1 = dataset_szs

    ### Loop over sample sizes
    for dset_sz in pbar1:
        if len(dataset_szs) > 1:
            pbar1.set_description("Dset %s" % dset_sz)
            pbar1.refresh()
        
        out_file = os.path.join(folder,f"Nested{args.out_folds}FoldCASTLE.Reg.Synth.{dset_sz}.{version}.pkl")
        if os.path.exists(out_file):
            result_metrics_dict = load_pickle(out_file, verbose=False)
        else:
            result_metrics_dict = {}

        if args.out_folds >= 2:
            kf_out = KFold(n_splits = args.out_folds, random_state = seed, shuffle = True)
            kf_out_splits = kf_out.split(df)
        else:
            train, test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
            kf_out_splits = [tuple([train.index.to_numpy(), test.index.to_numpy()])]

        out_fold = 0
        pbar2 = tqdm(kf_out_splits, leave=None)

    ### Loop over outer folds in nested xval
        for train_idx, val_idx in  pbar2:
            out_fold += 1

            # if out_fold <=1: ###################################################################################################
            #     continue

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

            ## Adjust LR for very small samples
            if any(i in version for i in ['adult']):
                if dset_sz <= 200:
                    lr = 0.001
                    b_size = 32
                elif dset_sz <= 500:
                    lr = 0.0005
                    b_size = 32
                else:
                    lr = 0.0001
                    b_size = 32
            elif any(i in version for i in ['cali']):
                if dset_sz <= 500:
                    lr = 0.0012
                    b_size = 32
                else:
                    lr = 0.001
                    b_size = 32
            elif any(i in version for i in ['fico']):
                if dset_sz <= 200:
                    lr = 0.112
                    b_size = 32
                elif dset_sz <= 500:
                    lr = 0.005
                    b_size = 32
                else:
                    lr = 0.001
                    b_size = 32
            else:
                lr = 0.001
                b_size = 32

            print("LR=",lr)

            ## Create baseline for injection
            net = InjectedNet(num_inputs = X_DAG.shape[1], n_hidden=int(X_DAG.shape[1]*3.2), type_output=problem_type, 
                            reg_lambda = args.reg_lambda, reg_beta = args.reg_beta, lr=lr, batch_size=b_size,
                            w_threshold = 0, ckpt_file = ckpt_file, seed = seed)
            net.fit(X=X_DAG, y=y_DAG, num_nodes=X_DAG.shape[1], X_val=X_test, y_val=y_test,
                    overwrite=args.force_refit, inject=False, injected=False, verbose=False)

            W_est = net.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
            save_pickle(W_est, os.path.join(folder,f"adjmats/W_est.{dset_sz}.{out_fold}.{version}.pkl"), verbose=False)

            ## Evaluate baseline
            if problem_type == 'class':
                preds = np.where(net.pred(X_test) > 0.5, 1, 0).flatten() 
                accuracy = classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy']

                base_score = roc_auc_score(y_test, net.pred(X_test))

                if verbose:
                    print(classification_report(y_test, preds, digits = 3))
                    print('Accuracy =',round(accuracy,3))
                    print('AUC =',round(base_score,3))

            else:
                base_score = mean_squared_error(net.pred(X_test), y_test)

            with open(args.output_log, "a") as logfile:
                logfile.write(str(dset_sz) + ",  baseliine" ",  "+
                        "${0:.6f}$".format(round(base_score,6)) + 
                        '\n')

            ## Load or create Adjacency matric
            if DAG_inject in ['toy','full'] or 'size' in DAG_inject or 'partial' in version:
                loaded_adj = load_adj_mat(df, DAG_inject, version)
            elif DAG_inject == 'partial':
                n = int(len(G.edges())*args.known_p)
                known_edges = random.sample(list(G.edges()), n)
                if verbose:
                    print(len(known_edges), known_edges)
                loaded_adj = 1 - np.identity(args.num_nodes)
                for item in known_edges:
                    loaded_adj[item[1],item[0]]=0            
            elif DAG_inject not in ['toy','random']:
                loaded_adj = net.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
            
            ## Create list of thetas by binning, if not provided
            if len(thetas) == 0:
                if theta_interval and len(theta_min_max)==0:
                    min_mat = min(loaded_adj.flatten())
                    max_mat = max(loaded_adj.flatten())                       
                    t =  [[-1],list(np.arange(min_mat, max_mat, round((max_mat-min_mat)/20,3)))]
                    thetas =  [i for sublist in t for i in sublist]
                elif len(theta_min_max)>0:
                    theta_min = theta_min_max[0]
                    theta_max = theta_min_max[1]
                    theta_inter = theta_min_max[2]
                    if theta_interval:
                        if theta_min<0:
                            t =  [[-1],list(np.arange(0, theta_max, round(abs(0-theta_max)/20,3)))]
                            thetas =  [i for sublist in t for i in sublist]
                        else:
                            thetas =  list(np.arange(theta_min, theta_max, round((theta_min-theta_max)/20,3)))
                    else:
                        thetas =  list(np.round(np.arange(theta_min, theta_max, theta_inter),3)) 
                else:
                    raise ValueError("A theta interval or specific value(s) must be provided")

            pbar3 = tqdm(enumerate(thetas), leave=None)

            REG_net = []
    ### Loop over thetas
            for n_t, theta in pbar3:
                fold = 0
                # print(n_t, theta)
                pbar3.set_description("theta %s" % theta)
                pbar3.refresh()

                # if theta in [-1,0,0.08,0.12,0.16,0.2] and out_fold==1: ############################################################################################
                #     continue

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

                if args.in_folds >= 2:
                    kf = KFold(n_splits = args.in_folds, random_state = seed, shuffle = True)
                    kf_splits = kf.split(X_DAG)
                elif args.in_folds == 1:
                    train_idx = random.sample(range(len(X_DAG)), k=int(len(X_DAG)*0.8))
                    test_idx = [i for i in range(len(X_DAG)) if i not in train_idx]
                    kf_splits = [tuple([train_idx, test_idx])]
                elif args.in_folds == 0:
                    kf_splits = kf_out_splits

                pbar4 = tqdm(kf_splits, leave=None)

    ### Loop over inner folds of xval
                for train_idx, val_idx in pbar4:
                    
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

                    if 'refine' in version:
                        DAG_mat_0 = DAG_retreive_np(loaded_adj, theta)
                        loaded_adj = refine_mat(df, DAG_mat_0, version)

                    ## Fit Network, with or without injection
                    start = datetime.datetime.now()
                    if theta >= 0:
                        net.__del__()
                        net = InjectedNet(num_inputs = X_train.shape[1], n_hidden=int(X_train.shape[1]*3.2), 
                                        reg_lambda = args.reg_lambda, reg_beta = args.reg_beta, seed = seed,  type_output=problem_type, 
                                        lr=lr, batch_size=b_size, w_threshold = theta, ckpt_file = ckpt_file, 
                                        inject = True, adj_mat=loaded_adj)
                        net.fit(X=X_train, y=y_train, num_nodes=X_train.shape[1], X_val=X_val, y_val=y_val,
                                overwrite=args.force_refit, injected=False, inject=True, verbose=verbose)
                    else:
                        net = InjectedNet(num_inputs = X_train.shape[1], n_hidden=int(X_train.shape[1]*3.2), 
                                        reg_lambda = args.reg_lambda, reg_beta = args.reg_beta, seed = seed,  type_output=problem_type, 
                                        lr=lr, batch_size=b_size, w_threshold = theta, ckpt_file = args.ckpt_file)
                        net.fit(X=X_train, y=y_train, num_nodes=X_train.shape[1], X_val=X_val, y_val=y_val,
                                overwrite=args.force_refit, injected=False, inject=False, verbose=verbose)
                    ct = datetime.datetime.now() - start

                    W_est = net.pred_W(X_DAG, np.expand_dims(X_DAG[:,0], -1))
                    save_pickle(W_est, os.path.join(folder, f"adjmats/W_est.{dset_sz}.{out_fold}.{fold}.{str(theta)}.{version}.pkl"), verbose=False)

                    ######################################
                    ############ Evaluation ##############
                    ######################################

                    ## Predictive Performance
                    if problem_type == 'class':

                        base_score = roc_auc_score(y_test, net.pred(X_test))
                        REG_net.append(base_score)
                        score["auc"] = base_score

                        preds = np.where(net.pred(X_test) > 0.5, 1, 0).flatten() 
                        score['report'] = classification_report(y_test, preds, digits = 3,output_dict=True)
                        score["accuracy"] = classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy']
                        score['f1'] = classification_report(y_test, preds, digits = 3,output_dict=True)['weighted avg']['f1-score']

                        if verbose:
                            print(classification_report(y_test, preds, digits = 3))
                            print('Accuracy =',round(classification_report(y_test, preds, digits = 3,output_dict=True)['accuracy'],3))
                            print('AUC =',round(base_score,3))

                    else:                    
                        REG_net.append(mean_squared_error(net.pred(X_test), y_test))
                        if verbose:
                            print("MSE = ", mean_squared_error(net.pred(X_test), y_test))
                            if fold > 1:
                                print("MEAN =", np.mean(REG_net), "STD =", np.std(REG_net))
                        score["MSE"] = mean_squared_error(net.pred(X_test), y_test)

                        for std_val in [1,2,3]:
                            X_test_sub = X_test[np.where((X_test[:,0] < -std_val) | (X_test[:,0] > std_val))]
                            y_test_sub = y_test[np.where((y_test < -std_val) | (y_test >std_val))]
                            if len(y_test_sub) > 0 :
                                score[f"MSE_std_{std_val}"] = mean_squared_error(net.pred(X_test_sub), y_test_sub)
                            else:
                                score[f"MSE_std_{std_val}"] = NaN
                            score[f"Test_size_{std_val}"] = X_test_sub.shape[0]

                    score["timestamp"] = ct
                    # score["dset"] = dset_sz
                    score["Test_size"] = X_test.shape
                    score["theta"] = theta
                    score["fold"] = fold

                    # result_metrics_dict["dset=",dset_sz,"theta="+str(theta)+", fold="+str(fold)] = score
                    result_metrics_dict["out_fold="+str(out_fold)+", theta="+str(theta)+", fold="+str(fold)] = score

                    # print("theta:",theta, ", fold:",fold, ", MSE:",score['MSE'])

                    save_pickle(result_metrics_dict, out_file, verbose=False)

                ## Log Results
                def format_str(mean, std):
                    return "${0:.6f}".format(round(mean,6)) + "  ({0:.6f})$    ".format(round(std,6))
                with open(args.output_log, "a") as logfile:
                    logfile.write(str(dset_sz) + ",  " + str(theta)  + ",  "+
                                format_str(np.mean(REG_net), np.std(REG_net)) + 
                                '\n')
                if verbose:    
                    print("MEAN =", np.mean(REG_net), "STD =", np.std(REG_net)) 