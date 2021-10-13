## Much of the code is taken from https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/alg/castle/CASTLE.py 
## This has been adapted to perform causal injection into a joint network beyond the regularization performed within CASTLE

import numpy as np
np.set_printoptions(suppress=True)
import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import os
# from emals.utils import *
import networkx as nx

from utils import random_stability, DAG_retrieve

# this allows wider numpy viewing for matrices
np.set_printoptions(linewidth=np.inf)

class CASTLE(object): ## change name...
    def __init__(
        self,
        num_train,
        lr=None,
        batch_size=32,
        num_inputs=1,
        num_outputs=1,
        type_output='reg',
        w_threshold=0.3,
        n_hidden=32,
        hidden_layers=2, #not used
        ckpt_file='tmp.ckpt',
        standardize=True,
        reg_lambda=None,
        reg_beta=None,
        DAG_min=0.5,
        max_steps = 200,
        seed = 0,
        tune=False,
        adj_mat = None,
        hypertrain = False,
        verbose = False
    ):
        random_stability(seed)

        self.verbose = verbose
        self.count = 0
        self.max_steps = max_steps
        self.saves = 50
        self.patience = 30
        self.metric = mean_squared_error  ## not used...
        self.w_threshold = w_threshold  # threshold for the acceptance of weights, below are set to 0
        self.DAG_min = DAG_min ## not used

        if lr is None:
            initial_learning_rate = 0.001
        else:
            initial_learning_rate = lr

        ## Not implemented as not accepted by Adam in tf1.15
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps = 100000,
            decay_rate = 0.96,
            staircase = True)

        if reg_lambda is None:
            self.reg_lambda = 1.  # coefficient for the R_DAG loss
        else:
            self.reg_lambda = reg_lambda

        if reg_beta is None:
            self.reg_beta = 1  # coefficient for the L1 regularizer part of the R_DAG loss
        else:
            self.reg_beta = reg_beta

        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.n_hidden = n_hidden
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.X = tf.placeholder("float", [None, self.num_inputs])
        self.y = tf.placeholder("float", [None, 1])
        self.rho = tf.placeholder("float", [1, 1])
        self.alpha = tf.placeholder("float", [1, 1])
        self.keep_prob = tf.placeholder("float")
        self.Lambda = tf.placeholder("float")
        self.noise = tf.placeholder("float")
        self.is_train = tf.placeholder(tf.bool, name="is_train")

        # One-hot vector indicating which nodes are trained
        self.sample = tf.placeholder(tf.int32, [self.num_inputs])
        
        # Hypertraining parameter
        ### comment out because we handle it externally in the xval loop
        # self.theta = tf.Variable(tf.random.uniform(shape=(1,), minval=0.2, maxval=0.3), name='theta')
        # self.theta = tf.get_variable('theta',
        #     dtype=tf.float32,
        #     shape=(1,),
        #     initializer=tf.random_uniform_initializer(minval=0.1, maxval=0.4),
        #     constraint=lambda z: tf.clip_by_value(z, 0.1, 0.4)
        #     )
        self.theta = self.w_threshold

        # Store layers weight & bias
        self.weights = {}
        self.biases = {}
        
        # Create the input and output weight matrix for each feature
        for i in range(self.num_inputs):
            self.weights['w_h0_'+str(i)] = tf.Variable(tf.random_normal([self.num_inputs, self.n_hidden], seed = seed)*0.01) #why make it smaller?
            self.weights['out_'+str(i)] = tf.Variable(tf.random_normal([self.n_hidden, self.num_outputs], seed = seed))
            
        for i in range(self.num_inputs):
            self.biases['b_h0_'+str(i)] = tf.Variable(tf.random_normal([self.n_hidden], seed = seed)*0.01)
            self.biases['out_'+str(i)] = tf.Variable(tf.random_normal([self.num_outputs], seed = seed))


        # The first and second layers are shared
        self.weights.update({'w_h1': tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))})
        self.biases.update({'b_h1': tf.Variable(tf.random_normal([self.n_hidden]))})

        #         self.weights.update({
        #             'w_h2': tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden], seed = 2))
        #         })
        #         self.biases.update({
        #             'b_h2': tf.Variable(tf.random_normal([self.n_hidden], seed = 2))
        #         })

        self.hidden_h0 = {}
        self.hidden_h1 = {}
        self.hidden_h2 = {}
        self.layer_1 = {}
        self.layer_1_dropout = {}
        self.out_layer = {}

        self.Out_0 = []

        # Mask removes the feature i from the network that is tasked to construct feature i
        self.mask = {}
        self.activation = tf.nn.relu
        self.out_activation = tf.math.sigmoid

        if tune and hypertrain:
            adj_mat = DAG_retrieve(adj_mat, self.theta)

        for i in range(self.num_inputs):
            
            if tune and adj_mat is not None:
                if i==0:
                    if verbose:
                        print("Masking non causal dependencies")
                if tf.is_tensor(adj_mat):
                    indices = tf.transpose(tf.tile(tf.where(tf.equal(tf.gather(adj_mat, i, axis=1),0)),[1,self.n_hidden]))
                else:
                    indices = [np.where(adj_mat[:,i]==0)[0].tolist()] * self.n_hidden

                self.mask[str(i)] = tf.math.reduce_prod(tf.transpose(
                    tf.one_hot(indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1)
                    ),axis=1)
            else:
                indices = [i] * self.n_hidden
                self.mask[str(i)] = tf.transpose(
                    tf.one_hot(indices, depth=self.num_inputs, on_value=0.0, off_value=1.0, axis=-1)
                )

            self.weights['w_h0_' + str(i)] = self.weights['w_h0_' + str(i)] * self.mask[str(i)]
            
            ## ReLU activation for hidden layers
            self.hidden_h0['nn_' + str(i)] = self.activation(
                tf.add(tf.matmul(self.X, self.weights['w_h0_' + str(i)]), self.biases['b_h0_' + str(i)])
            )
            self.hidden_h1['nn_' + str(i)] = self.activation(
                tf.add(tf.matmul(self.hidden_h0['nn_' + str(i)], self.weights['w_h1']), self.biases['b_h1'])
            )
            
            #             self.hidden_h2['nn_'+str(i)] = self.activation(tf.add(tf.matmul(self.hidden_h1['nn_'+str(i)], self.weights['w_h2']), self.biases['b_h2']))
            
            if i == 0 and type_output != 'reg': # binary prediction for y @ df[,0] hence sigmoid activation to output probabilities
                self.out_layer['nn_' + str(i)] = self.out_activation(
                    tf.add(
                        tf.matmul(self.hidden_h1['nn_' + str(i)], self.weights['out_' + str(i)]), self.biases['out_' + str(i)]
                    )
                )
            else:
                self.out_layer['nn_' + str(i)] = tf.add(
                    tf.matmul(self.hidden_h1['nn_' + str(i)], self.weights['out_' + str(i)]), self.biases['out_' + str(i)]
                )
            self.Out_0.append(self.out_layer['nn_' + str(i)])

        # Concatenate all the constructed features
        self.Out = tf.concat(self.Out_0, axis=1)

        ### Supervised loss definition ###
        if type_output == 'reg':
            # Supervised loss: MSE for regression
            self.supervised_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.out_layer['nn_0'] - self.y),axis=1),axis=0)
        elif type_output == 'class':
            ## Binary crossentropy for classification
            bce = tf.keras.losses.BinaryCrossentropy()
            self.supervised_loss = bce(self.y, self.out_layer['nn_0'])
        else:
            raise Exception(f'Invalid problem type! ({type_output})')

        self.W_0 = []
        for i in range(self.num_inputs):
            self.W_0.append(tf.math.sqrt(tf.reduce_sum(tf.square(self.weights['w_h0_' + str(i)]), axis=1, keepdims=True)))
        # output weights are the square root of the sum of the first layer weights across the different sub networks.

        self.W = tf.concat(self.W_0, axis=1)

        ## R_W
        coff = 1.0

        d = tf.cast(self.X.shape[1], tf.float32)
        dag_l = tf.cast(d, tf.float32)
        Z_in = tf.eye(d)
        Z = tf.multiply(self.W, self.W)

        for i in range(1, 10):  #truncated power series

            Z_in = tf.matmul(Z_in, Z)

            dag_l += 1. / coff * tf.linalg.trace(Z_in)
            coff = coff * (i + 1)

        self.h = dag_l - tf.cast(d, tf.float32)

        ## V_W group lasso??
        L1_loss = 0.0
        L1_alpha = 0.5
        for i in range(self.num_inputs): 
            #             print('weigths',self.weights['w_h0_'+str(i)])
            w_1 = tf.slice(self.weights['w_h0_' + str(i)], [0, 0], [i, -1])
            #             print('w_1',w_1)
            w_2 = tf.slice(self.weights['w_h0_' + str(i)], [i + 1, 0], [-1, -1])
            #             print('w_2',w_2)

            if tune and adj_mat is not None:

                if tf.is_tensor(adj_mat):
                    causaleffects = tf.reshape(tf.where(tf.not_equal(tf.gather(self.W, i, axis=1),0)),shape=[-1])
                    w_1_mask = tf.gather(causaleffects,tf.where(tf.less(causaleffects,i)))
                    w_2_mask = tf.subtract(tf.gather(causaleffects,tf.where(tf.greater(causaleffects,i))),i+1)
                else:      
                    causaleffects = np.where(adj_mat[:,i] != 0)[0].tolist()
                    w_1_mask = [c for c in causaleffects if c < i]
                    w_2_mask = [c-i-1 for c in causaleffects if c > i]

                w_1 = tf.gather(w_1,tf.cast(w_1_mask, dtype='int32'))
                w_2 = tf.gather(w_2,tf.cast(w_2_mask, dtype='int32'))

            L1_loss += tf.reduce_sum(tf.norm(w_1, axis=1)) + tf.reduce_sum(tf.norm(w_2, axis=1))
#             L1_loss += L1_alpha*(tf.math.sqrt(tf.cast(w_1.shape[0], tf.float32))*tf.reduce_sum(tf.norm(w_1, ord=2 ))+tf.math.sqrt(tf.cast(w_2.shape[0], tf.float32))*tf.reduce_sum(tf.norm(w_2, ord=2 ))) + (1-L1_alpha)*(tf.reduce_sum(tf.norm(w_1, ord=1 ))+tf.reduce_sum(tf.norm(w_2, ord=1 )))
#             print('L1_loss',L1_loss)

        # Residuals
        self.R = self.X - self.Out

        # Divide the residual into untrain and train subset
        _, subset_R = tf.dynamic_partition(tf.transpose(self.R), partitions=self.sample, num_partitions=2)
        subset_R = tf.transpose(subset_R)
        
        # Minimise loss difference after tuning
        

        #Combine all the loss
        ## Reconstruction loss
        self.mse_loss_subset = tf.cast(self.num_inputs, tf.float32) / tf.cast(tf.reduce_sum(
            self.sample
        ), tf.float32) * tf.reduce_sum(tf.square(subset_R))

        ## R_DAG loss = L_W + beta*V_W + R_W
        self.regularization_loss_subset = self.Lambda * (
            self.mse_loss_subset + self.reg_beta * L1_loss + 0.5 * self.rho * self.h * self.h + self.alpha * self.h
        )

        #Add in supervised loss
        self.regularization_loss_subset += self.Lambda * self.rho * self.supervised_loss

        self.optimizer_subset = tf.train.AdamOptimizer(learning_rate=#self.learning_rate, 
        initial_learning_rate,
        beta1=0.9, beta2=0.999, epsilon=1e-4)
        self.loss_op_dag = self.optimizer_subset.minimize(self.regularization_loss_subset)

        ###############################################
        #### Extract gradients (not fully working) ####

        # gradients, variables = zip(*self.optimizer_subset.compute_gradients(self.regularization_loss_subset))
        
        # def replace_none_with_zero(l):
        #     return [0.0 if i==None else i for i in l]

        # gradients = replace_none_with_zero(gradients)

        # # gradients = tf.Print(gradients,[gradients],"Gradients")

        # gradient_av = tf.math.reduce_mean( tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0))
        # gradient_av = tf.print(gradient_av,
        #     [gradient_av],
        #     "Mean")
        # gradient_max = tf.math.reduce_max( tf.concat([tf.reshape(g, [-1]) for g in [c for c in gradients if c is not None]], axis=0))
        # gradient_max = tf.print(gradient_max,
        #     [gradient_max],
        #     "Max")
        # gradient_min = tf.math.reduce_min( tf.concat([tf.reshape(g, [-1]) for g in [c for c in gradients if c is not None]], axis=0))
        # gradient_min = tf.print(gradient_min,
        #     [gradient_min],
        #     "Min")
        # gradient_stats = [gradient_min,gradient_av,gradient_max]
        
        # gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        # # gradients, _ = tf.clip_by_value([c for c in gradients if c is not None], 1e-8, 1e1)

        # self.loss_op_dag = self.optimizer_subset.apply_gradients(zip(gradients, variables))
        ############################################

        #############################################
        ## initialise regularisation loss
        # self.regularization_loss = 0

        # self.loss_op_supervised = self.optimizer_subset.minimize(
        #     self.supervised_loss + self.regularization_loss  # this is set to 0
        # )
        #############################################
        # config=tf.ConfigProto(     
        #     intra_op_parallelism_threads=14,     
        #     inter_op_parallelism_threads=14)
        # self.sess = tf.Session(config=config)
        self.sess = tf.Session()
        if not tune:
            self.sess.run(tf.global_variables_initializer())
        # self.saver = tf.train.Saver(var_list=tf.global_variables())
        self.saver = tf.train.Saver()
        self.tmp = ckpt_file

    def __del__(self):
        tf.reset_default_graph()
        if self.verbose:
            print("Destructor Called... Cleaning up")
        self.sess.close()
        del self.sess

    def gaussian_noise_layer(self, input_layer, std):
        noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    def __fit__(self, X, y, num_nodes, X_val, y_val, seed = 0, verbose=True):
        from random import sample
        random_stability(seed)

        rho_i = np.array([[1.0]])
        alpha_i = np.array([[1.0]])

        best = 1e9
        best_value = 1e9
        for step in range(1, self.max_steps):
            h_value, loss = self.sess.run(
                [self.h, self.supervised_loss],
                feed_dict={
                    # self.W: W,
                    self.X: X,
                    self.y: y,
                    self.keep_prob: 1,
                    self.rho: rho_i,
                    self.alpha: alpha_i,
                    self.is_train: True,
                    self.noise: 0
                }
            )
            if verbose:
                print(
                    "Step " + str(step) + ", Loss= " + "{:.4f}".format(loss) +
                    " h_value:",
                    h_value
                )

            for step1 in range(1, (X.shape[0] // self.batch_size) + 1):

                idxs = random.sample(range(X.shape[0]), self.batch_size)
                batch_x = X[idxs]
                batch_y = np.expand_dims(batch_x[:, 0], -1)
                one_hot_sample = [0] * self.num_inputs
                subset_ = sample(range(self.num_inputs), num_nodes)
                for j in subset_:
                    one_hot_sample[j] = 1
                self.sess.run(
                    self.loss_op_dag,
                    feed_dict={
                        # self.W: W,
                        self.X: batch_x,
                        self.y: batch_y,
                        self.sample: one_hot_sample,
                        self.keep_prob: 1,
                        self.rho: rho_i,
                        self.alpha: alpha_i,
                        self.Lambda: self.reg_lambda,
                        self.is_train: True,
                        self.noise: 0
                    }
                )

            val_loss = self.val_loss(X_val, y_val)
            if verbose:
                print("Val Loss= " + "{:.4f}".format(val_loss))

            if val_loss < best_value:
                best_value = val_loss

            if step >= self.saves:
                try:
                    if val_loss < best:
                        best = val_loss
                        self.saver.save(self.sess, self.tmp)
                        if verbose:
                            print("Saving model")
                        self.count = 0
                    else:
                        self.count += 1
                except:
                    print("Error caught in calculation")

            if self.count > self.patience:
                if verbose:
                    print("Early stopping")
                break

        self.saver.restore(self.sess, self.tmp)

    def fit(self, X, y, num_nodes, X_val, y_val, 
            overwrite=False, tune=False, maxed_adj=None, tuned=False, seed = 0, verbose=True):            

        file_path = self.tmp + ".data-00000-of-00001"

        if os.path.exists(file_path) and not overwrite and not tune and not tuned:
            self.saver.restore(self.sess, self.tmp)
            if verbose:
                print("Model Loaded from ", self.tmp)

        elif os.path.exists(file_path) and tuned:
            # tf.reset_default_graph()
            self.tmp = self.tmp + "_tuned"

            file_path1 = self.tmp + ".data-00000-of-00001"

            if os.path.exists(file_path1):       
                self.saver.restore(self.sess, self.tmp)
                if verbose:
                    print("Model Loaded from ", self.tmp)
            else:
                print("No tuned model found. Set tuned==False and tune==True")

        elif os.path.exists(file_path) and tune:
            self.saver.restore(self.sess, self.tmp)
            if verbose:
                print("Model Loaded from ", self.tmp)

            self.tmp = self.tmp + "_tuned"

            if verbose:
                print("Begin Tuning - Apply Mask from DAG")

            self.__fit__(X, y, num_nodes, X_val, y_val, seed = seed, verbose=True)

        else:
            self.__fit__(X, y, num_nodes, X_val, y_val, seed = seed, verbose=True)

            # W_est = self.sess.run(
            #     self.W,
            #     feed_dict={
            #         self.X: X,
            #         self.y: y,
            #         self.keep_prob: 1,
            #         self.rho: rho_i,
            #         self.alpha: alpha_i,
            #         self.is_train: True,
            #         self.noise: 0
            #     }
            # )
            # W_est[np.abs(W_est) < self.w_threshold] = 0

    def val_loss(self, X, y):
        if len(y.shape) < 2:
            y = np.expand_dims(y, -1)
        from random import sample
        one_hot_sample = [0] * self.num_inputs

        # use all values for validation
        subset_ = sample(range(self.num_inputs), self.num_inputs)
        for j in subset_:
            one_hot_sample[j] = 1

        return self.sess.run(
            self.supervised_loss,
            feed_dict={
                self.X: X,
                self.y: y,
                self.sample: one_hot_sample,
                self.keep_prob: 1,
                self.rho: np.array([[1.0]]),
                self.alpha: np.array([[0.0]]),
                self.Lambda: self.reg_lambda,
                self.is_train: False,
                self.noise: 0
            }
        )

    def pred(self, X):
        return self.sess.run(
            self.out_layer['nn_0'], 
            feed_dict={
                self.X: X,
                self.keep_prob: 1,
                self.is_train: False,
                self.noise: 0
            }
        )

    def get_weights(self, X, y):
        return self.sess.run(
            self.W,
            feed_dict={
                self.X: X,
                self.y: y,
                self.keep_prob: 1,
                self.rho: np.array([[1.0]]),
                self.alpha: np.array([[0.0]]),
                self.is_train: False,
                self.noise: 0
            }
        )

    def pred_W(self, X, y):
        W_est = self.sess.run(
            self.W,
            feed_dict={
                self.X: X,
                self.y: y,
                self.keep_prob: 1,
                self.rho: np.array([[1.0]]),
                self.alpha: np.array([[0.0]]),
                self.is_train: False,
                self.noise: 0
            }
        )
        return W_est #np.round_(W_est, decimals=3)

    def get_h(self, X):
        return self.sess.run(self.h, feed_dict={self.X: X, self.keep_prob: 1, self.is_train: False, self.noise: 0})

    def get_W0(self, X, y):
        return self.sess.run(
            [self.weights['w_h0_0'], self.biases['b_h0_0']],
            feed_dict={
                self.X: X,
                self.y: y,
                self.keep_prob: 1,
                self.rho: np.array([[1.0]]),
                self.alpha: np.array([[0.0]]),
                self.is_train: False,
                self.noise: 0
            }
        )

    def get_W1(self, X, y):
        return self.sess.run(
            [self.weights['w_h1'], self.biases['b_h1']],
            feed_dict={
                self.X: X,
                self.y: y,
                self.keep_prob: 1,
                self.rho: np.array([[1.0]]),
                self.alpha: np.array([[0.0]]),
                self.is_train: False,
                self.noise: 0
            }
        )

    def get_Wout(self, X, y):
        return self.sess.run(
            [self.weights['out_0'], self.biases['out_0']],
            feed_dict={
                self.X: X,
                self.y: y,
                self.keep_prob: 1,
                self.rho: np.array([[1.0]]),
                self.alpha: np.array([[0.0]]),
                self.is_train: False,
                self.noise: 0
            }
        )


## For DBX

    def predict_proba(self, X):
        return self.sess.run(
            self.out_layer['nn_0'], feed_dict={
                self.X: X,
                self.keep_prob: 1,
                self.is_train: False,
                self.noise: 0
            }
        )

    def predict_proba_x(self, X, Y):
        i = X.columns.get_loc(Y)
        return self.sess.run(
            self.out_layer['nn_' + str(i)], feed_dict={
                self.X: X,
                self.keep_prob: 1,
                self.is_train: False,
                self.noise: 0
            }
        )

    def get_all_influences(self, X, y, threshold, k_top):
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
        
        def select_top_k(mat, k):
            thresholded_mat = mat.copy()
            for i in range(mat.shape[1]):
                threshold = sorted(mat[:,i])[-k]
                mask_i = (mat[:,i]>=threshold).astype(int)
                mask = np.full(mat.shape,1)
                mask[:,i] = mask_i 
                thresholded_mat=thresholded_mat*mask
            return thresholded_mat

        adj_matrix = self.sess.run(
            self.W,
            feed_dict={
                self.X: X,
                self.y: y,
                self.keep_prob: 1,
                self.rho: np.array([[1.0]]),
                self.alpha: np.array([[0.0]]),
                self.is_train: False,
                self.noise: 0
            }
        )

        ## maxing is not done here anymore but externally
#         maxed_adj = zero_under_t(max_over_diag(adj_matrix), threshold)

#         if k_top is not None:
#             maxed_adj = select_top_k(maxed_adj, k_top)

        G1 = nx.from_numpy_matrix(adj_matrix, create_using=nx.DiGraph, parallel_edges=False)

        ##Debug
        print("Influence Threshold:",threshold)
        print("Total Number of Edges in G:", G1.number_of_edges())
        print("Max in degree:", max(sorted([d for n, d in G1.in_degree()], reverse=True)))
        print("DAG:", nx.is_directed_acyclic_graph(G1))

        return G1

    def get_adj_mat(self, X, y, threshold, k_top, disP = True):

        adj_matrix = self.sess.run(
            self.W,
            feed_dict={
                self.X: X,
                self.y: y,
                self.keep_prob: 1,
                self.rho: np.array([[1.0]]),
                self.alpha: np.array([[0.0]]),
                self.is_train: False,
                self.noise: 0
            }
        )

        import seaborn as sns
        cm = sns.light_palette("#003E74", as_cmap=True)
        x=pd.DataFrame(adj_matrix).round(3)
        x.style.background_gradient(cmap=cm, low=-0.1, high=0.1).format("{:.3}")        
        if disP:
            display(x)
        
        return adj_matrix