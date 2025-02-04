from keras.layers import Dense, Multiply, \
    Embedding, Activation, Conv1D, GlobalMaxPooling1D, Embedding, \
    Dropout, TimeDistributed, Layer
from keras import backend as K
from keras import optimizers

import numpy as np
import os
import tensorflow as tf
from utils import create_dataset_from_score_st_ai2tale_gt
import pickle
import scipy.io as io
import keras
import tensorflow_probability as tfp

"""
for example, using CUDA_VISIBLE_DEVICES=1 python thefilewewanttorun.py agurmentParser
"""
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

"""
tf2.x: tf.random.set_seed(10086)
"""
tf.random.set_seed(10086)
np.random.seed(10086)


def shuffle_aligned_list(data):
    """Shuffle arrays in a list by shuffling each array identically."""
    num = data[0].shape[0]
    np.random.seed(123)
    p = np.random.permutation(num)
    return [d[p] for d in data]


def load_data_v2(dir_path):
    """
    Load data.
    """
    with open(dir_path + 'ind_word.pickle', 'rb') as f:
        word_index = pickle.load(f)

    ind_word_vocab_size_path = dir_path + 'ind_word_vocab_size_m' + '.mat'
    ind_word_vocab_size = io.loadmat(ind_word_vocab_size_path)
    vocab_size = ind_word_vocab_size['vocab_size'][0][0]

    data_tr_path = dir_path + 'x_y_tr_m' + '.mat'
    data_tr = io.loadmat(data_tr_path)

    X_tr_non = data_tr['X_tr_non']
    X_tr_non_len = data_tr['X_tr_non_len']
    y_tr_non = np.reshape(data_tr['y_tr_non'], [-1])

    X_tr_vul = data_tr['X_tr_vul']
    X_tr_vul_len = data_tr['X_tr_vul_len']
    y_tr_vul = np.reshape(data_tr['y_tr_vul'], [-1])

    data_te_path = dir_path + 'x_y_te_m' + '.mat'
    data_te = io.loadmat(data_te_path)

    X_te = data_te['X_te']
    X_te_len = data_te['X_te_len']
    y_te = np.reshape(data_te['y_te'], [-1])

    X_train = np.concatenate((X_tr_non, X_tr_vul))
    X_len_train = np.concatenate((X_tr_non_len, X_tr_vul_len))
    y_train = np.concatenate((y_tr_non, y_tr_vul))

    X_train, X_len_train, y_train = shuffle_aligned_list([X_train, X_len_train, y_train])
    
    X_te, X_te_len, y_te = shuffle_aligned_list([X_te, X_te_len, y_te])

    return {"X_train": X_train, "y_train": y_train, "X_te": X_te, "y_te": y_te,
            "X_len_train": X_len_train, "X_te_len": X_te_len,
            "word_index": word_index, "vocab_size": vocab_size}


class SampleConcrete(Layer):
    def __init__(self, args, **kwargs):
        self.tau = args.tau
        super(SampleConcrete, self).__init__(**kwargs)

    def call(self, logits):
        lo_gits_ = K.permute_dimensions(logits, (0, 2, 1))

        uni_shape = K.shape(logits)[0]
        uniform_a = K.random_uniform(shape=(uni_shape, 1, args.slen),
                                     minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)
        uniform_b = K.random_uniform(shape=(uni_shape, 1, args.slen),
                                     minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)
        gumbel_a = -K.log(-K.log(uniform_a))
        gumbel_b = -K.log(-K.log(uniform_b))

        no_z_lo_gits = K.exp((gumbel_a + lo_gits_) / self.tau)
        de_z_lo_gits = no_z_lo_gits + K.exp((gumbel_b + (1.0 - lo_gits_)) / self.tau)

        samples = no_z_lo_gits / de_z_lo_gits

        logits = tf.reshape(lo_gits_, [-1, lo_gits_.shape[-1]])
        threshold = tf.expand_dims(tf.nn.top_k(logits, args.k, sorted=True)[0][:, -1], -1)
        discrete_logits = tf.cast(tf.greater_equal(logits, threshold), tf.float32)

        return K.in_train_phase(K.permute_dimensions(samples, (0, 2, 1)), tf.expand_dims(discrete_logits, -1))

    def compute_output_shape(self, input_shape):
        return input_shape


class Random_Bernoulli_Sampler(Layer):
    '''
    Layer to Sample r
    '''

    def __init__(self, **kwargs):
        super(Random_Bernoulli_Sampler, self).__init__(**kwargs)

    def call(self, logits):
        batch_size = tf.shape(logits)[0]
        d = tf.shape(logits)[1]

        u = tf.random.uniform(shape=(batch_size, d), 
                              minval=np.finfo(tf.float32.as_numpy_dtype).tiny, maxval=1.0)

        r = K.cast(tf.stop_gradient(u > 0.5), tf.float32)
        r = tf.expand_dims(r, -1)

        return r
    
    def compute_output_shape(self, input_shape):
        return input_shape


class Embeddings(tf.keras.Model):
    def __init__(self, MAX_NUM_WORDS, embedding_dim, max_sent_length):
        super().__init__()
        self.output_dim = 250
        self.embedding_layer = Embedding(MAX_NUM_WORDS,
                                         embedding_dim,
                                         input_length=max_sent_length,
                                         name='embedding',
                                         trainable=True)
        self.dropout = Dropout(0.2)
        self.conv1d = Conv1D(self.output_dim, 3,
                             padding='valid',
                             activation='relu',
                             strides=1)
        self.maxpooling1d = GlobalMaxPooling1D()

    def call(self, inputs):
        embedded_sequences = self.embedding_layer(inputs)
        net = self.dropout(embedded_sequences)
        net = self.conv1d(net)
        net = self.maxpooling1d(net)
        return net

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class Sum(Layer):
    def __init__(self, axis=1, **kwargs):
        self.axis = axis
        super(Sum, self).__init__(**kwargs)

    def call(self, inputs):
        return K.sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape


class Predictor(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.multiply = Multiply()
        self.sum = Sum(250)

        self.dense_cf = Dense(250)
        self.act_cf = Activation('relu')
        self.pred_cf = Dense(2, activation='softmax', name='new_dense')

        self.bernoulli_sampling = Random_Bernoulli_Sampler()

    def call(self, embedded_sequences, logits_T, T, joint=True):
        """
        embedded_sequences and T (from r or z) are the inputs of predictor.
        """
        if joint is True:
            selected_encoding = self.multiply([embedded_sequences, T])
        else:
            T_bernoulli = self.bernoulli_sampling(logits_T)
            T_bernoulli = tf.squeeze(T_bernoulli)
            sc_T_bernoulli = tfp.distributions.RelaxedBernoulli(0.5, logits=T_bernoulli)
            sc_T_bernoulli = tf.expand_dims(sc_T_bernoulli.sample(), axis=-1)
            
            selected_encoding = self.multiply(
                [embedded_sequences, sc_T_bernoulli])

        net = self.sum(selected_encoding)
        net = self.dense_cf(net)
        net = self.act_cf(net)
        preds = self.pred_cf(net)

        return preds


class Selector(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.drop_1 = Dropout(0.20, name='dropout_1')
        self.dense_1 = Dense(100, name='new_dense_1', activation='relu')
        
        self.drop_2 = Dropout(0.20, name='dropout_2')
        self.dense_2 = Dense(100, name='new_dense_2', activation='relu')
        
        self.dense_logits = Dense(1, name='new_dense_logits', activation='sigmoid')

        self.sampleconcrete = SampleConcrete(args)

    def call(self, embedded_sequences):
        """
        embedded_sequences is the input of selector.
        """
        net = self.drop_1(embedded_sequences)
        net = self.dense_1(net)

        net = self.drop_2(net)
        net = self.dense_2(net)

        logits_T = self.dense_logits(net)
        T = self.sampleconcrete(logits_T)

        return logits_T, T


class MyModel(tf.keras.Model):
    def __init__(self, MAX_NUM_WORDS, args):
        super().__init__()
        self.MAX_NUM_WORDS = MAX_NUM_WORDS
        self.embedding_dim = args.emb
        self.max_sent_length = args.slen

        self.embeddings = Embeddings(self.MAX_NUM_WORDS, self.embedding_dim, self.max_sent_length)
        self.encoding = TimeDistributed(self.embeddings)

        self.selector = Selector(args)

        self.joint = True
        self.predictor = Predictor()

    def call(self, inputs):
        embedded_sequences = self.encoding(inputs)

        logits_T, T = self.selector(embedded_sequences)
        preds = self.predictor(embedded_sequences, logits_T, T, self.joint)

        return embedded_sequences, logits_T, T, preds


def AI2TALE(data_set, home_dir, agrs):
    """
    print('Loading dataset...')
    """
    x_train, x_val, y_train, y_val = data_set['X_train'], data_set['X_te'], data_set['y_train'], data_set['y_te']
    word_index, vocabulary_size = data_set['word_index'], data_set['vocab_size']
    x_val_len = data_set['X_te_len']
    MAX_NUM_WORDS = vocabulary_size

    print('Creating model...')
    mymodel = MyModel(MAX_NUM_WORDS, args)
    opt = optimizers.Adam(learning_rate=agrs.lr, clipnorm=1.0)
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(args.batch).shuffle(buffer_size=1024)
    test_data = tf.data.Dataset.from_tensor_slices((x_val[:x_val.shape[0]//2], y_val[:y_val.shape[0]//2])).batch(args.batch)
    valid_data = tf.data.Dataset.from_tensor_slices((x_val[x_val.shape[0]//2:], y_val[y_val.shape[0]//2:])).batch(args.batch)

    if args.train:
        val_acc_max = 0.0
        saved_model_dir = home_dir + 'sent_models_xai_ai2tale/' + str(agrs.sig) + '_' + str(agrs.lr) + '_' + str(agrs.lam) + '/'
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)
        history_file.write(str(agrs.sig) + '-' + str(agrs.lr) + '-' + str(agrs.lam) + '\n')

        # for training and saving the model
        for epoch in range(args.epo):
            history_file.write("epoch: %d \n" % epoch)
            print("epoch: %d" % epoch)
            for batch_no, (data, labels) in enumerate(train_data):
                with tf.GradientTape(persistent=True) as tape_predictor:
                    # for Embeddings and Predictor parts only, not for Selector
                    mymodel.joint = False
                    mymodel.selector.trainable = False

                    embedded_sequences, logits_T, _, predicts = mymodel(data, training=True)
                    loss = loss_fn(labels, predicts)

                    variables = mymodel.trainable_variables
                    gradients = tape_predictor.gradient(loss, variables)
                    opt.apply_gradients(zip(gradients, variables))

                with tf.GradientTape(persistent=True) as tape_whole:
                    # for the whole model including Embeddings, Selector and Predictor
                    mymodel.joint = True
                    mymodel.selector.trainable = True
                    embedded_sequences, logits_T, _, predicts = mymodel(data, training=True)

                    loss = loss_fn(labels, predicts)

                    sigma = agrs.sig
                    x_embed = K.square(K.mean(embedded_sequences, axis=-1))
                    x_embed = tf.expand_dims(x_embed, axis=1)

                    sigma_b = (1.0 / (2.0 * K.square(sigma)))
                    w = tf.math.scalar_mul(sigma_b, logits_T)
                    w = tf.transpose(w, [0, 2, 1])

                    loss_re = tf.reshape(tf.matmul(w, x_embed, transpose_b=True), [-1])
                    loss_re = loss_re + args.slen * (K.log(sigma) + 0.5 * K.square(sigma))

                    total_loss = loss + agrs.lam*tf.reduce_mean(loss_re)

                    variables = mymodel.trainable_variables
                    gradients = tape_whole.gradient(total_loss, variables)
                    opt.apply_gradients(zip(gradients, variables))

                    train_acc_metric.update_state(labels, predicts)

                    # log every 10 batches.
                    if batch_no % 10 == 0:
                        history_file.write("Training loss at step %d -- loss: %.4f \n" % (batch_no, float(loss)))
                        history_file.write("Seen so far: %d samples \n" % ((batch_no + 1) * args.batch))
                        print("Training loss at step %d -- loss: %.4f \n" % (batch_no, float(loss)))
                        print("Seen so far: %d samples" % ((batch_no + 1) * args.batch))

            # display acc metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            history_file.write("Training acc over epoch: %.4f \n" % (float(train_acc)))
            print("Training acc over epoch: %.4f" % (float(train_acc)))
            train_acc_metric.reset_states()

            # run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in valid_data:
                embedded_sequences, logits_T, T, predicts = mymodel(x_batch_val, training=False)
                val_acc_metric.update_state(y_batch_val, predicts)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            history_file.write("Validation acc: %.4f \n" % (float(val_acc)))
            print("Validation acc: %.4f" % (float(val_acc)))

            if val_acc_max < val_acc:
                val_acc_max = val_acc
                print('saving a model')
                mymodel.save(saved_model_dir + 'my_network_model')

    else:
        saved_model_dir = home_dir + 'sent_models_xai_ai2tale/' + str(agrs.sig) + '_' + str(agrs.lr) + '_' + str(agrs.lam) + '/'
        if not os.path.exists(saved_model_dir):
            print('cannot find out the saved models')
            return None, None, None, None, None, None, None, None, None
        else:
            """
            tf.saved_model.load vs keras.models.load_model
            """
            history_file.write(str(agrs.sig) + '-' + str(agrs.lr) + '-' + str(agrs.lam) + '\n')

            selection_network = tf.saved_model.load(saved_model_dir + 'my_network_model')

            val_y_list = np.array([])
            val_logits_T_list = np.array([])
            val_x_list = np.array([])
            val_predicts_list = np.array([])

            # Execute the testing loop after loading the trained model.
            for x_batch_val, y_batch_val in test_data:
                embedded_sequences, logits_T, T, predicts = selection_network(x_batch_val, training=False)

                val_y_list = np.append(val_y_list, y_batch_val)
                val_logits_T_list = np.append(val_logits_T_list, logits_T)
                val_x_list = np.append(val_x_list, x_batch_val)
                val_predicts_list = np.append(val_predicts_list, predicts)

                val_acc_metric.update_state(y_batch_val, predicts)

            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()

            history_file.write("Test acc: %.4f \n" % (float(val_acc)))
            print("Test acc: %.4f" % (float(val_acc)))

            return saved_model_dir, val_logits_T_list, val_x_list, word_index, val_y_list, None, val_predicts_list, args.k, x_val_len


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='AI2TALE')
    parser.add_argument('--train', action='store_true', help='store_true will default to True when the command-line argument is present and vice versa')
    parser.add_argument('--data', type=str, default='./phishingdata/') 
    parser.add_argument('--home', type=str, default='ai2tale', help='The folder for saving the model')
    parser.add_argument('--tau', type=float, default=0.5, help='The tau value used in the Gumbel-Softmax distribution')
    parser.add_argument('--k', type=int, default=1, help='The number of selected sentence in each email data sample')
    parser.add_argument('--lr', type=str, default=0.001, help='The learning rate used in the training process')
    parser.add_argument("--epo", type=int, default=10, help="The number of epochs used to train the model")
    parser.add_argument("--batch", default=128, type=int, help="The number of data in each batch used to train the mode.")
    parser.add_argument('--sig', type=float, default=0.3, help='The variance value of the prior distribution mentioned in Eq.(8)')
    parser.add_argument('--lam', type=float, default=0.1, help='The trade-off hyperparameter described in Eq.(9)')
    parser.add_argument('--slen', type=int, default=100, help='The length of sentences in each email')
    parser.add_argument('--tlen', type=int, default=50, help='The length of tokens in each sentence')
    parser.add_argument('--emb', type=int, default=150, help='The embedding dimension')
    args = parser.parse_args()

    print('Loading data...')
    dataset = load_data_v2(args.data)

    if args.task == 'AI2TALE':
        home_dir = args.home + '_k' + str(args.k) + '_tau' + str(args.tau) + '/'
        result_dir =  home_dir + 'history_logs_sent/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        if args.train:
            history_file = open(result_dir + '_train_his_sent_threshold.txt', 'w')
            AI2TALE(dataset, home_dir, args)
            history_file.close()
        else:
            history_file = open(result_dir + '_test_his_sent_threshold.txt', 'w')
        
            saved_dir, scores, x_val, word_index, y_val, max_epoch, interp_val, selected_sents, x_val_len = AI2TALE(dataset, home_dir, args)
            if saved_dir != None:
                print('Creating data set with selected sentences...')
                scores = np.reshape(scores, [-1, args.slen])
                x_val = np.reshape(x_val, [-1, args.slen, args.tlen])
                interp_val = np.reshape(interp_val, [-1, 2])
                create_dataset_from_score_st_ai2tale_gt(
                    saved_dir, scores, x_val, y_val, max_epoch, word_index, selected_sents, interp_val, x_val_len)
                
            history_file.close()
