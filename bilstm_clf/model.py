import json

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.layers import Bidirectional, LSTM
from tqdm import tqdm_notebook


class BiLSTMClassifier:
    def __init__(self, config_path='build_config.json', is_training=True, pad_idx=0, chkp_dir='.'):
        self.config_path = config_path
        self.is_training = is_training
        self.pad_idx = pad_idx
        self.chkp_dir = chkp_dir
        tf.reset_default_graph()
        self.global_step = tf.train.get_or_create_global_step()
        self.build_model()
        self.extra_vars_to_save = [(self.global_step.op.name, self.global_step), (self.lr.op.name, self.lr)]
        self.trainable_variables = dict(
            self.extra_vars_to_save + [(x.op.name, x) for x in
                                       tf.trainable_variables()])
        self.saver = tf.train.Saver(self.trainable_variables)

    @staticmethod
    def create_config(config_path, config):
        with open(config_path, 'w') as f:
            json.dump(config, f)
        return config

    def build_model(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        max_seq_len = config['max_seq_len']
        hidden_size = config['hidden_size']
        voc_size = config['voc_size']
        num_layers = config['num_layers']
        num_classes = config['num_classes']
        lr = config['learning_rate']

        self.dropout = tf.placeholder(tf.float32, shape=[], name='dropout')
        self.keep_prob = 1.0 - self.dropout
        self.inputs = tf.placeholder(tf.int32, shape=[None, max_seq_len], name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=[None, max_seq_len], name='targets')
        self.weights = tf.placeholder(tf.float32, shape=[None, max_seq_len], name='weights')

        self.lr = tf.get_variable(initializer=lr, trainable=False, name='learning_rate')

        with tf.device("/cpu:0"):
            embeddings = tf.get_variable(
                "embeddings", [voc_size, hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embeddings, self.inputs)
            if self.is_training:
                inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        for i in range(num_layers):
            if i == 0:
                model = Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape=(max_seq_len, hidden_size))(
                    inputs)
            else:
                if self.is_training:
                    model = tf.nn.dropout(model, keep_prob=self.keep_prob)
                model = Bidirectional(LSTM(hidden_size, return_sequences=True),
                                      input_shape=(max_seq_len, 2 * hidden_size))(
                    model)
        self.dense = tf.get_variable('dense', [2 * hidden_size, num_classes], dtype=tf.float32)

        model = tf.reshape(model, [-1, 2 * hidden_size])
        self.model = tf.reshape(tf.matmul(model, self.dense), [-1, max_seq_len, num_classes])
        self.predictions = tf.argmax(self.model, axis=-1)

        if not self.is_training:
            return

        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.model,
                                                     targets=self.targets,
                                                     weights=self.weights,
                                                     average_across_timesteps=True,
                                                     average_across_batch=True,
                                                     name='loss')

        self.accuracy = tf.metrics.accuracy(self.targets, self.predictions, weights=self.weights, name='accuracy')
        if config['optimizer'].lower() == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config['optimizer'].lower() == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif config['optimizer'].lower() == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.lr)
        elif config['optimizer'].lower() == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        else:
            raise ValueError()

        self.train = self.optimizer.minimize(self.loss, global_step=self.global_step, name='train')

    def fit(self, inputs, targets, batch_size=30, epochs=5, from_chkp=None, dropout=0.2, with_lr=False, save_per_epoch=10, validation_split=0.2, validation_step=100):
        ep_acc, acc, i = 0., 0., 0
        extra_batch = 0 if len(X) % batch_size == 0 else 1
        n_batchs = len(X) // batch_size
        X = np.copy(inputs)
        y = np.copy(targets)
        border = int(len(X) * validation_split)
        X_tr, X_vl = X[border:], X[:border]
        y_tr, y_vl = y[border:], y[:border]
        validation_losses = {}
        with tf.Session() as sess:
            # try:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            if from_chkp is not None:
                self.saver.restore(sess, self.chkp_dir.rstrip('/') + f'/{from_chkp}')
            for epoch in range(epochs):
                accuracies = []
                progress_bar = tqdm_notebook(batch_generator(X_tr, y_tr, batch_size, to_shuffle=True),
                                             total=n_batchs + extra_batch)
                for x, y_ in progress_bar:
                    weights = np.zeros_like(x)
                    weights[np.where(x != 0)] = 1.0
                    fetches = {'train': self.train, 'accuracy': self.accuracy, 'step': self.global_step,
                               'loss': self.loss}
                    feed_dict = {self.targets: y_, self.inputs: x, self.weights: weights,
                                 self.dropout: dropout}
                    if with_lr:
                        feed_dict.update({self.lr: with_lr})

                    fetched = sess.run(fetches=fetches,
                                       feed_dict=feed_dict)
                    accuracies.append(fetched['accuracy'][1] * 100)
                    acc = fetched['accuracy'][1] * 100
                    loss = fetched['loss']
                    progress_bar.set_postfix_str(
                        f'ep: {epoch + 1}/{epochs}, loss: {"%.4f" % loss}, acc: {"%.3f" % acc}, ep_acc: {"%.3f" % ep_acc}')
                    if fetched['step'] % validation_step + 1 == validation_step:
                        validation_losses[fetched['step']] = validate(X_vl, y_vl, sess)
                    # if fetched['step'] % self.save_per_step == 0:
                    #     self.saver.save(sess, self.chkp_dir.rstrip('/') + '/' + 'my_model',
                    #                     global_step=fetched['step'])

                ep_acc = sum(accuracies) / (n_batchs + extra_batch)
                if epoch % save_per_epoch + 1 == save_per_epoch:
                    self.saver.save(sess, self.chkp_dir.rstrip('/') + '/' + 'my_model',
                                    global_step=fetched['step'])
                    !cp my_model* '/content/drive/My Drive/DEEP LEARNING/JUPYTER/'
                    !cp checkpoint '/content/drive/My Drive/DEEP LEARNING/JUPYTER/'
        # except:
        #     print(f'saving checkpoint to {self.chkp_dir + "/my_model"}')
        #     self.saver.save(sess, self.chkp_dir.rstrip('/') + '/' + 'my_model')
        return self

    def predict(self, X, batch_size, from_chkp):
        predictions = []
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer()])
            self.saver.restore(sess, self.chkp_dir.rstrip('/') + '/' + f'{from_chkp}')
            for x in batch_generator(X, bs=batch_size):
                batch_predictions = sess.run(fetches={'predictions': self.predictions},
                                             feed_dict={self.inputs: x, self.dropout: 0.0})
                predictions.append(batch_predictions['predictions'])
        return np.concatenate(predictions, axis=0)

    def validate(self, X, y, sess):
        fetches = {'loss': self.loss, 'accuracy': self.accuracy}
        for x, y_ in batch_generator(X, y, 100):
            feed_dict = {self.inputs: x, self.targets: y_}
            sess.run(fetches, feed_dict=)

def contiguous_batch_generator():
    pass


def batch_generator(X, y=None, bs=30, to_shuffle=False):
    extra_ep = 0 if len(X) % bs == 0 else 1
    ep_size = (len(X) // bs) + extra_ep
    A = np.copy(X)
    b = None
    if y is not None: b = np.copy(y)
    if to_shuffle:
        idxs = shuffle(np.arange(len(X)), random_state=2019)
        A = A[idxs]
        b = b[idxs]
    for i in range(ep_size):
        if y is not None:
            yield A[i * bs: (i + 1) * bs], b[i * bs: (i + 1) * bs]
        else:
            yield A[i * bs: (i + 1) * bs]


def save_predictions(filename, predictions, df, vocabulary):
    voc = dict([(int(idx), cat) for cat, idx in vocabulary.items()])
    with open(filename, 'w') as f:
        for i in range(len(df)):
            tokens = df.iloc[i].tokens
            IDs = df.iloc[i].IDs
            predicts = predictions[i]
            for token, predict, id in zip(tokens[1:], predicts, IDs):
                if token == '_pad_': break
                if predict not in voc:
                    cat = '_unk_#_unk_'
                else:
                    cat = voc[predict]
                pos = cat.split('#')[0]
                gram_cats = cat.split('#')[1]
                print(id, token, '_', pos, gram_cats, sep='\t', file=f)
            print(file=f)
