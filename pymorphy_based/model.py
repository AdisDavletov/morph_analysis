import sys

import numpy as np
import tensorflow as tf
from reader import BatchGenerator, Loader
from tensorflow import divide
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple
from tqdm import tqdm_notebook, tqdm

sys.path.append('../')
from vectorizers.endings_vectorizer import EndingsVectorizer
from vectorizers.grammems_vectorizer import GrammemsVectorizer

from configs import BuildConfig, TrainConfig


class Analyser:
    def __init__(self, build_config, train_config, is_training, chkp_dir='chkps'):
        self.chkp_dir = chkp_dir
        self.endings_vectorizer = EndingsVectorizer(n_endings=build_config.n_endings,
                                                    lower=build_config.lower)
        self.grammeme_vectorizer_input = GrammemsVectorizer()
        self.grammeme_vectorizer_output = GrammemsVectorizer()
        self.build_config = build_config
        self.train_config = train_config
        self.endings_input = None
        self.endings_embedding = None
        self.grammems_input = None
        self.first_layer_outputs = None
        self.saver = None
        self.is_training = is_training

    def prepare(self, filenames):
        loader = Loader(n_endings=self.build_config.n_endings, lower=self.build_config.lower)
        loader.parse_corpora(filenames)
        self.grammeme_vectorizer_input = loader.grammeme_vectorizer_input
        self.grammeme_vectorizer_output = loader.grammeme_vectorizer_output
        self.endings_vectorizer = loader.endings_vectorizer

    def build(self):
        config = self.build_config
        embeddings = []

        rnn_out_drop = tf.get_variable(name='rnn_out_drop', trainable=False, initializer=config.rnn_out_drop)
        endings_inp_drop = tf.get_variable(name='endings_inp_drop', trainable=False,
                                           initializer=config.endings_inp_drop)
        gram_inp_drop = tf.get_variable(name='gram_inp_drop', trainable=False,
                                        initializer=config.gram_inp_drop)
        rnn_state_drop = tf.get_variable(name='rnn_state_drop', trainable=False,
                                         initializer=config.rnn_state_drop)
        dense_drop = tf.get_variable(name='dense_drop', trainable=False, initializer=config.dense_drop)
        self.training = tf.get_variable(name='is_training', trainable=False, dtype=tf.bool, initializer=True)
        self.lr = tf.get_variable(name='lr', shape=[], trainable=False)
        tf.summary.scalar('lr__', self.lr)

        self.weights = tf.placeholder(dtype=tf.float32, shape=[None, None], name='weights')
        weights = tf.reshape(self.weights, shape=[-1])
        total = tf.reduce_sum(weights)
        self.total = total
        self.global_step = tf.train.get_or_create_global_step()

        if config.use_endings:
            with tf.variable_scope('word_endings'):
                voc_size = self.endings_vectorizer.get_size()
                self.endings_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='endings_input')
                self.endings_embedding = tf.get_variable('endings_embs', shape=[voc_size, config.endings_emb_size],
                                                         initializer=tf.initializers.random_normal)
                endings_input = tf.nn.embedding_lookup(self.endings_embedding, self.endings_input)
                endings_input = tf.nn.dropout(endings_input, rate=endings_inp_drop)
                embeddings.append(endings_input)

        if config.use_gram:
            with tf.variable_scope('grammems'):
                gram_vec_size = self.grammeme_vectorizer_input.grammemes_count()
                self.grammems_input = tf.placeholder(dtype=tf.float32, shape=[None, None, gram_vec_size],
                                                     name='grammems_input')
                grammems_input = tf.nn.dropout(self.grammems_input, rate=gram_inp_drop)
                embeddings.append(grammems_input)

        if len(embeddings) > 1:
            embeddings = tf.concat(embeddings, axis=-1, name='concatenated_inputs')
        else:
            embeddings = embeddings[0]
        batch_size = tf.shape(embeddings, name='batch_size')[0]

        with tf.variable_scope('lstm_input'):
            lstm_input = tf.get_variable(name='lstm_input',
                                         shape=[embeddings.get_shape().as_list()[-1], config.rnn_hidden_size])
            lstm_input_bias = tf.get_variable(name='lstm_input_bias', shape=[config.rnn_hidden_size])
            lstm_input = tf.tensordot(embeddings, lstm_input, axes=((-1), (0))) + lstm_input_bias
            lstm_input = tf.nn.relu(lstm_input)

        with tf.variable_scope('lstm'):
            initial_state_forward = tf.get_variable(name='f_initial_state_1', shape=[config.rnn_hidden_size * 2])
            initial_state_backward = tf.get_variable('b_initial_state_1', shape=[config.rnn_hidden_size * 2])

            f_init_state_c = tf.expand_dims(initial_state_forward[:config.rnn_hidden_size], axis=0)
            f_init_state_m = tf.expand_dims(initial_state_forward[config.rnn_hidden_size:], axis=0)
            b_init_state_c = tf.expand_dims(initial_state_backward[:config.rnn_hidden_size], axis=0)
            b_init_state_m = tf.expand_dims(initial_state_backward[config.rnn_hidden_size:], axis=0)

            f_init_state_c = tf.tile(f_init_state_c, multiples=[batch_size, 1])
            f_init_state_m = tf.tile(f_init_state_m, multiples=[batch_size, 1])
            b_init_state_c = tf.tile(b_init_state_c, multiples=[batch_size, 1])
            b_init_state_m = tf.tile(b_init_state_m, multiples=[batch_size, 1])

            f_init_state = LSTMStateTuple(f_init_state_c, f_init_state_m)
            b_init_state = LSTMStateTuple(b_init_state_c, b_init_state_m)

            f_lstm_cell = tf.nn.rnn_cell.LSTMCell(config.rnn_hidden_size, name='f_lstm_cell_1')
            b_lstm_cell = tf.nn.rnn_cell.LSTMCell(config.rnn_hidden_size, name='b_lstm_cell_1')
            f_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(f_lstm_cell, state_keep_prob=1. - rnn_state_drop,
                                                        output_keep_prob=1. - rnn_out_drop,
                                                        seed=config.seed)
            b_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(b_lstm_cell, state_keep_prob=1. - rnn_state_drop,
                                                        output_keep_prob=1. - rnn_out_drop,
                                                        seed=config.seed)

            (f_outputs, b_outputs), _ = bidirectional_dynamic_rnn(f_lstm_cell, b_lstm_cell, lstm_input,
                                                                  initial_state_fw=f_init_state,
                                                                  initial_state_bw=b_init_state)

            def merge_mode(forward, backward):
                if config.merge_mode == 'ave':
                    outputs = tf.reduce_mean(tf.stack([forward, backward], axis=0), axis=0)
                elif config.merge_mode == 'concat':
                    outputs = tf.concat([forward, backward], axis=-1, name='1st_birnn_layer_outputs')
                elif config.merge_mode == 'sum':
                    outputs = tf.reduce_sum(tf.stack([forward, backward], axis=0), axis=0)
                else:
                    raise ValueError()
                return outputs

            outputs = merge_mode(f_outputs, b_outputs)

            # self.first_layer_outputs = outputs  # [bs, seq_len, rnn_hidden_size (2 * rnn_hidden_size)]

            def make_cell(size, name):
                f_cell = tf.nn.rnn_cell.LSTMCell(size, name='f_' + name)
                b_cell = tf.nn.rnn_cell.LSTMCell(size, name='b_' + name)
                f_cell = tf.nn.rnn_cell.DropoutWrapper(f_cell, output_keep_prob=rnn_out_drop,
                                                       state_keep_prob=1. - rnn_state_drop)
                b_cell = tf.nn.rnn_cell.DropoutWrapper(b_cell, output_keep_prob=rnn_out_drop,
                                                       state_keep_prob=1. - rnn_state_drop)
                return (f_cell, b_cell)

            extra_rnn_layers = config.n_rnn_layers - 1
            if extra_rnn_layers > 0:
                initial_state_forward = tf.get_variable('f_initial_state_2', shape=[config.rnn_hidden_size * 2])
                initial_state_backward = tf.get_variable('b_initial_state_2', shape=[config.rnn_hidden_size * 2])

                f_init_state_c = tf.expand_dims(initial_state_forward[:config.rnn_hidden_size], axis=0)
                f_init_state_m = tf.expand_dims(initial_state_forward[config.rnn_hidden_size:], axis=0)
                b_init_state_c = tf.expand_dims(initial_state_backward[:config.rnn_hidden_size], axis=0)
                b_init_state_m = tf.expand_dims(initial_state_backward[config.rnn_hidden_size:], axis=0)

                f_init_state_c = tf.tile(f_init_state_c, multiples=[batch_size, 1])
                f_init_state_m = tf.tile(f_init_state_m, multiples=[batch_size, 1])
                b_init_state_c = tf.tile(b_init_state_c, multiples=[batch_size, 1])
                b_init_state_m = tf.tile(b_init_state_m, multiples=[batch_size, 1])

                f_init_state = LSTMStateTuple(f_init_state_c, f_init_state_m)
                b_init_state = LSTMStateTuple(b_init_state_c, b_init_state_m)

                f_init_state = tuple([f_init_state] * extra_rnn_layers)
                b_init_state = tuple([b_init_state] * extra_rnn_layers)

                cells = [make_cell(config.rnn_hidden_size, name=f'lstm_cell_{i + 2}') for i in
                         range(extra_rnn_layers)]
                f_cells = [x for (x, y) in cells]
                b_cells = [y for (x, y) in cells]
                f_cell = tf.nn.rnn_cell.MultiRNNCell(f_cells, state_is_tuple=True)
                b_cell = tf.nn.rnn_cell.MultiRNNCell(b_cells, state_is_tuple=True)

                (f_rnn_outputs, b_rnn_outputs), _ = bidirectional_dynamic_rnn(f_cell, b_cell, outputs,
                                                                              initial_state_fw=f_init_state,
                                                                              initial_state_bw=b_init_state)
                outputs = merge_mode(f_rnn_outputs, b_rnn_outputs)  # [bs, seq_len, rnn_size (2 * rnn_size)]

        with tf.variable_scope('after_lstm'):
            rnn_output_size = config.rnn_hidden_size if config.merge_mode != 'concat' else config.rnn_hidden_size

            outputs = self.dense_layer(rnn_output_size, config.dense_size, 'dense_post_rnn', outputs)
            outputs = tf.nn.dropout(outputs, rate=dense_drop)
            outputs = tf.layers.batch_normalization(outputs, training=self.training, trainable=True)
            outputs = tf.nn.relu(outputs)

        if config.use_pos_lm:
            with tf.variable_scope('next_pos'):
                self.next_pos_target = tf.placeholder(dtype=tf.int32, shape=[None, None])
                next_pos = self.dense_layer(rnn_output_size, config.dense_size, 'dense_next_pos', f_outputs, 'relu')
                next_pos = self.dense_layer(config.dense_size, self.grammeme_vectorizer_output.pos_count() + 1,
                                            'next_pos',
                                            next_pos, 'softmax')
                next_pos_loss = sequence_loss(logits=next_pos,
                                              targets=self.next_pos_target,
                                              weights=self.weights,
                                              average_across_timesteps=False,
                                              average_across_batch=False,
                                              name='next_pos_loss')
                next_pos_loss = tf.reshape(next_pos_loss, shape=[-1])
                next_pos_loss *= weights
                self.next_pos_loss = tf.reduce_sum(next_pos_loss)
                self.next_pos_loss_avg = self.next_pos_loss / total
                tf.summary.scalar('next_pos_loss__', self.next_pos_loss_avg)

            with tf.variable_scope('pred_pos'):
                self.pred_pos_target = tf.placeholder(dtype=tf.int32, shape=[None, None])
                pred_pos = self.dense_layer(rnn_output_size, config.dense_size, 'dense_pred_pos', f_outputs, 'relu')
                pred_pos = self.dense_layer(config.dense_size, self.grammeme_vectorizer_output.pos_count() + 1,
                                            'pred_pos',
                                            pred_pos, 'softmax')

                pred_pos_loss = sequence_loss(logits=pred_pos,
                                              targets=self.pred_pos_target,
                                              weights=self.weights,
                                              average_across_timesteps=False,
                                              average_across_batch=False,
                                              name='pred_pos_loss')
                pred_pos_loss = tf.reshape(pred_pos_loss, shape=[-1])
                pred_pos_loss *= weights
                self.pred_pos_loss = tf.reduce_sum(pred_pos_loss)
                self.pred_pos_loss_avg = self.pred_pos_loss / total
                tf.summary.scalar('pred_pos_loss__', self.pred_pos_loss_avg)

        with tf.variable_scope('main_pred'):
            self.target = tf.placeholder(dtype=tf.int32, shape=[None, None])
            outputs = self.dense_layer(config.dense_size, self.grammeme_vectorizer_output.get_size() + 1, 'main_pred',
                                       outputs, 'softmax')
            main_loss = sequence_loss(logits=outputs,
                                      targets=self.target,
                                      weights=self.weights,
                                      average_across_timesteps=False,
                                      average_across_batch=False,
                                      name='main_loss')
            main_loss = tf.reshape(main_loss, shape=[-1])
            main_loss *= weights
            self.main_loss = tf.reduce_sum(main_loss)
            self.main_loss_avg = self.main_loss / total
            tf.summary.scalar('main_loss__', self.main_loss_avg)
            targets = tf.reshape(self.target, shape=[-1])
            predictions = tf.cast(tf.reshape(tf.argmax(outputs, axis=-1), shape=[-1]), dtype=tf.int32)
            correct = tf.cast(tf.equal(predictions, targets), dtype=tf.float32)
            self.correct = tf.reduce_sum(correct * weights)
            self.accuracy = divide(self.correct, total + 1e-12, name='accuracy')
            tf.summary.scalar('accuracy__', self.accuracy)

        self.summaries = tf.summary.merge_all()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        if self.is_training:
            with tf.control_dependencies(update_ops):
                def get_optimizer(build_config, lr):
                    if build_config.optimizer.lower() == 'adam':
                        optimizer = tf.train.AdamOptimizer(lr)
                    elif build_config.optimizer.lower() == 'sgd':
                        optimizer = tf.train.GradientDescentOptimizer(lr)
                    elif build_config.optimizer.lower() == 'rmsprop':
                        optimizer = tf.train.RMSPropOptimizer(lr)
                    elif build_config.optimizer.lower() == 'adagrad':
                        optimizer = tf.train.AdagradOptimizer(lr)
                    else:
                        raise ValueError()
                    return optimizer

                optimizer = get_optimizer(self.build_config, self.lr)

                trainable_variables = tf.trainable_variables()
                loss = tf.constant(0.0, dtype=tf.float32)
                if config.use_pos_lm:
                    loss += pred_pos_loss + next_pos_loss
                loss += self.main_loss_avg
                if config.use_wd:
                    self.wd = tf.get_variable(name='weight_decay', initializer=config.wd, trainable=False)
                    l2_loss = tf.constant(0.0, dtype=tf.float32)
                    for var in tf.trainable_variables():
                        l2_loss += tf.nn.l2_loss(var)
                    loss = loss + l2_loss * self.wd

                grads = tf.gradients(loss, trainable_variables)
                self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables), global_step=self.global_step,
                                                          name='train_op')

        self.variables_to_save = {'lr': self.lr, 'global_step': self.global_step}
        self.variables_to_save.update(dict([(x.op.name, x) for x in tf.trainable_variables()]))
        self.saver = tf.train.Saver(self.variables_to_save)

    def dense_layer(self, in_size, out_size, name, inputs, activation=None):
        weights = tf.get_variable('w_' + name, shape=[in_size, out_size])
        bias = tf.get_variable('b_' + name, shape=[out_size])
        result = tf.tensordot(inputs, weights, axes=((-1), (0))) + bias
        if activation == 'relu':
            result = tf.nn.relu(result)
        elif activation == 'softmax':
            result = tf.nn.softmax(result)
        return result

    def train(self, filenames, with_lr=None, bs=30, summary_step=10, validation_step=200):
        np.random.seed(self.train_config.random_seed)
        sample_counter = self.count_samples(filenames)
        train_idx, val_idx = self.get_split(sample_counter, self.train_config.val_part)

        total = len(train_idx) // self.train_config.external_batch_size + min(
            len(train_idx) % self.train_config.external_batch_size, 1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tr_wr = tf.summary.FileWriter(self.chkp_dir + '/train', sess.graph)
            val_wr = tf.summary.FileWriter(self.chkp_dir + '/dev', sess.graph)
            for epoch in range(self.train_config.n_epochs):
                batch_generator = tqdm(BatchGenerator(file_names=filenames,
                                                               train_config=self.train_config,
                                                               build_config=self.build_config,
                                                               grammeme_vectorizer_input=self.grammeme_vectorizer_input,
                                                               grammeme_vectorizer_output=self.grammeme_vectorizer_output,
                                                               endings_vectorizer=self.endings_vectorizer,
                                                               indices=train_idx
                                                               ),
                                                total=total)

                for data, target in batch_generator:
                    step = self.fit(sess, data, target, val_idx, filenames, bs, summary_step, validation_step, tr_wr,
                                    val_wr,
                                    with_lr)

                self.saver.save(sess, self.chkp_dir + '/my_model', global_step=step)

    def fit(self, sess, data, target, val_idx, filenames, bs, summary_step, validation_step, tr_wr, val_wr,
            with_lr=None):
        total = len(target['main']) // bs + min(len(target['main']) % bs, 1)
        progress = tqdm(range(total), total=total)
        fetches = {}
        config = self.build_config
        fetches.update({'main_loss': self.main_loss_avg, 'train_op': self.train_op,
                        'accuracy': self.accuracy})
        if config.use_pos_lm:
            fetches['pred'] = self.pred_pos_loss_avg
            fetches['next'] = self.next_pos_loss_avg
        fetches['step'] = self.global_step

        v_loss = '0.'
        v_acc = '0.'

        for i, _ in enumerate(progress):
            feed_dict = {}
            if with_lr is not None:
                feed_dict[self.lr] = with_lr
            feed_dict[self.target] = target['main'][i * bs:(i + 1) * bs]
            feed_dict[self.weights] = data['weights'][i * bs:(i + 1) * bs]
            if config.use_gram:
                feed_dict[self.grammems_input] = data['grammems'][i * bs:(i + 1) * bs]
            if config.use_endings:
                feed_dict[self.endings_input] = data['endings'][i * bs:(i + 1) * bs]
            if config.use_pos_lm:
                feed_dict[self.next_pos_target] = target['next'][i * bs:(i + 1) * bs]
                feed_dict[self.pred_pos_target] = target['pred'][i * bs:(i + 1) * bs]

            fetched = sess.run(fetches=fetches, feed_dict=feed_dict)

            acc = "%.0f" % (fetched['accuracy'] * 100)
            main_loss = "%.3f" % fetched['main_loss']
            next = "%.3f" % fetched['next']
            pred = "%.3f" % fetched['pred']
            step = fetched['step']

            if 'summaries' in fetches:
                tr_wr.add_summary(fetched['summaries'], global_step=step)
                fetches.pop('summaries')

            if step % summary_step + 1 == summary_step:
                fetches['summaries'] = self.summaries

            if step % validation_step + 1 == validation_step:
                v_total = 0
                v_loss = 0.0
                v_acc = 0.
                v_step = 1
                for v_data, v_target in BatchGenerator(file_names=filenames,
                                                       train_config=self.train_config,
                                                       grammeme_vectorizer_input=self.grammeme_vectorizer_input,
                                                       grammeme_vectorizer_output=self.grammeme_vectorizer_output,
                                                       endings_vectorizer=self.endings_vectorizer,
                                                       indices=val_idx,
                                                       build_config=self.build_config):
                    for j in range(len(v_target['main']) // bs + min(1, len(v_target['main']) % bs)):
                        new_feed_dict = {self.training: False, self.target: v_target['main'],
                                         self.weights: v_data['weights']}
                        if config.use_endings:
                            new_feed_dict[self.endings_input] = v_data['endings'][j * bs:(j + 1) * bs]
                        if config.use_gram:
                            new_feed_dict[self.grammems_input] = v_data['grammems'][j * bs:(j + 1) * bs]

                        results = sess.run(fetches={'loss': self.main_loss, 'step': self.global_step,
                                                    'correct': self.correct, 'total': self.total},
                                           feed_dict=new_feed_dict)

                        v_loss += results['loss']
                        v_total += results['total']
                        v_step = results['step']
                        v_acc += results['correct']
                v_loss = v_loss / v_total
                v_acc = v_acc / v_total
                summary_1 = tf.Summary(value=[
                    tf.Summary.Value(tag="main_loss__", simple_value=v_loss),
                ])
                summary_2 = tf.Summary(value=[
                    tf.Summary.Value(tag="accuracy__", simple_value=v_acc),
                ])
                val_wr.add_summary(summary_1, global_step=v_step)
                val_wr.add_summary(summary_2, global_step=v_step)
                v_loss = "%.3f" % v_loss
                v_acc = "%.0f" % (v_acc * 100)

            progress.set_postfix_str(
                f'acc: {"/".join([acc, v_acc])}, pred: {pred}, loss: {"/".join([main_loss, v_loss])}, next: {next}')
        return step

    @staticmethod
    def get_split(sample_counter: int, val_part: float):
        perm = np.random.permutation(sample_counter)
        border = int(sample_counter * (1 - val_part))
        train_idx = perm[:border]
        val_idx = perm[border:]
        return train_idx, val_idx

    @staticmethod
    def count_samples(file_names):
        sample_counter = 0
        flag = True
        for filename in file_names:
            with open(filename, "r", encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) != 0 and flag:
                        sample_counter += 1
                        flag = False
                        continue
                    if len(line) == 0:
                        flag = True

        return sample_counter


def main(filenames=['../datasets/gikrya_new_train.out']):
    train_config = TrainConfig()
    build_config = BuildConfig()

    print(build_config.__dict__)
    analyser = Analyser(build_config, train_config, is_training=True)
    analyser.prepare(filenames)
    analyser.build()
    analyser.train(filenames, bs=50, validation_step=250)

# def if __name__ == '__main__':
#     main()
