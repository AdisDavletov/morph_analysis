from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import collections

import numpy as np
import tensorflow as tf

import reader
import util

from tensorflow.python.client import device_lib

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", 'datasets/',
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", 'chkps/_tmp_/',
                    "Model output directory.")
flags.DEFINE_string("train_file", 'gikrya_new_train.out',
                    "train_data_file name")
flags.DEFINE_string("valid_file", 'gikrya_new_test.out',
                    "valid_data_file name")
flags.DEFINE_string("test_file", 'gikrya_new_test.out',
                    "test_data_file name")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32-bit floats")
flags.DEFINE_integer("num_gpus", 0,
                     "If larger than 1, Grappler AutoParallel optimizer "
                     "will create multiple training replicas with each GPU "
                     "running one replica.")
flags.DEFINE_bool("debug", False,
                  "Turn on/off debug mode.")
flags.DEFINE_bool("suffix_model", False,
                  "If true model would use suffix_model mode")

flags.DEFINE_float("lr", 1.0,
                  "Learning rate")
flags.DEFINE_list('classifiers_to_train', ['pos','case','gender','number','animacy', 'tense','person', 'verbform','mood', 'variant','degree', 'numform'],
                  "Classifiers to train, this defines architecture of classifiers layer")
flags.DEFINE_bool("prediction", False,
                  "Prediction mode to evaluate")
flags.DEFINE_float("keep_prob", 0.5,
                  "keep probability")

FLAGS = flags.FLAGS

def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class Input(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data['word']) // batch_size) - 1) // num_steps
    self.input_data = reader.ptb_producer(
        data, batch_size, num_steps, name=name)


class Model(object):
  """The model."""

  def __init__(self, is_training, config, input_, name=""):
    self.debug = True
    self._name = name
    self._is_training = is_training
    self._input = input_
    self._rnn_out_size = config.hidden_size[-1]
    self._losses = {}
    self._logits = {}
    self.batch_size = input_.batch_size
    self.num_steps = input_.num_steps
    self._init_global_step_and_epoch()
    self._build_model(config)

  def _build_model(self, config):
    with tf.variable_scope("emb_part"):
      inputs = self._add_embed_layer(
        config.vocab_size if not FLAGS.suffix_model else config.suffix_size,
        config.embedding_size,
        config.keep_prob['embedding_layer'])
    with tf.variable_scope("rnn_part"):
      output = self._add_rnn_layer(
        inputs,
        config.num_layers,
        config.keep_prob['rnn_layer'],
        config.hidden_size,
        config.cell_type)
    with tf.variable_scope("lm_part"):
      output = tf.reshape(output, [-1, self._rnn_out_size])

      logits = self._add_clf_layer(output, self._rnn_out_size,
        config.vocab_size if not FLAGS.suffix_model else config.suffix_size,
        layer_name='lm_layer')

      self._logits.update({'word_target': logits})
    with tf.variable_scope("clfs_part", reuse=tf.AUTO_REUSE):
      self._add_classifiers(config.classifiers)
    self._add_predictions()

    self._collect_trainable_variables()
    self._add_losses()
    self._add_l2_losses(config.weight_decay)

    self._losses.update({'clfs_loss':
      self._weighted_losses_sum([(f'{clf}_loss', 1) for clf in config.classifiers])})

    self._losses.update({'word_target_loss_l2_reg': self._losses['word_target_loss'] + \
      self._l2_losses['lang_model']})

    self._losses.update({'clfs_loss_l2_reg': self._losses['clfs_loss'] + \
      self._l2_losses['classifiers']})

    self._losses.update({'clfs_loss_withfreezedrnn_l2_reg': self._losses['clfs_loss'] + \
      self._l2_losses['classifiers_with_freezed_rnn']})

    self._losses.update({'clfs_loss_withfreezedrnnembs_l2_reg': self._losses['clfs_loss'] + \
      self._l2_losses['classifiers_with_freezed_rnn_embs']})

    self._losses.update({'total_loss':
      self._weighted_losses_sum([('word_target_loss', config.rnn_loss_weight),
        ('clfs_loss', config.clfs_loss_weight)])})
    self._losses.update({'total_loss_l2_reg':
      self._weighted_losses_sum([('word_target_loss_l2_reg', config.rnn_loss_weight),
        ('clfs_loss_l2_reg', config.clfs_loss_weight)])})

    self._losses_for_train = {}
    losses = ['word_target_loss_l2_reg', 'clfs_loss_l2_reg', 'clfs_loss_withfreezedrnn_l2_reg',
      'clfs_loss_withfreezedrnnembs_l2_reg', 'total_loss_l2_reg']
    assoc_names = ['lang_model', 'classifiers', 'classifiers_with_freezed_rnn',
      'classifiers_with_freezed_rnn_embs', 'total']
    for new_name, old_name in zip(assoc_names, losses):
      self._losses_for_train[new_name] = self._losses[old_name]

    if self._is_training:
      self._add_optimizers(config.max_grad_norm, config.optimizer)
  
  def _init_global_step_and_epoch(self):
    self._global_step = tf.train.get_or_create_global_step()
    self._cur_epoch = tf.get_variable('epoch', dtype=tf.int32, initializer=0, trainable=False)
    if self._is_training:
      self._new_val = tf.placeholder(name='new_val', shape=[], dtype=tf.int32)
      self._update_epoch = tf.assign(self._cur_epoch, self._new_val)

  def _add_embed_layer(self, vocab_size, embedding_size, keep_prob):
    with tf.device("/cpu:0"):
      embedding = tf.get_variable(
          "embedding", [vocab_size, embedding_size], dtype=data_type())
      inputs = tf.nn.embedding_lookup(embedding, self._input.input_data['word'])

    if self._is_training and keep_prob < 1:
      inputs = tf.nn.dropout(inputs, keep_prob)
    return inputs

  def _add_rnn_layer(self, inputs, num_layers, keep_prob, hidden_size, cell_type):
    """add the inference graph using canonical LSTM cells."""
    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    def make_cell(layer):
      if cell_type == 'lstm_block':
        cell = tf.contrib.rnn.LSTMBlockCell(hidden_size[layer], forget_bias=1.0, name=f'rnn_cell_{layer}')
      elif cell_type == 'lstm_block_fused':
        cell = tf.contrib.rnn.LSTMBlockFusedCell(hidden_size[layer], name=f'rnn_cell_{layer}')
      else:
        raise ValueError('Not supported cell_type!')

      if self._is_training and keep_prob[layer] < 1:
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=keep_prob[layer])
      return cell

    cell = tf.contrib.rnn.MultiRNNCell(
        [make_cell(i) for i in range(num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(self.batch_size, data_type())

    inputs = tf.unstack(inputs, num=self.num_steps, axis=1)
    outputs, self._final_state = tf.nn.static_rnn(cell, inputs,
                                      initial_state=self._initial_state)
    output = tf.concat(outputs, axis=1)
    self._rnn_output = tf.reshape(output, [self.batch_size, self.num_steps, self._rnn_out_size])

    return output

  def _add_classifiers(self, classifiers):
    self._padding = tf.get_variable(name=f'{self._name}_clfs_padd', shape=[self._rnn_output.get_shape()[0], 1, self._rnn_out_size],
      dtype=data_type(), trainable=False)
    self._last_step = tf.strided_slice(self._rnn_output, [0, self.num_steps-1, 0],
      [self.batch_size, self.num_steps, self._rnn_out_size], name=f'{self._name}_last_step')
    prev_output = tf.concat([self._padding, tf.strided_slice(self._rnn_output,
        [0, 0, 0], [self.batch_size, self.num_steps-1, self._rnn_out_size])], axis=1)
    prev_output.set_shape([self.batch_size, self.num_steps, self._rnn_out_size])
    inputs = tf.concat([prev_output, self._rnn_output], axis=2)
    if self.debug: print('shape of the inputs to classifier layers:', inputs.get_shape(), flush=True)
    inputs = tf.reshape(inputs, shape=[-1, self._rnn_out_size * 2])
    for name, out_size in classifiers.items():
      logits = self._add_clf_layer(
        inputs=inputs,
        in_size=self._rnn_out_size * 2,
        out_size=out_size,
        layer_name=name)
      self._logits.update({name: logits})

  def _add_clf_layer(self, inputs, in_size, out_size, layer_name):
    weights = tf.get_variable(
        layer_name + '_w', [in_size, out_size], dtype=data_type())
    bias = tf.get_variable(layer_name + "_b", [out_size], dtype=data_type())

    logits = tf.nn.xw_plus_b(inputs, weights, bias)
    # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.batch_size, self.num_steps, out_size])

    return logits

  def _collect_trainable_variables(self):
    scope = tf.get_variable_scope().name
    self._tvars = {}
    self._tvars['emb_part'] = tf.trainable_variables(scope=f"{scope+'/' if scope else ''}emb_part")
    self._tvars['rnn_part'] = tf.trainable_variables(scope=f"{scope+'/' if scope else ''}rnn_part")
    self._tvars['lm_part'] = tf.trainable_variables(scope=f"{scope+'/' if scope else ''}lm_part")
    self._tvars['clfs_part'] = tf.trainable_variables(scope=f"{scope+'/' if scope else ''}clfs_part")

    tvars = self._tvars
    self._for_train = {}
    self._for_train['lang_model'] = tvars['emb_part'] + tvars['rnn_part'] + tvars['lm_part']
    self._for_train['classifiers'] = tvars['emb_part'] + tvars['rnn_part'] + tvars['clfs_part']
    self._for_train['classifiers_with_freezed_rnn'] = tvars['emb_part'] + tvars['clfs_part']
    self._for_train['classifiers_with_freezed_rnn_embs'] = tvars['clfs_part']
    self._for_train['total'] = tvars['emb_part'] + tvars['rnn_part'] + \
      tvars['lm_part'] + tvars['clfs_part']

  def _add_losses(self):
    for name, logits in self._logits.items():
      targets = self._input.input_data[name]
      loss = self._add_loss(logits, targets, name)
      self._losses.update({f'{name}_loss': loss})

  def _add_loss(self, answers, targets, name):
    # Use the contrib sequence loss and average over the batches
    loss = tf.contrib.seq2seq.sequence_loss(
        logits=answers,
        targets=targets,
        weights=tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
        average_across_timesteps=False,
        average_across_batch=True,
        name=name)

    return tf.reduce_sum(loss)

  def _add_l2_losses(self, weight_decay):
    self._l2_losses = {}
    for name, tvars in self._for_train.items():
      l2_loss = tf.constant(0.0, name=name)
      for i, var in enumerate(tvars):
        l2_loss += tf.nn.l2_loss(var, name=f'{name}_{i}')
      self._l2_losses[name] = l2_loss * weight_decay[name]


  def _weighted_losses_sum(self, names_with_weights):
    total_sum = tf.constant(0.0)
    for name, weight in names_with_weights:
      total_sum += self._losses[name] * weight
    return total_sum

  def _add_optimizers(self, max_grad_norm, optimizer):
    self._lr = tf.Variable(1.0, trainable=False)
    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    self._train_ops = {}
    optimizer = self._get_optimizer(optimizer)

    for key, loss in self._losses_for_train.items():
      grads = tf.gradients(loss, self._for_train[key])
      grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)
      self._train_ops[key] = optimizer.apply_gradients(
        zip(grads, self._for_train[key]),
        global_step=self._global_step,
        name=key)

  def _get_optimizer(self, optimizer='SGD'):
    optimizer = optimizer.upper()
    if optimizer == 'ADAGRAD':
      optimizer = tf.train.AdagradOptimizer(self._lr)
    elif optimizer == 'SGD':
      optimizer = tf.train.GradientDescentOptimizer(self._lr)
    else:
      raise ValueError('Not supposed optimizer!')
    return optimizer

  def _add_predictions(self):
    self._predictions = {}
    for clf in self._logits: # bs x seq_len x logits_size
      if clf == 'word_target': continue
      preds = tf.nn.softmax(self._logits[clf], axis=2, name=f"softmax_{clf}")
      self._predictions[clf] = tf.argmax(preds,
        axis=2, name=f"argmax_{clf}",
        output_type=tf.int32)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def _increment_epoch(self, session, new_val):
    session.run(self._update_epoch, feed_dict={self._new_val: new_val})

  def describe_model(self):
    print(self._name)
    for key, loss in self._losses_for_train.items():
      print(f"loss {key}:\n{loss}")
    for key, tvars in self._for_train.items():
      print(f"trainable variables for training {self._losses_for_train[key]} model:\n{tvars}")
    for key, op in self._train_ops.items():
      print(f"training op {key}:\n{op}")

  def export_ops(self, name):
    """Exports ops to collections."""
    self._name = name
    self._exported_ops = {}
    ops = {}
    ops[util.with_prefix(self._name, 'clfs_padd')] = self._padding
    ops[util.with_prefix(self._name, 'last_step')] = self._last_step
    for key, preds in self._predictions.items():
      ops[util.with_prefix(self._name, key)] = preds
    self._exported_ops.update({'predictions': [f'{self._name}/{k}' for k in self._predictions]})
    for key, loss in self._losses.items():
      ops[util.with_prefix(self._name, key)] = loss
    for key, loss in self._l2_losses.items():
      ops[util.with_prefix(self._name, key)] = loss
    self._exported_ops.update({'losses': [f'{self._name}/{k}' for k in self._losses.keys()],
      'l2_losses': [f'{self._name}/{k}' for k in self._l2_losses.keys()]})
    ops.update({f'{name}/cur_epoch': self._cur_epoch,
      f'{name}/global_step': self._global_step})
    self._exported_ops.update({'epoch_and_step': [f'{name}/cur_epoch', f'{name}/global_step']})
    if self._is_training:
      for key, train_op in self._train_ops.items():
        ops[key] = train_op
      self._exported_ops.update({'train_ops': self._train_ops.keys()})
      ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update,
        new_val=self._new_val, update_epoch=self._update_epoch)
      
    for name, op in ops.items():
      tf.add_to_collection(name, op)

    self._initial_state_name = util.with_prefix(self._name, "initial")
    self._final_state_name = util.with_prefix(self._name, "final")
    util.export_state_tuples(self._initial_state, self._initial_state_name)
    util.export_state_tuples(self._final_state, self._final_state_name)

  def import_ops(self):
    """Imports ops from collections."""
    self._padding = tf.get_collection_ref(util.with_prefix(self._name, 'clfs_padd'))[0]
    self._last_step = tf.get_collection_ref(util.with_prefix(self._name, 'last_step'))[0]
    if self._is_training:
      self._lr = tf.get_collection_ref("lr")[0]
      self._new_lr = tf.get_collection_ref("new_lr")[0]
      self._lr_update = tf.get_collection_ref("lr_update")[0]
      self._new_val = tf.get_collection_ref("new_val")[0]
      self._update_epoch = tf.get_collection_ref("update_epoch")

      for key in self._exported_ops['train_ops']:
        self._train_ops[key] = tf.get_collection_ref(key)[0]
    for key in self._exported_ops['epoch_and_step']:
      if key == f'{self._name}/cur_epoch':
        self._cur_epoch = tf.get_collection_ref(key)[0]
      else:
        self._global_step = tf.get_collection_ref(key)[0]
    for key in self._exported_ops['losses']:
      self._losses[key.split('/')[1]] = tf.get_collection_ref(key)[0]
    for key in self._exported_ops['l2_losses']:
      self._l2_losses[key.split('/')[1]] = tf.get_collection_ref(key)[0]
    for key in self._exported_ops['predictions']:
      self._predictions[key.split('/')[1]] = tf.get_collection_ref(key)[0]
    num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
    self._initial_state = util.import_state_tuples(
        self._initial_state, self._initial_state_name, num_replicas)
    self._final_state = util.import_state_tuples(
        self._final_state, self._final_state_name, num_replicas)

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def initial_state_name(self):
    return self._initial_state_name

  @property
  def final_state_name(self):
    return self._final_state_name

  @property
  def train_ops(self):
    return self._train_ops

  def get_cost(self, cost_name):
    return self._losses[cost_name]
  

class SmallConfig(object):
  """Small config."""
  def __init__(
      self,
      init_scale=0.1,
      learning_rate=1.0,
      max_grad_norm=5,
      num_layers=2,
      num_steps=35,
      hidden_size=640,
      embedding_size=640,
      max_epoch=10,
      max_max_epoch=39,
      keep_prob={'embedding_layer':0.5,'rnn_layer':[0.5, 0.5]},
      lr_decay=0.95,
      batch_size=20,
      vocab_size=46099,
      classifiers={
        'pos':14,'case':7,
        'gender':4,'number':3,
        'animacy':3, 'tense':3,
        'person':4, 'verbform':4,
        'mood':3, 'variant':3,
        'degree':3, 'numform':3},
      cell_type='lstm_block',
      weight_decay={
        'lang_model':0.0002,
        'classifiers':0.002,
        'classifiers_with_freezed_rnn':0.0002,
        'classifiers_with_freezed_rnn_embs':0.0002,
        'total':0.0002
      },
      rnn_loss_weight=1.,
      clfs_loss_weight=1.,
      optimizer='SGD',
      train_op='lang_model',
      loss_to_view='word_target_loss_l2_reg',
      min_tf=2,
      word_to_id='vocabularies/vocabulary'
    ):
    self.init_scale = init_scale
    self.learning_rate = learning_rate
    self.max_grad_norm = max_grad_norm
    self.num_layers = num_layers
    self.num_steps = num_steps
    self.hidden_size = [hidden_size] * num_layers
    self.embedding_size = embedding_size
    self.max_epoch = max_epoch
    self.max_max_epoch = max_max_epoch
    self.keep_prob = keep_prob
    self.lr_decay = lr_decay
    self.batch_size = batch_size
    self.vocab_size = vocab_size
    self.classifiers = classifiers
    self.cell_type = cell_type
    self.weight_decay = weight_decay
    self.rnn_loss_weight = rnn_loss_weight
    self.clfs_loss_weight = clfs_loss_weight
    self.optimizer = optimizer
    self.train_op = train_op
    self.loss_to_view = loss_to_view
    self.min_tf = min_tf
    self.word_to_id = word_to_id


  def print(self):
    for key in self.__dict__:
      print(f"{key}: {self.__dict__[key]}", flush=True)


def run_epoch(session, model, cost_name, eval_op=None, verbose=False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.get_cost(cost_name),
      "final_state": model.final_state,
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op

  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    if iters != 0:
      feed_dict[model._padding] = vals['padding']
    else:
      fetches['padding'] = model._last_step
      feed_dict[model._padding] = np.zeros(shape=[model.batch_size, 1, model._rnn_out_size], dtype=np.float32)

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]

    costs += cost
    iters += model.input.num_steps

    if verbose and step % (model.input.epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size * max(1, FLAGS.num_gpus) /
             (time.time() - start_time)), flush=True)

  return np.exp(costs / iters)

def predict(session, model, classifiers, word_to_id, file='predicted.out'):
  state = session.run(model.initial_state)
  id_to_word = {}
  start = True
  for key in word_to_id:
    if key != 'word':
      id_to_word[key] = dict([(y, x) for x, y in word_to_id[key].items()])
  predictions = collections.defaultdict(lambda:[])

  fetches = {"final_state": model.final_state, 'preds': model._predictions, 'padding': model._last_step}
  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    if not start:
      feed_dict[model._padding] = vals['padding']
    else:
      start = False
      feed_dict[model._padding] = np.zeros(shape=[model.batch_size, 1, model._rnn_out_size], dtype=np.float32)

    vals = session.run(fetches, feed_dict)

    preds = vals['preds']
    state = vals['final_state']

    # for x in preds['case'][0]:
    #   print(id_to_word[key][x])

    for clf in preds:
      predictions[clf].append(preds[clf])

  for clf in predictions:
    predictions[clf] = np.concatenate(predictions[clf], axis=1)
    predictions[clf] = predictions[clf].reshape([1,-1])

  for clf in predictions:
    print(clf)
    for x in predictions[clf][0,:50]:
      print(id_to_word[clf][x], sep=' ')

  return predictions

def run_op(session, op, feed_dict):
  res = session.run(op, feed_dict=feed_dict)
  return res

def get_config(**kwargs):
  """Get model config."""
  return SmallConfig(**kwargs)


def main(_):
  
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to data directory")
  gpus = [
      x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"
  ]
  if FLAGS.num_gpus > len(gpus):
    raise ValueError(
        "Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))

  train_data = FLAGS.train_file
  valid_data = FLAGS.valid_file
  test_data = FLAGS.test_file

  print(f'root_folder:{FLAGS.data_path}|train_file:{train_data}|valid_file:{valid_data}|test_file:{test_data}')
  config = get_config(train_op='total', loss_to_view='total_loss', max_max_epoch=60)
  eval_config = get_config(train_op='total', loss_to_view='total_loss', max_max_epoch=60)
  eval_config.batch_size = 1

  word_to_id = config.word_to_id #if FLAGS.prediction else None
  # with_tags_and_pos = False if FLAGS.prediction else True
  with_tags_and_pos = True

  raw_data = reader.ptb_raw_data(FLAGS.data_path,
      word_to_id=word_to_id,
      # word_to_id=None,
      train=train_data,
      dev=valid_data,
      test=test_data,
      additional_file=None,
      with_tags_and_pos=with_tags_and_pos,
      lower=True,
      unk_with_suffix=True,
      min_tf=config.min_tf,
      vocab_save_path=config.word_to_id)

  train_data, valid_data, test_data, word_to_id = raw_data
  id_to_word = collections.defaultdict(lambda: '<unk>', [(y, x) for x, y in word_to_id['word'].items()])
  
  if not FLAGS.prediction:
    config.classifiers, eval_config.classifiers = {}, {}
    config.learning_rate = FLAGS.lr
    eval_config.learning_rate = FLAGS.lr

    config.keep_prob = {'embedding_layer':FLAGS.keep_prob, 'rnn_layer':[FLAGS.keep_prob] * config.num_layers}
    vocabularies_size = []
    for key in word_to_id:
      if (key != 'word') and (key in FLAGS.classifiers_to_train):
        config.classifiers[key] = len(word_to_id[key])
        eval_config.classifiers[key] = len(word_to_id[key])
      elif key == 'word':
        config.vocab_size = len(word_to_id[key])
        eval_config.vocab_size = len(word_to_id[key])
      vocabularies_size += [f'{key}:{len(word_to_id[key])}']
    print(f"{'|'.join(vocabularies_size)}", flush=True)

  config.print()
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = Input(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = Model(is_training=True, config=config, input_=train_input, name='Train')
      tf.summary.scalar("Training Loss", m.get_cost(config.loss_to_view))
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = Input(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = Model(is_training=False, config=config, input_=valid_input, name='Valid')
      tf.summary.scalar("Validation Loss", mvalid.get_cost(config.loss_to_view))

    with tf.name_scope("Test"):
      test_input = Input(
          config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = Model(is_training=False, config=eval_config,
                         input_=test_input, name='Test')

    models = {"Train": m, "Valid": mvalid, "Test": mtest}
    for name, model in models.items():
      model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    if tf.__version__ < "1.1.0" and FLAGS.num_gpus > 1:
      raise ValueError("num_gpus > 1 is not supported for TensorFlow versions "
                       "below 1.1.0")
    soft_placement = False
    if FLAGS.num_gpus > 1:
      soft_placement = True
      util.auto_parallel(metagraph, m)

  with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
      model.import_ops()
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with tf.train.MonitoredTrainingSession(config=config_proto, checkpoint_dir=FLAGS.save_path) as session:
      if not FLAGS.prediction:
        # predict(session, mtest, FLAGS.classifiers_to_train, id_to_word)
        # exit(0)
        for i in range(config.max_max_epoch):
          lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          m.assign_lr(session, config.learning_rate * lr_decay)
          print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)), flush=True)
          train_perplexity = run_epoch(session, m, config.loss_to_view, eval_op=m.train_ops[config.train_op],
                                       verbose=True)
          print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity), flush=True)
          valid_perplexity = run_epoch(session, mvalid, config.loss_to_view)
          print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity), flush=True)

        test_perplexity = run_epoch(session, mtest, config.loss_to_view)
        print("Test Perplexity: %.3f" % test_perplexity, flush=True)

        if FLAGS.save_path:
          print("Saving model to %s." % FLAGS.save_path, flush=True)
          sv.saver.save(session, FLAGS.save_path, global_step=m._global_step)
      else:
        predict(session, mtest, FLAGS.classifiers_to_train, word_to_id)

if __name__ == "__main__":
  tf.app.run()