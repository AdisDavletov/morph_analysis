# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
from collections import defaultdict

import tensorflow as tf

Py3 = sys.version_info[0] == 3
FLDS = "word|pos|case|gender|number|animacy|tense|person|verbform|mood|variant|degree|numform"


def _parse_lines(lines, data_buf, lower=False, suffix=-1):
    cur_sent = defaultdict(list)
    for line in lines:
        line = line.lower().strip() if lower else line.strip()
        if line == "":
            if len(cur_sent) > 0:
                _update_data_buf(data_buf, cur_sent)
            cur_sent = defaultdict(list)
            continue
        splitted = line.split()
        if len(splitted) == 5:
            word, _, pos, tags = splitted[1:]
        elif len(splitted) == 4:
            word, pos, tags = splitted[1:]
        else:
            raise ValueError("Each line should have 4 or 5 columns")
        word = word[-suffix:] if suffix > 0 else word
        curr_step = {}
        curr_step['word'], curr_step['pos'] = word, pos
        if tags != '_':
            tags = tags.lower().split('|')
            for tag in tags:
                tag = tag.split('=')
                curr_step[tag[0]] = tag[1]
        for fld in FLDS.split('|'):
            try:
                val = curr_step[fld]
                cur_sent[fld].append(val)
            except KeyError:
                cur_sent[fld].append(f"<{fld}_unk>")
    if len(cur_sent) > 0:
        _update_data_buf(data_buf, cur_sent)


def _update_data_buf(data_buf, cur_sent):
    for fld in cur_sent:
        eos = '<eos>' if fld == 'word' else f"<{fld}_unk>"
        data_buf[fld] += (cur_sent[fld] + [eos])


def _read_words(filename, with_tags_and_pos=True, lower=False, suffix=-1):
    data = defaultdict(list)
    if with_tags_and_pos:
        with tf.gfile.GFile(filename, "r") as f:
            lines = f.readlines()
            _parse_lines(lines, data, lower, suffix)
    else:
        _construct_data(filename, data, lower, suffix)
    return data


def _construct_data(filename, data, lower, suffix=-1):
    words = _read_words_2(filename, lower)
    data['word'] = [word[-suffix:] for word in words] if suffix > 0 else words
    for word in words:
        for fld in FLDS.split('|')[1:]:
            data[fld].append(f"{fld}_unk")


def _read_words_2(filename, lower=False, to_replace='\n\n'):
    with tf.gfile.GFile(filename, "r") as f:
        if lower:
            return f.read().replace(to_replace, " <eos> ").lower().split()
        else:
            return f.read().replace(to_replace, " <eos> ").split()


def _build_vocab(filename, additional_file=None, lower=False):
    data = _read_words(filename, lower=lower)
    word_to_id = {}
    extra_words = []
    if additional_file:
        extra_words = _read_words_2(additional_file, lower)
    for key, toks in data.items():
        if key == 'word':
            toks += extra_words
        counter = collections.Counter(toks)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id[key] = dict(zip(words, range(len(words))))
    return word_to_id


def _file_to_word_ids(filename, word_to_id, with_tags_and_pos=True, lower=False):
    data = _read_words(filename, with_tags_and_pos, lower)
    res = {}
    for key, toks in data.items():
        res[key] = [word_to_id[key][word] for word in toks if word in word_to_id[key]]
    return res


def ptb_raw_data(data_path=None,
                 word_to_id=None,
                 train="gikrya_new_train.out",
                 dev="gikrya_new_test.out",
                 test="gikrya_new_test.out",
                 additional_file=None,
                 with_tags_and_pos=True,
                 lower=True):
    """Load PTB raw data from data directory "data_path".
    Reads PTB text files, converts strings to integer ids,
    and performs mini-batching of the inputs.
    The PTB dataset comes from Tomas Mikolov's webpage:
    http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
    Args:
      data_path: string path to the directory where simple-examples.tgz has
        been extracted.
    Returns:
      tuple (train_data, valid_data, test_data, vocabulary)
      where each of the data objects can be passed to PTBIterator.
    """

    train_path = os.path.join(data_path, train)
    valid_path = os.path.join(data_path, dev)
    test_path = os.path.join(data_path, test)
    if word_to_id is None:
        word_to_id = _build_vocab(train_path, additional_file, lower)
    train_data = _file_to_word_ids(train_path, word_to_id, with_tags_and_pos, lower)
    valid_data = _file_to_word_ids(valid_path, word_to_id, with_tags_and_pos, lower)
    test_data = _file_to_word_ids(test_path, word_to_id, with_tags_and_pos, lower)

    return train_data, valid_data, test_data, word_to_id


def ptb_producer(raw_data, batch_size, num_steps, name=None):
    """Iterate on the raw PTB data.
    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.
    Args:
      raw_data: one of the raw data outputs from ptb_raw_data.
      batch_size: int, the batch size.
      num_steps: int, the number of unrolls.
      name: the name of this operation (optional).
    Returns:
      A pair of Tensors, each shaped [batch_size, num_steps]. The second element
      of the tuple is the same data time-shifted to the right by one.
    Raises:
      tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
    """
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        for key, data in raw_data.items():
            raw_data[key] = tf.convert_to_tensor(data, name=f"{key}_raw_data", dtype=tf.int32)

        data_len = tf.size(raw_data['word'])
        batch_len = data_len // batch_size
        data = {}
        for key in raw_data:
            data[key] = tf.reshape(raw_data[key][0: batch_size * batch_len],
                                   [batch_size, batch_len])

        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()

        res = {}
        for key in data:
            res[key] = tf.strided_slice(data[key], [0, i * num_steps],
                                        [batch_size, (i + 1) * num_steps])
            res[key].set_shape([batch_size, num_steps])

        res['word_target'] = tf.strided_slice(data['word'], [0, i * num_steps + 1],
                                              [batch_size, (i + 1) * num_steps + 1])
        res['word_target'].set_shape([batch_size, num_steps])
    return res

import pandas as pd

def load_df(filename, sep='\t', part='train'):
    result, sentence = defaultdict(list), defaultdict(list)
    FIELDS = ['IDs', 'tokens', 'lemmas', 'POSs', 'gram_cats']

    def update_sentence(sentence, IDs=None, tokens=None, lemmas=None, POSs=None, gram_cats=None):
        sentence['IDs'].append(IDs)
        sentence['tokens'].append(tokens)
        sentence['lemmas'].append(lemmas)
        sentence['POSs'].append(POSs)
        sentence['gram_cats'].append(gram_cats)

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 and len(sentence['IDs']) > 0:
                for field in FIELDS:
                    result[field].append(sentence[field])
                sentence = defaultdict(list)
                continue
            kwargs = dict(zip(FIELDS, line.split(sep))) if part == 'train' else dict(
                zip(FIELDS, line.split(sep)[:2] + ['<unk>'] * (len(FIELDS) - 2)))
            update_sentence(sentence, **kwargs)
        if len(sentence['IDs']) > 0:
            for field in FIELDS:
                result[field].append(sentence[field])

    return pd.DataFrame.from_dict(result)

if __name__ == '__main__':
    load_df('datasets/gikrya_train.txt')
