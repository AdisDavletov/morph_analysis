BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1'
DATA_DIR = '/content/drive/My Drive/DEEP LEARNING/datasets/POS_TAGGING/gikrya/'

import torch
import bert
import tensorflow as tf
import tensorflow_hub as hub

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)

def create_input_features(df, max_seq=90, field='tokens'):
    df = df.copy()
    df[field] = df[field].apply(lambda x: ' '.join(x))
    input_examples = df.apply(lambda x: bert.run_classifier.InputExample(guid=None,
                                                                         text_a=x[field],
                                                                         text_b=None,
                                                                         label=None), axis=1)

    tokenizer = create_tokenizer_from_hub_module()

    features = bert.run_classifier.convert_examples_to_features(input_examples, [None], max_seq, tokenizer)
    return features, tokenizer
