from collections import defaultdict, Counter

import pandas as pd


class GikryaReader:
    def __init__(self, filename, stop_words=None, sep='\t', lowercase=True,
                 joint_categories=True, n_ending_chars=None, pad_to=90,
                 shuffle=True, part='train', add_start_token=True, min_tf=10):

        self.joint_categories = joint_categories
        self.stop_words = [] if stop_words is None else stop_words
        self.df = self.load_df(filename, sep=sep, lowercase=lowercase, shuffle=shuffle, part=part)

        if n_ending_chars is not None:
            self.df = self.chunk_words(self.df, n_ends=n_ending_chars)
        if add_start_token:
            self.df = self.add_start_token(self.df)
        if pad_to:
            self.df = self.pad_df(self.df, pad_to)

        if joint_categories:
            vocabularies = self.get_joint_vocabulary(self.df, stop_words=self.stop_words, min_tf=min_tf)
            self.pos_gram_cats_vocabulary = vocabularies['pos_gram_cats_voc']
            self.num_classes = len(self.pos_gram_cats_vocabulary)
        else:
            vocabularies = self.get_vocabulary(self.df, stop_words=self.stop_words, min_tf=min_tf)
            self.pos_vocabulary = vocabularies['pos_voc']
            self.gram_cats_vocabulary = vocabularies['gram_cats_voc']
        self.vocabulary = vocabularies['tokens_voc']

    @staticmethod
    def add_start_token(df, field='tokens', start_token='_start_'):
        df = df.copy()
        df[field] = df[field].apply(lambda x: [start_token] + x)
        return df

    @staticmethod
    def pad_df(df, max_seq=90):
        df = df.copy()
        for column in df.columns:
            df[column] = df[column].apply(lambda x: x[:max_seq] + ['_pad_'] * max(0, (max_seq - len(x))))
        return df

    @staticmethod
    def encode_sentences(df, vocabulary, field='tokens'):
        return [[vocabulary[word] if word in vocabulary else len(vocabulary) for word in sentence] for sentence in
                df[field]]

    @staticmethod
    def encode_gram_cats(df, vocabulary):
        return [[vocabulary[gram_cat] for gram_cat in sentence] for sentence in df.gram_cats]

    @staticmethod
    def encode_POSs(df, vocabulary):
        return [[vocabulary[pos] for pos in sentence] for sentence in df.POSs]

    @staticmethod
    def encode_categories_jointly(df, vocabulary):
        output = []
        undefined = vocabulary.get('_pad_#_pad_')
        for POSs, gram_cats in zip(df.POSs, df.gram_cats):
            sentence = ['#'.join([pos, cat]) for pos, cat in zip(POSs, gram_cats)]
            output.append(
                [vocabulary[pos_gram_cat] if pos_gram_cat in vocabulary else undefined for pos_gram_cat in sentence])
        return output

    @staticmethod
    def get_vocabulary(df, offset=0, sent_field='tokens', min_tf=None, stop_words=[]):
        pos_vocabulary = dict(
            [(pos, i) for i, pos in enumerate(sorted(set([pos for sentence in df.POSs for pos in sentence])))])
        gram_cat_vocabulary = dict(
            [(cat, i) for i, cat in enumerate(sorted(set([cat for sentence in df.gram_cats for cat in sentence])))])
        if min_tf is None: min_tf = 1
        sentences = [word for sentence in df[sent_field] for word in sentence if word not in stop_words]
        vocabulary = dict([(word, cnt) for word, cnt in Counter(sentences).most_common() if cnt >= min_tf])
        vocabulary = dict([(word, index + offset) for index, (word, cnt) in enumerate(vocabulary.items())])
        return {'tokens_voc': vocabulary, 'pos_voc': pos_vocabulary, 'gram_cats_voc': gram_cat_vocabulary}

    @staticmethod
    def get_joint_vocabulary(df, offset=0, sent_field='tokens', min_tf=None, stop_words=[]):
        POSs = [pos for sentence in df.POSs for pos in sentence]
        gram_cats = [cat for sentence in df.gram_cats for cat in sentence]
        pos_gram_cats_vocabulary = dict((pos_gram_cat, i) for i, pos_gram_cat in enumerate(
            sorted(set(['#'.join([pos, cat]) for pos, cat in zip(POSs, gram_cats)]))))
        if min_tf is None: min_tf = 1
        sentences = [word for sentence in df[sent_field] for word in sentence if word not in stop_words]
        vocabulary = dict([(word, cnt) for word, cnt in Counter(sentences).most_common() if cnt >= min_tf])
        vocabulary = dict([(word, index + offset) for index, (word, cnt) in enumerate(vocabulary.items())])
        return {'tokens_voc': vocabulary, 'pos_gram_cats_voc': pos_gram_cats_vocabulary}

    @staticmethod
    def chunk_words(df, n_ends=3):
        df = df.copy()
        df.tokens = df.tokens.apply(lambda tokens: [token[-n_ends:] for token in tokens])
        return df

    @staticmethod
    def pad_rows(df, batch_size):
        df = df.copy()
        to_pad = len(df) % batch_size
        df = df.append(df.sample(n=to_pad, random_state=2019), ignore_index=True).reset_index(drop=True)
        return df

    # @staticmethod
    # def load__contiguous_df(filename, sep='\t', part='train', lowercase=True, sent_sep='_sep_', add_start_token=False, max_seq_len=90):
    #     df = GikryaReader.load_df(filename=filename, sep=sep, part=part, lowercase=lowercase, shuffle=False)
    #     new_df = pd.DataFrame()
    #     for field in ['IDs', 'tokens', 'lemmas', 'POSs', 'gram_cats']:
    #         sentences = [sentence + [sent_sep] for sentence in df[field]]
    #         sentences = [word for sentence in sentences for word in sentence]
    #         length = len(sentences)
    #         epoch_length = length // batch_size
    #         to_pad = () - len(sentences) % batch_size
    #         sentences = sentences + ['_pad_'] * to_pad
    #         sentences = [sentences[]]
    #         new_df[field] = sentences

    @staticmethod
    def load_df(filename, sep='\t', part='train', lowercase=True, shuffle=True):
        result, sentence = defaultdict(list), defaultdict(list)
        FIELDS = ['IDs', 'tokens', 'lemmas', 'POSs', 'gram_cats']

        def update_sentence(sentence, IDs=None, tokens=None, lemmas=None, POSs=None, gram_cats=None, lowercase=False):
            if lowercase: tokens = tokens.lower()
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
                    zip(FIELDS, line.split(sep)[:2] + ['_pad_'] * (len(FIELDS) - 2)))
                update_sentence(sentence, lowercase=lowercase, **kwargs)
            if len(sentence['IDs']) > 0:
                for field in FIELDS:
                    result[field].append(sentence[field])

        df = pd.DataFrame.from_dict(result)
        df['POSs_and_gram_cats'] = [['#'.join([pos, gram_cat]) for pos, gram_cat in zip(pos_values, gran_cat_values)]
                                    for pos_values, gran_cat_values in zip(df.POSs, df.gram_cats)]
        df['lengths'] = df.tokens.apply(len)
        df = df.sample(frac=1.0, random_state=2019) if shuffle else df
        return df

    # if __name__ == '__main__':
#     reader = GikryaReader('datasets/gikrya_train.txt', stop_words=['.', ','], pad_to=5)
#     print('dataset loaded!')
