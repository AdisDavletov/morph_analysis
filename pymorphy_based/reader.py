# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Генератор батчей с определёнными параметрами.

import sys
from collections import namedtuple
from typing import List, Tuple

import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

sys.path.append('../')
from vectorizers.endings_vectorizer import EndingsVectorizer
from vectorizers.grammems_vectorizer import GrammemsVectorizer
from vectorizers.process_tag import convert_from_opencorpora_tag, process_gram_tag

WordForm = namedtuple("WordForm", "text gram_vector_index")


class BatchGenerator:
    """
    Генератор наборов примеров для обучения.
    """

    def __init__(self,
                 file_names: List[str],
                 train_config,
                 grammeme_vectorizer_input: GrammemsVectorizer,
                 grammeme_vectorizer_output: GrammemsVectorizer,
                 endings_vectorizer: EndingsVectorizer,
                 indices: np.array,
                 build_config):
        self.file_names = file_names  # type: List[str]
        # Параметры батчей.
        self.batch_size = train_config.external_batch_size  # type: int
        self.bucket_borders = train_config.sentence_len_groups  # type: List[Tuple[int]]
        self.buckets = [list() for _ in range(len(self.bucket_borders))]
        self.build_config = build_config
        # Разбиение на выборки.
        self.indices = indices  # type: np.array
        # Подготовленные словари.
        self.grammeme_vectorizer_input = grammeme_vectorizer_input  # type: GrammemeVectorizer
        self.grammeme_vectorizer_output = grammeme_vectorizer_output  # type: GrammemeVectorizer
        self.endings_vectorizer = endings_vectorizer
        self.morph = MorphAnalyzer()
        self.converter = converters.converter('opencorpora-int', 'ud14')

    def __to_tensor(self, sentences):

        n = len(sentences)
        grammemes_count = self.grammeme_vectorizer_input.grammemes_count()
        sentence_max_len = max([len(sentence) for sentence in sentences])

        data = {}
        target = {}

        weights = np.zeros((n, sentence_max_len), dtype=np.float)
        words = np.zeros((n, sentence_max_len), dtype=np.int)
        grammemes = np.zeros((n, sentence_max_len, grammemes_count), dtype=np.float)
        y = np.zeros((n, sentence_max_len), dtype=np.int)

        for i, sentence in enumerate(sentences):
            word_indices, gram_vectors = self.get_sample(
                [x.text for x in sentence],
                converter=self.converter,
                morph=self.morph,
                grammeme_vectorizer=self.grammeme_vectorizer_input,
                endings_vectorizer=self.endings_vectorizer)
            assert len(word_indices) == len(sentence) and \
                   len(gram_vectors) == len(sentence)

            weights[i, -len(sentence):] = 1.
            words[i, -len(sentence):] = word_indices
            grammemes[i, -len(sentence):] = gram_vectors
            y[i, -len(sentence):] = [word.gram_vector_index + 1 for word in sentence]
        if self.build_config.use_endings:
            data['endings'] = words
        if self.build_config.use_gram:
            data['grammems'] = grammemes
        data['weights'] = weights
        target['main'] = y

        if self.build_config.use_pos_lm:
            y_prev = np.zeros_like(y)
            y_prev[:, 1:] = y[:, :-1]
            target['pred'] = y_prev
            y_next = np.zeros_like(y)
            y_next[:, :-1] = y[:, 1:]
            target['next'] = y_next
        return data, target

    @staticmethod
    def get_sample(sentence: List[str],
                   converter,
                   morph: MorphAnalyzer,
                   grammeme_vectorizer: GrammemsVectorizer,
                   endings_vectorizer: EndingsVectorizer):

        word_gram_vectors = []
        word_indices = []
        for word in sentence:
            gram_value_indices = np.zeros(grammeme_vectorizer.grammemes_count())

            # Индексы слов.
            word_index = endings_vectorizer.get_index(word)
            word_indices.append(word_index)

            for parse in morph.parse(word):
                pos, gram = convert_from_opencorpora_tag(converter, parse.tag, word)
                gram = process_gram_tag(gram)
                gram_value_indices += np.array(grammeme_vectorizer.get_vector(pos + "#" + gram))

            # Нормируем по каждой категории отдельно.
            sorted_grammemes = sorted(grammeme_vectorizer.all_grammemes.items(), key=lambda x: x[0])
            index = 0
            for category, values in sorted_grammemes:
                mask = gram_value_indices[index:index + len(values)]
                s = sum(mask)
                gram_value_indices[index:index + len(values)] = mask / s
                index += len(values)
            word_gram_vectors.append(gram_value_indices)

        return word_indices, word_gram_vectors

    def __iter__(self):
        """
        Получение очередного батча.

        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        last_sentence = []
        i = 0
        for filename in self.file_names:
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        if i not in self.indices:
                            last_sentence = []
                            i += 1
                            continue
                        for index, bucket in enumerate(self.buckets):
                            if self.bucket_borders[index][0] <= len(last_sentence) < self.bucket_borders[index][1]:
                                bucket.append(last_sentence)
                            if len(bucket) >= self.batch_size:
                                yield self.__to_tensor(bucket)
                                self.buckets[index] = []
                        last_sentence = []
                        i += 1
                    else:
                        word, _, pos, tags = line.split('\t')[1:5]
                        gram_vector_index = self.grammeme_vectorizer_output.get_index_by_name(pos + "#" + tags)
                        last_sentence.append(WordForm(text=word, gram_vector_index=gram_vector_index))
        for index, bucket in enumerate(self.buckets):
            yield self.__to_tensor(bucket)
