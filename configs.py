import copy
import json


class BuildConfig(object):
    def __init__(self,
                 endings_inp_drop=0.3,
                 gram_inp_drop=0.3,
                 rnn_state_drop=0.01,
                 dense_drop=0.2,
                 use_endings=True,
                 endings_emb_size=50,
                 use_gram=True,
                 rnn_hidden_size=128,
                 seed=2019,
                 merge_mode='ave',
                 n_rnn_layers=3,
                 dense_size=128,
                 optimizer='adam',
                 use_pos_lm=True,
                 use_wd=True,
                 wd=0.0002,
                 n_endings=3
                 ):
        self.endings_inp_drop = endings_inp_drop
        self.gram_inp_drop = gram_inp_drop
        self.rnn_state_drop = rnn_state_drop
        self.dense_drop = dense_drop
        self.use_endings = use_endings
        self.endings_emb_size = endings_emb_size
        self.use_gram = use_gram

        self.rnn_hidden_size = rnn_hidden_size
        self.seed = seed
        self.merge_mode = merge_mode
        self.n_rnn_layers = n_rnn_layers
        self.dense_size = dense_size
        self.optimizer = optimizer
        self.use_pos_lm = use_pos_lm

        self.use_wd = use_wd
        self.wd = wd
        self.n_endings = n_endings

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            deep_copy = copy.deepcopy(self.__dict__)
            f.write(json.dumps(deep_copy, sort_keys=True, indent=4) + '\n')

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            deep_copy = json.loads(f.read())
            self.__dict__.update(deep_copy)


class TrainConfig(object):
    def __init__(self,
                 sentence_len_groups=[[26, 40],[40,50],[15,25],[1,6],[7,14]],
                 random_seed=2019,
                 val_part=0.1,
                 external_batch_size=5000,
                 n_epochs=1
                 ):
        self.sentence_len_groups = sentence_len_groups
        self.random_seed = random_seed
        self.val_part = val_part
        self.external_batch_size = external_batch_size
        self.n_epochs = n_epochs

    def save(self, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            deep_copy = copy.deepcopy(self.__dict__)
            f.write(json.dumps(deep_copy, sort_keys=True, indent=4) + '\n')

    def load(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            deep_copy = json.loads(f.read())
            self.__dict__.update(deep_copy)

# if __name__ == '__main__':
#     train_config = {
#         'n_epochs': 10,
#         'batch_size': 5
#     }
#     build_config = {
#         'use_word_embeddings': True,
#         'dropout': 0.3
#     }
#     train_config = TrainConfig()
#     build_config = BuildConfig(**build_config)
#
#     # print(train_config.batch_size, train_config.n_epochs)
#
#     if build_config.use_word_embeddings == True:
#         print(build_config.dropout, build_config.use_word_embeddings)
