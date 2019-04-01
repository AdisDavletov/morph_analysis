import numpy as np
import torch
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertForTokenClassification, BertModel
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler
from torch import nn
# from datasets import reader
from tqdm import tqdm_notebook


class BertForTokenClassificationNew(BertForTokenClassification):
    def __init__(self, config, num_labels, layers='0,6,11'):
        super().__init__(config, num_labels)
        # self.bert = BertModel(config)
        # self.num_labels = num_labels
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)
        self.layers = [int(x) for x in layers.split(',')]

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        sequence_output = [layer for i, layer in enumerate(sequence_output) if i in self.layers]
        sequence_output = torch.mean(torch.stack(sequence_output), dim=0)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cuda'):
        device = torch.device(device)
        self.load_state_dict(torch.load(filename, map_location=device))


class BertMorphAnalyzer:
    def __init__(self, build_config, train_config):
        self.bert_model = build_config.bert_model
        self.tokenizer = None
        self.model = None
        self.build_config = build_config
        self.train_config = train_config
        self.jointly_classification = build_config.jointly_classification
        self.num_labels = None
        self.pos_num_labels = None
        self.gram_cats_num_labels = None
        self.max_seq_len = build_config.max_seq_len
        self.pos2idx = None
        self.gram_cats2idx = None
        self.pos_gram_cats2idx = None
        self.validation_size = train_config.validation_size
        self.full_fine_tune = train_config.full_fine_tune
        self.train_dataloader = None
        self.valid_dataloader = None
        self.layers = build_config.layers
        pass

    def prepare(self, filename):
        df = GikryaReader.load_df(filename, shuffle=False, lowercase='uncased' in self.bert_model)
        inputs, targets, masks = self.to_features(df)
        split = self.split_data(inputs, targets, masks)
        if not self.jointly_classification:
            raise NotImplementedError()
        else:
            tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks = split
        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_config.batch_size,
                                           drop_last=False)

        valid_data = TensorDataset(val_inputs, val_masks, val_tags)
        valid_sampler = SequentialSampler(valid_data)
        self.valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.train_config.batch_size,
                                           drop_last=False)

        self.model = BertForTokenClassificationNew.from_pretrained(self.bert_model, num_labels=self.num_labels, layers=self.layers)
        self.model.cuda()

    def train(self, log_dir=None, chkp_dir=None):
        if self.full_fine_tune:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': self.train_config.wd},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': 0.0}
            ]
        else:
            param_optimizer = list(self.model.classifier.named_parameters())
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
        optimizer = Adam(optimizer_grouped_parameters, lr=self.train_config.lr)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def flat_accuracy(preds, labels, mask):
            mask = mask.flatten() == 1.0
            pred_flat = np.argmax(preds, axis=2).flatten()[mask]
            labels_flat = labels.flatten()[mask]
            return np.sum(pred_flat == labels_flat) / len(labels_flat)

        tr_len = len(self.train_dataloader.dataset)
        val_len = len(self.valid_dataloader.dataset)
        for i in range(self.train_config.epochs):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            tr_extra_batch = 1 if tr_len % self.train_config.batch_size > 0 else 0
            tr_total = tr_len // self.train_config.batch_size + tr_extra_batch
            tr_progress = tqdm_notebook(self.train_dataloader, total=tr_total)
            for step, batch in enumerate(tr_progress):
                # add batch to gpu
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                # forward pass
                loss = self.model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
                # backward pass
                loss.backward()
                # track train loss
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
                # gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=self.train_config.max_grad_norm)
                # update parameters
                optimizer.step()
                self.model.zero_grad()
                tr_progress.set_postfix_str(f'{tr_loss / nb_tr_steps}')
            # print train loss per epoch
            print("Train loss: {}".format(tr_loss / nb_tr_steps))
            # VALIDATION on validation set
            self.model.eval()
            val_extra_batch = 1 if val_len % self.train_config.batch_size > 0 else 0
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            predictions, true_labels = [], []

            val_total = val_len // self.train_config.batch_size + val_extra_batch
            val_progress = tqdm_notebook(self.valid_dataloader, total=val_total)
            for batch in val_progress:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    tmp_eval_loss = self.model(b_input_ids, token_type_ids=None,
                                               attention_mask=b_input_mask, labels=b_labels)
                    logits = self.model(b_input_ids, token_type_ids=None,
                                        attention_mask=b_input_mask)
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                mask = b_input_mask.to('cpu').numpy()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.append(label_ids)

                tmp_eval_accuracy = flat_accuracy(logits, label_ids, mask)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += b_input_ids.size(0)
                nb_eval_steps += 1
            eval_loss = eval_loss / nb_eval_steps
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        return self

    def split_data(self, inputs, targets, attention_masks):
        if self.jointly_classification:
            tr_inp, val_inp, \
            tr_trg, val_trg, \
            tr_mask, val_mask = \
                train_test_split(inputs, targets,
                                 attention_masks, random_state=2019,
                                 test_size=self.validation_size)
            tr_trg = torch.tensor(tr_trg)
            val_trg = torch.tensor(val_trg)
        else:
            tr_inp, val_inp, \
            tr_p_trg, val_p_trg, \
            tr_g_trg, val_g_trg, \
            tr_mask, val_mask = \
                train_test_split(inputs, targets[0], targets[1],
                                 attention_masks, random_state=2019,
                                 test_size=self.validation_size)
            tr_p_trg = torch.tensor(tr_p_trg)
            val_p_trg = torch.tensor(val_p_trg)
            tr_g_trg = torch.tensor(tr_g_trg)
            val_g_trg = torch.tensor(val_g_trg)

        tr_inp = torch.tensor(tr_inp)
        val_inp = torch.tensor(val_inp)
        tr_mask = torch.tensor(tr_mask)
        val_mask = torch.tensor(val_mask)

        if self.jointly_classification:
            split = (tr_inp, val_inp, tr_trg, val_trg, tr_mask, val_mask)
        else:
            split = (tr_inp, val_inp, tr_p_trg, val_p_trg, tr_g_trg, val_g_trg, tr_mask, val_mask)
        return split

    def to_features(self, df):
        df, all_pos, all_gram_cats, all_pos_gram_cats = self.prepare_df(df)
        df.lengths.hist()
        self.pos2idx = all_pos
        self.gram_cats2idx = all_gram_cats
        self.pos_gram_cats2idx = all_pos_gram_cats

        if self.jointly_classification:
            self.num_labels = len(all_pos_gram_cats)
        else:
            pos_num_labels = len(set([cat for sentence in df.POSs for cat in sentence]))
            gram_cat_num_labels = len(set([cat for sentence in df.gram_cats for cat in sentence]))
            self.pos_num_labels = pos_num_labels
            self.gram_cats_num_labels = gram_cat_num_labels

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(sentence) for sentence in df.tokens],
                                  maxlen=self.max_seq_len, dtype="long", truncating='post', padding='post')

        if self.jointly_classification:
            targets = pad_sequences(
                [[self.pos_gram_cats2idx.get(cat) for cat in sentence] for sentence in df.POSs_and_gram_cats],
                value=0, maxlen=self.max_seq_len, truncating='post', padding='post', dtype="long"
            )
        else:
            pos_targets = pad_sequences(
                [[self.pos2idx.get(pos) for pos in sentence] for sentence in df.POSs],
                value=0, maxlen=self.max_seq_len, truncating='post', padding='post', dtype="long"
            )
            gram_cats_targets = pad_sequences(
                [[self.gram_cats2idx.get(cat) for cat in sentence] for sentence in df.gram_cats],
                value=0, maxlen=self.max_seq_len, truncating='post', padding='post', dtype="long"
            )
            targets = (pos_targets, gram_cats_targets)
        attention_masks = [[float(x) for x in sentence][:self.max_seq_len] + [0.] * (
            len(input_id) - min(self.max_seq_len, len(sentence))) for sentence, input_id in
                           zip(df.correspondence, input_ids)]
        return input_ids, targets, attention_masks

    def prepare_df(self, df, replicate_to_sub_words=False, fill_sub_word='X'):
        df = df.copy()
        all_gram_cats, all_pos = set(), set()
        all_pos_gram_cats = set()
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model,
                                                       do_lower_case='uncased' in self.bert_model)
        all_new_tokens, all_new_gram_cats, all_new_pos = [], [], []
        all_new_pos_and_gram_cats = []
        correspondence = [list() for _ in range(len(df))]

        for i, (_, row) in enumerate(df.iterrows()):
            all_gram_cats.update(row.gram_cats)
            all_pos.update(row.POSs)
            all_pos_gram_cats.update(row.POSs_and_gram_cats)
            new_tokens = []
            new_gram_cats, new_POSs, new_POSs_and_gram_cats = [], [], []

            for j, token in enumerate(row.tokens):
                sub_words = self.tokenizer.tokenize(token)
                new_tokens.extend(sub_words)
                gram_cat = row.gram_cats[j]
                pos = row.POSs[j]
                joined = row.POSs_and_gram_cats[j]
                extra = len(sub_words) - 1
                if replicate_to_sub_words:
                    gram_cat = [gram_cat] + [gram_cat] * extra
                    pos = [pos] + [pos] * extra
                    joined = [joined] + [joined] * extra
                else:
                    gram_cat = [gram_cat] + [fill_sub_word] * extra
                    pos = [pos] + [fill_sub_word] * extra
                    joined = [joined] + [fill_sub_word] * extra

                new_POSs.extend(pos)
                new_gram_cats.extend(gram_cat)
                new_POSs_and_gram_cats.extend(joined)

                correspondence[i].extend([True] + [False] * extra)

            all_new_tokens.append(new_tokens)
            all_new_pos.append(new_POSs)
            all_new_gram_cats.append(new_gram_cats)
            all_new_pos_and_gram_cats.append(new_POSs_and_gram_cats)

        df['tokens'] = all_new_tokens
        df['POSs'] = all_new_pos
        df['gram_cats'] = all_new_gram_cats
        df['POSs_and_gram_cats'] = all_new_pos_and_gram_cats
        df['correspondence'] = correspondence
        df['lengths'] = df.tokens.apply(len)
        all_pos = [fill_sub_word] + sorted(all_pos) if not replicate_to_sub_words else sorted(all_pos)
        all_gram_cats = [fill_sub_word] + sorted(all_gram_cats) if not replicate_to_sub_words else sorted(
            all_gram_cats)
        all_pos_gram_cats = [fill_sub_word] + sorted(
            all_pos_gram_cats) if not replicate_to_sub_words else sorted(
            all_pos_gram_cats)

        all_pos = dict([(pos, i) for i, pos in enumerate(all_pos)])
        all_gram_cats = dict([(gram_cat, i) for i, gram_cat in enumerate(all_gram_cats)])
        all_pos_gram_cats = dict([(joint, i) for i, joint in enumerate(all_pos_gram_cats)])
        return df, all_pos, all_gram_cats, all_pos_gram_cats
