import os
import pickle
import random
import numpy as np

import pandas as pd 
import torch
import torch.utils.data as data 
from nltk.tokenize import RegexpTokenizer

class Text_Dataset(data.Dataset):
    def __init__(self, data_dir, split, words_num, print_shape=False):
        self.words_num = words_num
        self.data_dir = data_dir
        self.split = split
        # self.device = device

        self.filenames, self.captions, self.idx2word, self.word2idx, self.n_word \
            = self.load_text_data(data_dir, split)

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        if not data_dir.find('coco'):
            test_names = self.load_filenames(data_dir, 'test')
        else:
            test_names = self.load_filenames(data_dir, 'val')

        if not os.path.isfile(filepath):
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, idx2word, word2idx, n_words = self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             idx2word, word2idx], f, protocol=2)
                print('Save to: ', filepath)


        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                idx2word, word2idx = x[2], x[3]
                del x
                n_words = len(idx2word)
                print('Load from: ', filepath)

        if split == 'train':
            filenames = train_names
            captions = train_captions
        
        else:
            filenames = test_names
            captions = test_captions
        
        return filenames, captions, idx2word, word2idx, n_words

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        idx2word = {}
        idx2word[0] = '<end>'

        word2idx = {}
        word2idx['<end>'] = 0
        ix = 1
        for w in vocab:
            word2idx[w] = ix
            idx2word[ix] = w
            ix += 1

        # Add Begin of Sentence token
        # Remove this token when adapt this weights to the main model
        # word2idx['<bos>'] = ix
        # idx2word[ix] = '<bos>'

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in word2idx:
                    rev.append(word2idx[w])
            rev.append(0)
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in word2idx:
                    rev.append(word2idx[w])
            rev.append(0)
            test_captions_new.append(rev)

        return train_captions_new, test_captions_new, idx2word, word2idx, len(idx2word)


    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            print("cap_path", cap_path)

            with open(cap_path, "r", encoding='utf-8') as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
            print('cap cnt', cnt)

        return all_captions

    def load_filenames(self, data_dir, split):
        filepath = '%s/filenames/%s/filenames.pickle' % (data_dir, split)

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        
        return filenames

    def get_caption(self, sent_idx):
        sent_caption = np.asarray(self.captions[sent_idx]).astype('int64')
        
        # Ignore this warning
        # if (sent_caption == 0).sum() > 0:
        #     print('ERROR: do not need END (0) token', sent_caption)

        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.zeros((self.words_num), dtype='int64')
        # x = np.zeros((self.words_num, 1), dtype='int64')

        x_len = num_words
        if num_words <= self.words_num:
            x[:num_words] = sent_caption

        else: # For LM pretraining 
            x[:] = sent_caption[:self.words_num]
            x_len = self.words_num

        return x, x_len
 
    def __len__(self):
        return len(self.captions)


    def __getitem__(self, idx):
        data_dir = self.data_dir
        caps, cap_len = self.get_caption(idx)
        
        return caps, cap_len

def prepare_data(data):
    caps, cap_len = data
    sorted_cap_len, sorted_cap_idx = torch.sort(cap_len, 0, True)

    caps = caps[sorted_cap_idx].squeeze()

    if torch.cuda.is_available():
        caps =caps.cuda()
        sorted_cap_len = sorted_cap_len.cuda()
    else:
        sorted_cap_len = sorted_cap_len
        pass
    return caps, sorted_cap_len