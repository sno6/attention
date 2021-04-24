from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch


class RecipeDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_json(file_path)

        self.enc_vocab = self.build_enc_vocab(self.data)
        self.dec_quantity_vocab = self.build_quantity_vocab(self.data)
        self.dec_unit_vocab = self.build_unit_vocab(self.data)

        # Find the largest sentence in terms of words to ensure accurate padding
        # of smaller sentences.
        self.largest_sentence = self.find_largest_sentence(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sample = {
            'x': torch.tensor(self.encode_sentence(item['x'])).float(),
            'y': {
                'quantity': torch.tensor(self.encode_quantity(item['y']['quantity'])).float(),
                'unit': torch.tensor(self.encode_unit(item['y']['unit'])).float()
            }
        }

        return sample

    def find_largest_sentence(self, data):
        largest = 0
        for sample in data.iloc:
            sentence = sample['x']
            n_words = len(sentence.split(' '))
            if n_words > largest:
                largest = n_words

        return largest

    def encode_quantity(self, q: int):
        assert q in self.dec_quantity_vocab

        idx = self.dec_quantity_vocab[q]
        hot = np.zeros(len(self.dec_quantity_vocab))
        hot[idx] = 1

        return hot

    def decode_quantity(self, q_vec):
        idx = torch.argmax(q_vec)
        for k in self.dec_quantity_vocab:
            if self.dec_quantity_vocab[k] == idx:
                return k

        return '<UNK>'

    def encode_unit(self, u: int):
        assert u in self.dec_unit_vocab

        idx = self.dec_unit_vocab[u]
        hot = np.zeros(len(self.dec_unit_vocab))
        hot[idx] = 1

        return hot

    def decode_unit(self, u_vec):
        idx = torch.argmax(u_vec)
        for k in self.dec_unit_vocab:
            if self.dec_unit_vocab[k] == idx:
                return k

        return '<UNK>'

    def encode_word(self, w: str):
        w = self.clean_word(w)

        if w not in self.enc_vocab:
            return np.zeros(len(self.enc_vocab))

        idx = self.enc_vocab[w]
        hot = np.zeros(len(self.enc_vocab))
        hot[idx] = 1

        return hot

    def decode_word(self, w_vec):
        idx = torch.argmax(w_vec)
        for k in self.enc_vocab:
            if self.enc_vocab[k] == idx:
                return k

        return '<UNK>'

    def encode_sentence(self, sentence: str):
        sentence = [self.encode_word(w) for w in sentence.split(' ')]
        if len(sentence) < self.largest_sentence:
            # Add padding to the sentence if we don't have enough word vectors.
            diff = self.largest_sentence - len(sentence)
            for _ in range(0, diff):
                sentence.append(np.zeros(len(self.enc_vocab)))

        return sentence

    def decode_sentence(self, s_vec):
        sentence = [self.decode_word(w) for w in s_vec]
        return " ".join(sentence)

    def clean_word(self, s: str):
        s = s.lower().strip()
        return s

    def build_enc_vocab(self, data):
        vocab = {'<PAD>': 0}

        # Start index at 1, reserve idx 0 for null/padding
        idx = 1

        for sample in data.iloc:
            for word in sample['x'].split(' '):
                word = self.clean_word(word)
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1

        return vocab

    def build_unit_vocab(self, data):
        vocab = {}
        idx = 0

        for sample in data.iloc:
            q = sample['y']['unit']
            if q not in vocab:
                vocab[q] = idx
                idx += 1

        return vocab

    def build_quantity_vocab(self, data):
        vocab = {}
        idx = 0

        for sample in data.iloc:
            q = sample['y']['quantity']
            if q not in vocab:
                vocab[q] = idx
                idx += 1

        return vocab
