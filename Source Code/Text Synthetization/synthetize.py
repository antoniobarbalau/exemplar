import numpy as np
import tensorflow as tf
import os
import json
import torch as T
import argparse

from model import VAE
from dataset import get_iterators
from helper_functions import get_cuda, get_sentences_in_batch
import torch.nn.functional as F


import tensorflow as tf
from optimizer import optimize
import pickle
import argparse
import re

CLASS = 0
TEMPERATURE = 0.1

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_vocab', type=int, default=12000)
parser.add_argument('--epochs', type=int, default=121)
parser.add_argument('--n_hidden_G', type=int, default=512)
parser.add_argument('--n_layers_G', type=int, default=2)
parser.add_argument('--n_hidden_E', type=int, default=512)
parser.add_argument('--n_layers_E', type=int, default=1)
parser.add_argument('--n_z', type=int, default=128)
parser.add_argument('--word_dropout', type=float, default=0.5)
parser.add_argument('--rec_coef', type=float, default=7)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gpu_device', type=int, default=1)
parser.add_argument('--n_highway_layers', type=int, default=2)
parser.add_argument('--n_embed', type=int, default=300)
parser.add_argument('--unk_token', type=str, default="<unk>")
parser.add_argument('--pad_token', type=str, default="<pad>")
parser.add_argument('--start_token', type=str, default="<sos>")
parser.add_argument('--end_token', type=str, default="<eos>")
opt = parser.parse_args()

my_punc = "!\"#$%&\()*+?_/:;[]{}|~,`"
table = dict((ord(char), u' ') for char in my_punc)

def clean_str(string):
    string = re.sub(r"\'s ", " ", string)
    string = re.sub(r"\'m ", " ", string)
    string = re.sub(r"\'ve ", " ", string)
    string = re.sub(r"n\'t ", " not ", string)
    string = re.sub(r"\'re ", " ", string)
    string = re.sub(r"\'d ", " ", string)
    string = re.sub(r"\'ll ", " ", string)
    string = re.sub("-", " ", string)
    string = re.sub(r"@", " ", string)
    string = re.sub('\'', '', string)
    string = string.translate(table)
    string = string.replace("..", "").strip()
    return string

class BlackBox(object):
    def __init__(self):
        with open('tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)

        self.model = tf.keras.models.load_model('classifier.h5')

    def predict(self, texts):
        texts = [clean_str(t) for t in texts]
        sequences = self.tokenizer.texts_to_sequences(texts)
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, 128)
        return self.model.predict(sequences)

class Generator(object):
    def __init__(self):
        _, _, self.vocab = get_iterators(opt)

        self.vae = VAE(opt)
        self.vae.embedding.weight.data.copy_(self.vocab.vectors)

        self.vae = get_cuda(self.vae)
        checkpoint = T.load('data/saved_models/vae_model.121.pyt')
        self.vae.load_state_dict(checkpoint['vae_dict'])
        self.vae.eval()
        del checkpoint

    def generate(self, encodings):
        sentences = []
        for z in encodings.numpy():
            z = get_cuda(T.from_numpy(z)).view((1, -1))
            h_0 = get_cuda(T.zeros(opt.n_layers_G, 1, opt.n_hidden_G))
            c_0 = get_cuda(T.zeros(opt.n_layers_G, 1, opt.n_hidden_G))
            G_hidden = (h_0, c_0)
            G_inp = T.LongTensor(1,1).fill_(self.vocab.stoi[opt.start_token])
            G_inp = get_cuda(G_inp)
            sentence = opt.start_token+" "
            num_words = 0
            while G_inp[0][0].item() != self.vocab.stoi[opt.end_token]:
                with T.autograd.no_grad():
                    logit, G_hidden, _ = self.vae(None, G_inp, z, G_hidden)
                probs = F.softmax(logit[0] / TEMPERATURE, dim=1)
                G_inp = T.multinomial(probs,1)
                sentence += (self.vocab.itos[G_inp[0][0].item()]+" ")
                num_words += 1
                if num_words > 64:
                    break
            sentence = sentence.replace('<unk>', '').replace('<sos>', '').replace('<eos>', '').replace('<pad>', '')
            sentences.append(sentence)

        return sentences

optimize(
    np.eye(2)[CLASS].astype(np.float32),
    BlackBox(),
    Generator(),
    population_size = 256,
    encoding_size = 128,
    elite_size = 32,
    max_iter = 500,
)
