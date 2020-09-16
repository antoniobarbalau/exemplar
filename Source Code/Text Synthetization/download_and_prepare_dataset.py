import pandas as pd
import numpy as np
import os
import pyprind
import nltk

import requests, zipfile, io
print("Downloading data ... ")
os.system('wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
print("Extracting ... ")
os.system('tar zxvf aclImdb_v1.tar.gz')
os.system('rm -fr aclImdb_v1.tar.gz')

print("Processing files ... ")

basepath = 'aclImdb'

labels = {'pos': 1, 'neg': 0}

df_train = pd.DataFrame()
df_test = pd.DataFrame()

pbar = pyprind.ProgBar(25000)
for l in ('pos', 'neg'):
    path = os.path.join(basepath, 'train', l)
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            txt = f.read()
        df_train = df_train.append([[txt, labels[l]]], ignore_index=True)
        pbar.update()

pbar = pyprind.ProgBar(25000)
for l in ('pos', 'neg'):
    path = os.path.join(basepath, 'test', l)
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            txt = f.read()
        df_test = df_test.append([[txt, labels[l]]], ignore_index=True)
        pbar.update()

df_train.columns = ['review', 'sentiment']
df_test.columns = ['review', 'sentiment']
np.random.seed(0)

df_train = df_train.reindex(np.random.permutation(df_train.index))
df_test = df_test.reindex(np.random.permutation(df_test.index))

print("Datasets saved to 'IMDB_Reviews_train.csv' / 'IMDB_Reviews_test.csv'")
df_train.to_csv('IMDB_Reviews_train.csv', index = False, encoding = 'utf-8')
df_test.to_csv('IMDB_Reviews_test.csv', index = False, encoding = 'utf-8')

os.system('rm -fr aclImdb')

print("Processing sentences for Generator data.")
os.makedirs('data/imdb/', exist_ok = True)

train_sentences = []
for i, row in df_train.iterrows():
    review = row['review']
    review = review.replace('<br>', '\n')
    review = review.replace('<br />', '\n')

    sentences = []
    for s in review.split('\n'):
        for s2 in nltk.tokenize.sent_tokenize(s):
            s2 = s2.lower().strip()

            num_words = len(s2.split(' '))
            if num_words >= 4 and num_words <= 32:
                sentences.append(s2 + '\n')

    train_sentences += sentences

with open('data/imdb/train.txt', 'wt') as f:
    f.writelines(train_sentences)

test_sentences = []
for i, row in df_test.iterrows():
    review = row['review']
    review = review.replace('<br>', '\n')
    review = review.replace('<br />', '\n')

    sentences = []
    for s in review.split('\n'):
        for s2 in nltk.tokenize.sent_tokenize(s):
            s2 = s2.lower().strip()

            num_words = len(s2.split(' '))
            if num_words >= 4 and num_words <= 32:
                sentences.append(s2 + '\n')

    test_sentences += sentences

with open('data/imdb/test.txt', 'wt') as f:
    f.writelines(test_sentences)

print("Data saved to 'data/imdb/train.txt' / 'data/imdb/test.txt'")
