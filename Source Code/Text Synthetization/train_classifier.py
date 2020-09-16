import pandas as pd
import numpy as np

import tensorflow as tf

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import re

from sklearn.model_selection import train_test_split

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


df_train = pd.read_csv('IMDB_Reviews_train.csv')
df_train['sentiment'] = df_train['sentiment'].astype('category').cat.codes
df_train['review'] = df_train['review'].apply(clean_str)

df_test = pd.read_csv('IMDB_Reviews_test.csv')
df_test['sentiment'] = df_test['sentiment'].astype('category').cat.codes
df_test['review'] = df_test['review'].apply(clean_str)


X_train = df_train.review
X_test = df_test.review

y_train = df_train.sentiment.values
y_test = df_test.sentiment.values

tokenize = tf.keras.preprocessing.text.Tokenizer()
tokenize.fit_on_texts(X_train.values)

X_train = tokenize.texts_to_sequences(X_train)
X_test = tokenize.texts_to_sequences(X_test)

max_length = 128
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, max_length)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, max_length)
print(X_train.shape)
print(X_test.shape)

vocab_size = len(tokenize.word_index) + 1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 256))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, recurrent_dropout = 0.3,)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenize, f)

model.summary()

model.fit(
    X_train,
    y_train,
    batch_size = 128,
    epochs = 5,
    validation_data=(X_test, y_test),
)
from sklearn.pipeline import Pipeline

model.save('classifier.h5')
