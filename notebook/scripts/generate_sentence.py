# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### preprocess original data

# +
import re
import os

data_path = "../datas/wagahaiwa_nekodearu.txt"

# open files as binary data
bin_data = open(data_path, "rb")

lines = bin_data.readlines()
text = ""
for line in lines:
    tmp_text = line.decode("Shift_JIS")
    tmp_text = re.split(r'\r', tmp_text)[0]
    tmp_text = re.split(r'底本', tmp_text)[0]
    tmp_text = tmp_text.replace('|', '')
    tmp_text = re.sub(r'《.+?》','', tmp_text)
    tmp_text = re.sub(r'［＃.+?］','', tmp_text)
    text += tmp_text

os.makedirs('../processed_data/', exist_ok=True)
file = open('../processed_data/wagahai.txt', 'w', encoding='utf-8').write(text)
# -

# ### prepare data for LSTM

# +
import numpy as np

processed_data_path = '../processed_data/wagahai.txt'
bin_data = open(processed_data_path, "rb").read()
text = bin_data.decode("utf-8")
chars = sorted(list(set(text)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
stride = 3

sentences = [] # training data
next_chars = [] # answer data
# for i in range(0, len(text)-maxlen, stride):
for i in range(0, 10000, stride):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
Y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for j, char_ in enumerate(sentence):
        X[i, j, char_indices[char_]] = 1
    Y[i, char_indices[next_chars[i]]] = 1
# -

# ### build network

# +
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop

def simple_LSTM(input_shape, chars_num):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape))
    model.add(Dense(chars_num))
    model.add(Activation("softmax"))
    optimizer = RMSprop()
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


# -

# ### train network and generate sentence

import random
model = simple_LSTM(input_shape=X.shape[1:], chars_num=len(chars))
model.fit(X, Y, batch_size=128, verbose=1, epochs=10)
random_index = random.randint(0, len(sentences)-1)
sentence = sentences[random_index]
print("original sentence: ", sentence)
generated_sentence = sentence
for i in range(40):
    x = np.zeros((1,maxlen,len(chars)))
    for j, char_ in enumerate(sentence):
        x[0, j, char_indices[char_]] = 1
    preds = model.predict(x)[0]

    next_index = np.argmax(preds)
    next_char = indices_char[next_index]
    generated_sentence += next_char

    sentence = sentence[1:] + next_char
print("generated sentence: ", generated_sentence)

# %debug


