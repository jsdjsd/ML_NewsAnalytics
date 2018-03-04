#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 18:09:24 2018

@author: Jean-Sebastien
"""

import sys
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import nltk
import re

stopwords = nltk.corpus.stopwords.words('english')

# You would see five .txt files after unzip 'a_song_of_ice_and_fire.zip'
input_file_names = ["embedding/a_song_of_ice_and_fire/001ssb.txt", "embedding/a_song_of_ice_and_fire/002ssb.txt", "embedding/a_song_of_ice_and_fire/003ssb.txt", 
                    "embedding/a_song_of_ice_and_fire/004ssb.txt", "embedding/a_song_of_ice_and_fire/005ssb.txt"]

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

sentence_words = []
for file_name in input_file_names:
    for line in open(file_name,'r',encoding='UTF-8'):
        sentence_words.append(tokenize_only(line))        
GOT_SENTENCE_WORDS = [word for word in sentence_words if word not in stopwords]

from gensim.models import Word2Vec

# size: the dimensionality of the embedding vectors.
# window: the maximum distance between the current and predicted word within a sentence.
model = Word2Vec(GOT_SENTENCE_WORDS, size=300, window=3, min_count=20, workers=4)
#model = Word2Vec(GOT_SENTENCE_WORDS, size=100, window=20, min_count=100, workers=4)
model.wv.save_word2vec_format("got_word2vec.txt", binary=False)
model.most_similar('king', topn=10)

def read_glove_vecs(glove_file):
    with open(glove_file, 'r',encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map
        
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('got_word2vec.txt')

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
tsne_plot(model)

