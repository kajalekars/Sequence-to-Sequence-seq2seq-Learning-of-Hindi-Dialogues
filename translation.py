# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 08:42:29 2019

@author: ASUS
"""
#import tensorflow as tf 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import string
from string import digits
import matplotlib.pyplot as plt
#%matplotlib inline
import re
import collections
import helper

import seaborn as sns
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split





#import tensorflow as tf


    
    
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

print(os.listdir("../translation/"))

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)

PATH = "../translation/hindi_english.csv"

os.chdir(r'E:\NCI\translation')

lines=pd.read_csv(r'E:/NCI/translation/hindi2english.csv')
lines['source'].value_counts()

lines.drop_duplicates(inplace=True)

lines=lines[~pd.isnull(lines['english_sentence'])]

hindi_sentences = lines['hindi_sentence']
english_sentences = lines['english_sentence']
print('Dataset Loaded')


'''
#SENTIMENT ANALYSIS

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))
    


score_list=[] 
positive_score=[]
negative_score=[]
neutral_score=[]
compound_score=[]

for i in english_sentences:
    score=analyser.polarity_scores(i)
    print(score)
    positive_score.append(score['pos'])
    negative_score.append(score['neg'])
    neutral_score.append(score['neu'])
    compound_score.append(score['compound'])

score_tuples=list(zip(english_sentences,positive_score,neutral_score,negative_score,compound_score))

score_list=pd.DataFrame(score_tuples,columns=['Text','Positive','Neutral','Negative','Compound'])


'''


for i in range(2):
    print('English Line {}:  {}'.format(i + 1, english_sentences[i]))
    print('Hindi Line {}:  {}'.format(i + 1, hindi_sentences[i]))




lines['english_sentence'].astype(str)



# Lowercase all characters
english_sentences=english_sentences.apply(lambda x: x.lower())
hindi_sentences=hindi_sentences.apply(lambda x: x.lower())



#Remove quotes

english_sentences=english_sentences.apply(lambda y: re.sub("'", '', y))
hindi_sentences=hindi_sentences.apply(lambda y: re.sub("'", '', y))

english_sentences=english_sentences.apply(lambda z: re.sub("“”", '', z))
hindi_sentences=hindi_sentences.apply(lambda z: re.sub("“”", '', z))

special_characters= set(string.punctuation) # Set of all special characters
# Remove all the special characters
english_sentences=english_sentences.apply(lambda z: ''.join(ch for ch in z if ch not in special_characters))
hindi_sentences=hindi_sentences.apply(lambda z: ''.join(ch for ch in z if ch not in special_characters))
hindi_sentences=hindi_sentences.apply(lambda z: re.sub("[।]", "", z)) #removing the hindi sentence ender ; dont get confused with the blank space within re.sub

# Remove all numbers from text
remove_digits = str.maketrans('', '', digits)
english_sentences=english_sentences.apply(lambda m: m.translate(remove_digits))
hindi_sentences=hindi_sentences.apply(lambda n: n.translate(remove_digits))

hindi_sentences=hindi_sentences.apply(lambda o: re.sub("[०८९५७२१३४६]", "", o))

# Remove extra spaces
english_sentences=english_sentences.apply(lambda y: y.strip())
hindi_sentences=hindi_sentences.apply(lambda y: y.strip())
english_sentences=english_sentences.apply(lambda z: re.sub(" +", " ", z))
hindi_sentences=hindi_sentences.apply(lambda z: re.sub(" +", " ", z))



#Vocabulary Statistics
english_words_counter = collections.Counter([word for sentence in english_sentences for word in sentence.split()])
hindi_words_counter = collections.Counter([word for sentence in hindi_sentences for word in sentence.split()])

print('English words : {}'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('Unique English words : {}'.format(len(english_words_counter)))
print('Top 10 common English words :')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('Hindi words : {}'.format(len([word for sentence in hindi_sentences for word in sentence.split()])))
print('Unique Hindi words : {}'.format(len(hindi_words_counter)))
print('Top 10 common Hindi words :')
print('"' + '" "'.join(list(zip(*hindi_words_counter.most_common(10)))[0]) + '"')



#Sampling 20000 records randomly with a random state=50


sentences_tuples=list(zip(english_sentences,hindi_sentences))
sentences_list=pd.DataFrame(sentences_tuples,columns=['English Sentences','Hindi Sentences'])

data_samples=sentences_list.sample(n=20000,random_state=50)
data_samples.shape

#Tokenization

def tokenize(x):
    x_tk = Tokenizer(char_level = False)
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk
text_sentences = [
    'The quick brown fox jumps over the lazy dog .',
    'By Jove , my quick study of lexicography won a prize .',
    'This is a short sentence .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)
print()
for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))
    

#Padding
    
def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen = length, padding = 'post')

#tests.test_pad(pad)

# Pad Tokenized output
test_pad = pad(text_tokenized)
for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(np.array(token_sent)))
    print('  Output: {}'.format(pad_sent))
    
#pre process pipeline
    

def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


preproc_english_sentences, preproc_hindi_sentences, english_tokenizer, hindi_tokenizer =\
    preprocess(english_sentences, hindi_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_hindi_sequence_length = preproc_hindi_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
hindi_vocab_size = len(hindi_tokenizer.word_index)
print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max Hindi sentence length:", max_hindi_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("Hindi vocabulary size:", hindi_vocab_size)

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

print('`logits_to_text` function loaded.')