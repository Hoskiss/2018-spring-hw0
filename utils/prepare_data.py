import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
names = ["class", "title", "content"]


def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]


def load_data(file_name, sample_ratio=1, n_class=15, names=names, one_hot=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name, names=names)
    shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffle_csv["content"])
    y = pd.Series(shuffle_csv["class"])
    if one_hot:
        y = to_one_hot(y, n_class)
    return x, y


def data_preprocessing(train, test, max_len):
    """transform to one-hot idx vector by VocabularyProcessor"""
    """VocabularyProcessor is deprecated, use v2 instead"""
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_len)
    x_transform_train = vocab_processor.fit_transform(train)
    x_transform_test = vocab_processor.transform(test)
    vocab = vocab_processor.vocabulary_
    vocab_size = len(vocab)
    x_train_list = list(x_transform_train)
    x_test_list = list(x_transform_test)
    x_train = np.array(x_train_list)
    x_test = np.array(x_test_list)

    return x_train, x_test, vocab, vocab_size


def data_preprocessing_v2(train, test, max_len, max_words=50000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, max_words + 2


def data_preprocessing_with_dict(train, test, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2


def split_dataset(x_test, y_test, dev_ratio):
    """split test dataset to test and dev set with ratio """
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size


def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first.
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch

        
def get_cutted_sentences(raw_lines):
    stopword_list = stopwords.words("english")
    sentences = []
    for line in raw_lines:
        line = line.strip()
        line = line.replace(" ' ", "'")
        line = re.sub("[^a-zA-Z']", " ", line)

        words = line.lower().split()
        words = [word for word in words if word not in stopword_list and len(word)>1]
        sentences.append(words)
        
    print(len(sentences))
    return sentences

def get_train_test_data(labeled_data_path=os.path.join(os.getcwd(), "data", "training_label.csv")):
    labeled_data = []

    with open(labeled_data_path, 'r') as labeled_file:
        for line in labeled_file.readlines():
            (label, text) = line.split("+++$+++")
            labeled_data.append([label.strip(), text.strip()])

    labeled_dataframe = pd.DataFrame(labeled_data, columns =['Label', 'Text']) 
    training_frame, validation_frame = train_test_split(labeled_dataframe, test_size=0.1, random_state=42)
    
    x_train = training_frame['Text'].tolist()
    x_train = get_cutted_sentences(x_train)
    x_train = pd.Series([" ".join(sentence) for sentence in x_train])
    y_train = training_frame['Label']
    
    x_test = validation_frame['Text'].tolist()
    x_test = get_cutted_sentences(x_test)
    x_test = pd.Series([" ".join(sentence) for sentence in x_test])
    y_test = validation_frame['Label']
    
    print(type(x_train))
    print(type(y_train))
    return (x_train, y_train, x_test, y_test)
    
