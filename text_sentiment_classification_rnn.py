#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import nltk
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
# tokenizer = TweetTokenizer()


# In[2]:


VECTOR_SIZE = 300

stopword_list = stopwords.words("english")
print(stopword_list)


# In[3]:


def get_cutted_sentences(raw_lines):
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
    


# In[4]:


no_labeled_path = os.path.join(os.getcwd(), "data", "training_nolabel.csv")
total_sentences = None

with open(no_labeled_path, 'r') as no_labeled_file:
    total_sentences = get_cutted_sentences(no_labeled_file.readlines())


# In[5]:


# # build word2vec
# # sg=0 CBOW ; sg=1 skip-gram
# model = word2vec.Word2Vec(size=VECTOR_SIZE, min_count=30, window=7, sg=1)
# model.build_vocab(sentences)


# In[6]:


# # train word2vec model ; shuffle data every epoch
# WORD2VEC_TRAINING_TIMES = 20
# for _ in range(WORD2VEC_TRAINING_TIMES):
#     random.shuffle(sentences)
#     model.train(sentences, total_examples=len(sentences), epochs=1)
    


# In[7]:


# saved_folder = os.path.join(os.getcwd(), "saved_model")
# if not os.path.exists(saved_folder):
#     os.makedirs(saved_folder)

# model.save(os.path.join(os.getcwd(), "saved_model", 'dimension_300_window_7_skip_gram'))


# In[8]:


word2vec_model_path = os.path.join(os.getcwd(), "saved_model", 'dimension_300_window_7_skip_gram')
word2vec_model = word2vec.Word2Vec.load(word2vec_model_path)

print(word2vec_model['bye'])
print(word2vec_model.most_similar('fever'))


# In[ ]:





# In[9]:


labeled_path = os.path.join(os.getcwd(), "data", "training_label.csv")
labeled_data = []

with open(labeled_path, 'r') as labeled_file:
    for line in labeled_file.readlines():
        (label, text) = line.split("+++$+++")
        labeled_data.append([label.strip(), text.strip()])

labeled_dataframe = pd.DataFrame(labeled_data, columns =['Label', 'Text']) 
labeled_dataframe.head()


# In[10]:


testing_path = os.path.join(os.getcwd(), "data", "testing_data.csv")
testing_data = []

with open(testing_path, 'r') as testing_file:
    for line in testing_file.readlines()[1:]:
        line_split = line.split(",")
        testing_id = line_split[0]
        text = ",".join(line_split[1:])
        testing_data.append([testing_id.strip(), text.strip()])

testing_dataframe = pd.DataFrame(testing_data, columns =['Id', 'Text']) 
testing_dataframe.head()


# In[11]:


training_frame, validation_frame = train_test_split(labeled_dataframe, test_size=0.1, random_state=42)
print(training_frame['Label'].value_counts())
print(validation_frame['Label'].value_counts())


# In[12]:


training_x = training_frame['Text'].tolist()
training_x = get_cutted_sentences(training_x)
print(len(training_x))
print(training_x[:20])

validation_x = validation_frame['Text'].tolist()
validation_x = get_cutted_sentences(validation_x)

testing_x = testing_dataframe['Text'].tolist()
testing_x = get_cutted_sentences(testing_x)


# In[13]:


max_length = max(len(x) for x in training_x)
print(max_length)

max_length = max(len(x) for x in validation_x)
print(max_length)

max_length = max(len(x) for x in testing_x)
print(max_length)


# In[14]:


training_y = training_frame['Label'].as_matrix()
print(training_y.shape)
print(training_y[:20])

validation_y = validation_frame['Label'].as_matrix()


# In[15]:


# word2vec_model.wv.vocab.keys(), 所有可以轉成vector的字, length: 14386


# In[16]:


# vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
# vectorizer = TfidfVectorizer(analyzer='word', min_df=10)
# matrix = vectorizer.fit_transform([word for sentence in total_sentences for word in sentence])
# tfidf_map = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
# print ('vocab size :', len(tfidf_map))

#Save the tfidf 
tfidf_map_name = "tfidf_map.pickle"
# with open(tfidf_map_name, "wb") as pickle_file:
#     pickle.dump(tfidf_map, pickle_file)
with open(tfidf_map_name, "rb") as pickle_file:
    tfidf_map = pickle.load(pickle_file)


# In[17]:


def get_document_vector(tokens, vector_size=300, token_size=30):
    vector = np.zeros((token_size, vector_size))
    index = 0
    for word in tokens:
        try:
            # vector[index] = (word2vec_model[word].reshape((1, vector_size))) * tfidf_map[word] # combining w2v vectors with tfidf value of words in the tweet.
            vector[index] = (word2vec_model[word].reshape((1, vector_size))) 
            index += 1
        except KeyError: # handling the case where the token is not
#             print(word)
#             print(word2vec_model[word])
#             print(tfidf_map[word])
            
            continue
    return vector


# In[18]:


training_vector_x = np.array([get_document_vector(documnet) for documnet in training_x])
print(training_vector_x.shape)
print(training_vector_x[0])

validation_vector_x = np.array([get_document_vector(documnet) for documnet in validation_x])
print(validation_vector_x.shape)

testing_vector_x = np.array([get_document_vector(documnet) for documnet in testing_x])
print(testing_vector_x.shape)


# In[19]:


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LSTM
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint


# In[30]:


model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(training_vector_x.shape[1], training_vector_x.shape[2]),
               dropout=0.1, recurrent_dropout=0.1))
model.add(Dropout(0.2))
model.add(LSTM(48, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(48, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(48, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(16, return_sequences=False))
model.add(Dropout(0.2))

# Fully connected layer
model.add(Dense(64, activation='relu'))
# Dropout for regularization
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

model_name = "multiple_lstm_without_tfidf_classifier.h5"
model_path = os.path.join(os.getcwd(), model_name)
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_acc', patience=20, verbose=1)

model_history = model.fit(training_vector_x, training_y, validation_data=(validation_vector_x, validation_y), 
                          epochs=200, batch_size=50,
                          callbacks = [checkpoint, earlystop])


# In[ ]:


training_acc = model_history.history['acc']
val_acc = model_history.history['val_acc']

plt.plot(training_acc, label="training_accuracy")
plt.plot(val_acc, label="validation_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Binary Accuracy")
plt.title("Accuracy Curve")
plt.legend(loc='best')
plt.show()


# In[ ]:





# In[ ]:


# from keras.models import load_model

# model_path = os.path.join(os.getcwd(), model_name)
# model  = load_model(model_path)


# In[ ]:


result = model.predict(testing_vector_x)
print(result.shape)
print(result)
result = (result>0.5).astype(int)
result = result.reshape(-1)
print(result.shape)
print(result)


# In[ ]:


result_frame = pd.DataFrame({
    'id': testing_dataframe['Id'].tolist(),
    'label': result })
print(result_frame.head())
output_path = os.path.join(os.getcwd(), "result", "multiple_lstm_without_tfidf.csv")
result_frame.to_csv(output_path)


# In[ ]:




