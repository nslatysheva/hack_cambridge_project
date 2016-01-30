
# coding: utf-8

# # Python code to classify text into known text categories

# This code takes a .json file as input. The file should contain two columns - a "lyrics" column containing lyrics and a "song" column containing the song label.
# 
# Lexical similarity. 
# 
# Won't capture if similar words, not identical. 
# Lexical similarity
# 

# In[75]:

import os
import csv
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[76]:

#==============================================================================
# Load in text data, split into train/test 
#==============================================================================

# load in data
os.chdir("/Users/natashal/Projects/random_things/hack_cambridge/swift_lyrics/")
# lyrics = pd.read_json('swift.json') # works
lyrics = pd.read_json('dataset_swift3.json')

print lyrics


# In[77]:

# what's the shape (rows, columns) of the data?
print(lyrics.shape); type(lyrics)
lyrics.head


# In[78]:

# drop entries with null values, check shape afterwards
lyrics = lyrics.dropna(how='any')
lyrics.shape

# split my data into train and test sets
from sklearn.cross_validation import train_test_split
train, test = train_test_split(lyrics, test_size=0.2, random_state=42)
print train.shape; print test.shape


# In[79]:

#==============================================================================
# Process description fields of train set
#==============================================================================

# tokenize the text using countvectoriser
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(lowercase=True, stop_words='english', strip_accents='unicode')
print count_vect.get_stop_words()

# if wanting to use n-grams
count_vect = CountVectorizer(analyzer='word', ngram_range=(1,2), lowercase=True, stop_words='english', strip_accents='unicode')


# In[80]:

train


# In[81]:

# fit the count vectoriser
X_train_counts = count_vect.fit_transform(train.lyrics)
X_train_counts.shape
print count_vect.get_feature_names()[0:10]

# get term frequencies (tf), scale by inverse document frequenies (idf)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(use_idf=True, smooth_idf=True)

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print X_train_tfidf.shape

# explore the matrix by converting back to dense format
dense=X_train_tfidf.todense()
dense[0:10, 0:10]


# In[82]:

#==============================================================================
# Training classifiers 
#==============================================================================
###################### 1) Start with naive bayes
from sklearn.naive_bayes import MultinomialNB
classifier_NB = MultinomialNB()
# train NB classifier
classifier_NB_fit = classifier_NB.fit(X_train_tfidf, train.song)
                    
# predict ratings on test set using model                    
test_counts = count_vect.transform(test.lyrics)
test_tfidf = tfidf_transformer.transform(test_counts)
predicted_nb = classifier_NB_fit.predict(test_tfidf)
print predicted_nb
print test.song

## get accuracy i.e. how often the predicted value eqausl the target values
print("NB accuracy", np.mean(predicted_nb == test.lyrics))


# In[88]:

####################### 2) Train the linear SVM 
from sklearn.linear_model import SGDClassifier
classifier_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
classifier_svm_fit = classifier_svm.fit(X_train_tfidf, train.song)                               
                  
# predict ratings on test set using model
predicted_svm = classifier_svm_fit.predict(test_tfidf) 

# get accuracy again, to compare to NB
print("SVM accuracy", np.mean(predicted_svm == test.song))


# In[89]:

import pickle

pickle.dump(classifier_svm_fit, open("./../svm_classifier.p", "wb"))


# In[93]:

x = pickle.load(open("./../svm_classifier.p", "rb"))

predicted_svm2 = x.predict(test_tfidf) 
print predicted_svm2


# In[35]:

#==============================================================================
# Detailed performance metrics
#==============================================================================
# write out classification performance report
from sklearn import metrics
report = metrics.classification_report(test.song, predicted_svm)
print(report)

# write out confusion matrix
confusion = metrics.confusion_matrix(test.song, predicted_svm)
print(confusion)


# In[ ]:




# In[34]:

### NOT TESTED
# DONT RUN THIS YET


def plot_confusion_matrix(confusion_matrix, title="Confusion matrix"):
    plt.matshow(confusion_matrix) 
    plt.xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    plt.yticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
    plt.colorbar()
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

plot_confusion_matrix(confusion)

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
confusion_normalized = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]
print(confusion_normalized)
plot_confusion_matrix(confusion_normalized, title="Normalised confusion matrix")

