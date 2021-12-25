#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import threading
from threading import Thread
import pandas as pd
import re
import nltk
import spacy
import string
import queue
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from sklearn.decomposition import TruncatedSVD
from io import StringIO
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import time
from collections import Counter
from subprocess import call


# # READ DATA

# In[2]:


df112 = pd.read_csv('1 Billion Citation Dataset, v1 (112).csv') 
df109 = pd.read_csv('1 Billion Citation Dataset, v1 (109).csv') 


# # SPLIT DATA 

# In[3]:


# Split data 112
d1f12 = df112[:200000]
d2f12 = df112[200000:400000]
d3f12 = df112[400000:600000]
d4f12 = df112[600000:800000]
d5f12 = df112[800000:1000000]
d6f12 = df112[1000000:1200000]
d7f12 = df112[1200000:1400000]
# d8f12 = df112[1400000:1600000]
# d9f12 = df112[1600000:1800000]
# d0f12 = df112[1800000:2000000]


# In[4]:


# Split data 109
d1f09 = df109[:200000]
d2f09 = df109[200000:400000]
d3f09 = df109[400000:600000]
d4f09 = df109[600000:800000]
d5f09 = df109[800000:1000000]
d6f09 = df109[1000000:1200000]
d7f09 = df109[1200000:1400000]
# d8f09 = df109[1400000:1600000]
# d9f09 = df109[1600000:1800000]
# d0f09 = df109[1800000:2000000]


# # PREPROCESS

# In[5]:


def remove_urls(text):
  url_pattern = re.compile(r"https?://\S+|www\.\S+")
  return url_pattern.sub(r"", text)

def remove_html(text):
  html_pattern = re.compile("<.*?>")
  return html_pattern.sub(r"", text)

PUNCT = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","" , PUNCT))

PUNCT = string.punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","" , PUNCT))

nltk.download('stopwords')
",".join(stopwords.words("english"))
STOPW = set(stopwords.words("english"),)
def remove_stopwords(text):
  return" ".join([word for word in str(text).split() if word not in STOPW])        


# In[6]:


def preprocess(df):
    # Convert_lowerCase(df, "citationStringAnnotated")
    # df["Preprocess"] = df["citationStringAnnotated"].apply(lambda text: stem_words(text))
    # df["Preprocess"] = df["Preprocess"].apply(lambda text: lemmatize_words(text))
    df_preprocess = df.loc[:,'citationStringAnnotated']
    df_preprocess = df_preprocess.astype(str)
    df_preprocess = df_preprocess.str.lower()
    df_preprocess = df_preprocess.apply(lambda text: remove_urls(text))
    df_preprocess = df_preprocess.apply(lambda text: remove_html(text))
    df_preprocess= df_preprocess.apply(lambda text: remove_punctuation(text))
    df_preprocess = df_preprocess.apply(lambda text: remove_stopwords(text))
    df_preprocess = df_preprocess.str.replace('\d+', '',regex=True)

    return df_preprocess


# In[7]:


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


# # MULTITHREADING PREPROCESS FOR CSV112

# In[8]:


thread1 = ThreadWithReturnValue(target = preprocess, args=(d1f12,))
thread2 = ThreadWithReturnValue(target = preprocess, args=(d2f12,))
thread3 = ThreadWithReturnValue(target = preprocess, args=(d3f12,))
thread4 = ThreadWithReturnValue(target = preprocess, args=(d4f12,))
thread5 = ThreadWithReturnValue(target = preprocess, args=(d5f12,))
thread6 = ThreadWithReturnValue(target = preprocess, args=(d6f12,))
thread7 = ThreadWithReturnValue(target = preprocess, args=(d7f12,))
# thread8 = ThreadWithReturnValue(target = preprocess, args=(d8f12,))
# thread9 = ThreadWithReturnValue(target = preprocess, args=(d9f12,))
# thread0 = ThreadWithReturnValue(target = preprocess, args=(d0f12,))

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()
# thread8.start()
# thread9.start()
# thread0.start()

d1f12 = thread1.join()
d2f12 = thread2.join()
d3f12 = thread3.join()
d4f12 = thread4.join()
d5f12 = thread5.join()
d6f12 = thread6.join()
d7f12 = thread7.join()
# d8f12 = thread8.join()
# d9f12 = thread9.join()
# d0f12 = thread0.join()


# # MULTITHREADING PREPROCESS FOR CSV109

# In[9]:


thread1 = ThreadWithReturnValue(target = preprocess, args=(d1f09,))
thread2 = ThreadWithReturnValue(target = preprocess, args=(d2f09,))
thread3 = ThreadWithReturnValue(target = preprocess, args=(d3f09,))
thread4 = ThreadWithReturnValue(target = preprocess, args=(d4f09,))
thread5 = ThreadWithReturnValue(target = preprocess, args=(d5f09,))
thread6 = ThreadWithReturnValue(target = preprocess, args=(d6f09,))
thread7 = ThreadWithReturnValue(target = preprocess, args=(d7f09,))
# thread8 = ThreadWithReturnValue(target = preprocess, args=(d8f09,))
# thread9 = ThreadWithReturnValue(target = preprocess, args=(d9f09,))
# thread0 = ThreadWithReturnValue(target = preprocess, args=(d0f09,))

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()
# thread8.start()
# thread9.start()
# thread0.start()

d1f09 = thread1.join()
d2f09 = thread2.join()
d3f09 = thread3.join()
d4f09 = thread4.join()
d5f09 = thread5.join()
d6f09 = thread6.join()
d7f09 = thread7.join()
# d8f09 = thread8.join()
# d9f09 = thread9.join()
# d0f09 = thread0.join()


# # COMBINE DATA

# In[10]:


# 112
frames_112 = [d1f12, d2f12, d3f12, d4f12, d5f12, d6f12, d7f12]
  
result_112 = pd.concat(frames_112)
display(result_112)


# In[11]:


# 109
frames_109 = [d1f09, d2f09, d3f09, d4f09, d5f09, d6f09, d7f09]
  
result_109 = pd.concat(frames_109)
display(result_109)


# # BUILD LSA

# In[12]:


def build_lsa(dataset, dim, queue ):
        tfidf_vec = TfidfVectorizer(use_idf=True, norm='l2')
        svd = TruncatedSVD(n_components=dim)
    
        transformed_x_train = tfidf_vec.fit_transform(dataset)
    
        print('TF-IDF output shape:', transformed_x_train.shape)
    
        data_svd = svd.fit_transform(transformed_x_train)
    
        print('LSA output shape:', data_svd.shape)
    
        explained_variance = svd.explained_variance_ratio_.sum()
        print("Sum of explained variance ratio: %d%%" % (int(explained_variance * 100)))

        queue.put((tfidf_vec, transformed_x_train,svd, data_svd))
         


# In[13]:


queue1 = queue.Queue()
queue2 = queue.Queue()
thread_lsa1 = threading.Thread(target = build_lsa, args = (result_112,100, queue1,)) 
thread_lsa2 = threading.Thread(target = build_lsa, args = (result_109,100, queue2, )) 
thread_lsa1.start()
thread_lsa2.start()
thread_lsa1.join()
thread_lsa2.join()


# In[14]:


tfidf_vec_112, transformed_x_train_112, svd_112, data_svd_112 = queue1.get()
tfidf_vec_109, transformed_x_train_109, svd_109, data_svd_109 = queue2.get()


# # TOPIC MODELING

# In[17]:


#112
topic_weight_112 = data_svd_112[0]
for i,topic in enumerate(topic_weight_112):
  print("Topic ",i," : ",topic*100)


# In[18]:


#109
topic_weight_109 = data_svd_109[0]
for i,topic in enumerate(topic_weight_109):
  print("Topic ",i," : ",topic*100)


# In[22]:


topic_weight = np.concatenate(topic_weight_112,topic_weight_109)     


# In[23]:


vocab_112 = tfidf_vec_112.get_feature_names()
df_topicweight_112 = []
topicname_112 = []
for i, comp in enumerate(svd_112.components_):
    vocab_comp = zip(vocab_112, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    df_topicweight_112.insert(i,np.array(sorted_words)[:,1])
    topicname_112.append(" ".join(np.array(sorted_words)[:,0]))
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[24]:


vocab_109 = tfidf_vec_109.get_feature_names()
df_topicweight_109 = []
topicname_109 = []
for i, comp in enumerate(svd_109.components_):
    vocab_comp = zip(vocab_109, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    df_topicweight_109.insert(i,np.array(sorted_words)[:,1])
    topicname_109.append(" ".join(np.array(sorted_words)[:,0]))
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[27]:


df_topicweight = pd.DataFrame(df_topicweight_112 + df_topicweight_109)
df_topicweight


# In[28]:


topicweight = df_topicweight_112 + df_topicweight_109


# In[30]:


list_app_name = list(topicname_109 + topicname_112)

dic_recommended_1 = {}

for index in range(df_topicweight.shape[0]):

    similarities_1 = linear_kernel(topicweight[index].reshape(1, -1) ,topicweight).flatten()
    related_docs_indices_1 = (-similarities_1).argsort()[:10]

    dic_recommended_1.update({list_app_name[index]:[list_app_name[i] for i in related_docs_indices_1]})

df_content_based_results_1 = pd.DataFrame(dic_recommended_1)
df_content_based_results_1.reset_index(inplace=True)
df_content_based_results_1 = df_content_based_results_1.T
df_content_based_results_1

