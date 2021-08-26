#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:17:17 2020

@author: prarabdh
"""

import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from keras.models import load_model

import warnings
warnings.filterwarnings("ignore")

# from gensim.models.doc2vec import Doc2Vec, TaggedDocument

reconstructed_model = load_model("my_model")

def preprocessing_data(df_train):
    import re
    #Data Cleaning
    #Removing all the punctuation marks and converting to lowercase
    def clean_str(string):
        """Tokenization/string cleaning for all datasets except for SST.

        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"[.,#!$%&;:{}=_`~()/\\]", "", string)
        string = re.sub(r"http\S+", " " ,string)
        string = re.sub("[^a-zA-Z]", " ",string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
        string = re.sub(r"\s{2,}", " ", string)
        re.sub(r'(.)\1+', r'\1\1', string) 

        return string.strip().lower()


    df_train['sentence'] = df_train['sentence'].astype(str).map(clean_str,na_action=None)
    df_train['label'] = df_train['label'].astype(str)

    
    def remove_shorter_words(text):
      return " ".join([word for word in str(text).split() if len(word) > 2])

    df_train["sentence"] = df_train["sentence"].apply(lambda text: remove_shorter_words(text))

    def remove_longer_words(text):
      """ remove words longer than 12 char """
      return " ".join([word for word in str(text).split() if len(word) < 12])

    df_train["sentence"] = df_train["sentence"].apply(lambda text: remove_longer_words(text))

    def remove_words_digits(text):
      """ remove words with digits """
      return " ".join([word for word in str(text).split() if not any(c.isdigit() for c in word) ])

    df_train["sentence"] = df_train["sentence"].apply(lambda text: remove_words_digits(text))


    # from nltk.corpus import stopwords
    # https://github.com/TrigonaMinima/HinglishNLP/blob/master/data/assets/stop_hinglish
    # save inside /nltk_data/corpora/stopwords
    STOPWORDS = set(stopwords.words('english')+stopwords.words('stop_hinglish'))
    def remove_stopwords(text):
        """custom function to remove the stopwords"""
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])

    df_train["sentence"] = df_train["sentence"].apply(lambda text: remove_stopwords(text))
    
    return df_train


def corpus_word_embedding_dict(df_train):

    # converts the word embeddings in dictionary form
    ft=open("word_embeddings.txt")
    wp=0
    embd_dict={}
    for line in ft.readlines():
        if wp==0:
            # print(line) # no. of unique word embeddings we have from finetuning   
            wp=1
            continue
        line1=line.split()
        word=line1[0]
        embd=np.array([line1[1:]])
        embd = np.asarray(embd, dtype='float64')
        embd_dict[word]=embd
    
    return embd_dict

# Converting Hinglish Profanity List in dictionary form
# https://github.com/pmathur5k10/Hinglish-Offensive-Text-Classification
prof_dict={}
hing_prof=pd.read_csv('Hinglish_Profanity_List.csv', names=["hing_word","eng_mean", "label"])
for index, row in hing_prof.iterrows(): 
    prof_dict[row["hing_word"]]= row["label"]



def corpus_sen_embedding_dict(embd_dict,sen_data, prof_dict=prof_dict):
    all_sen_embd=[]
    sen_index=0

    for sen in sen_data:
        new_embd=[]
        leng=0
        sent=sen.split()
        for word in sent:
            if word in embd_dict:
                if word in prof_dict:
                    new_embd.append(embd_dict[word]*prof_dict[word])
                    leng+=prof_dict[word]
                else:
                    new_embd.append(embd_dict[word])
                    leng+=1
            else:
                pass   #later byte pair encoding

        if leng != 0:
            sen_embd=np.array(new_embd)
            weighted_sen_embd=sen_embd.sum(0)/leng            
        else:
            alp=np.random.uniform(-0.004,0.004,300)
            weighted_sen_embd=np.array([alp])

        all_sen_embd.append(weighted_sen_embd)
        sen_index+=1

    all_sen_embd=np.array(all_sen_embd) 
    all_sen_embd=all_sen_embd.reshape(sen_index,-1)

    return all_sen_embd

df_1=pd.read_csv('train_new1.csv', index_col=0)
df_1=preprocessing_data(df_1)
corpus_embeddings=corpus_word_embedding_dict(df_1)


def check_custom_sen(custm_sent,words_embd_dic=corpus_embeddings,model_in=reconstructed_model):
    
    cust_sent=pd.DataFrame([[custm_sent,0]],columns=["sentence",'label'])
    cust_sent=preprocessing_data(cust_sent)
    cust_sent=cust_sent['sentence']
    
    custm_embd=corpus_sen_embedding_dict(words_embd_dic, sen_data=cust_sent)
    custom_sent_embd=custm_embd.reshape(custm_embd.shape[0], custm_embd.shape[1], 1)
    
    category_prob=model_in.predict(np.array(custom_sent_embd))
    assigned_label=np.argmax(category_prob)
    
    if assigned_label == 0:
        custom_prediction="0 Negative"
    elif assigned_label == 1:
        custom_prediction="1 Neutral"
    else:
        custom_prediction="2 Positive"
    
    return custom_prediction


# all small letters
# custom_sent="zara mushroom twacha gori chutiya banane dhanda"
# custom_sent= 'calling twitter help gov' 

#from test cases
# custom_sent = 'winning election important congress goes extent calling incumbent neech aadmi' 
# custom_sent = 'simply display neech' 
# custom_sent = 'mullan choot fat chooki bete hindu mard ranbir' 

# check cases
custom_sent="acha dost hai yaar tu" #neutral
# custom_sent="bhai aap best ho"
# custom_sent="thanku papa saving love alot plz paa"
# custom_sent="aap mujhe bahot pasand ho"
# custom_sent="mai aapse bahot pyar karta hun ho"  #positive
# custom_sent="chutiya hai kya"   #negative
# custom_sent="toh nahi hai tu"
# custom_sent="tune mere sath dhoka kiya"
# custom_sent="jiyo mere laal"

out_class=check_custom_sen(custom_sent,model_in=reconstructed_model)
print(custom_sent)
print(out_class)















