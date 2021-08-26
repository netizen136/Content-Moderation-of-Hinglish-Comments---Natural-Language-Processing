#!/usr/bin/env python
# coding: utf-8

import time 
start_time=time.time()
import numpy as np
import pandas as pd


from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import re

import warnings
warnings.filterwarnings("ignore")



def preprocessing_data(df_train):
    
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

    np.random.seed(7)
    embedding_dim = 300
    learning_rate = 0.0003
    num_epochs = 100
    min_word_count = 1
    context = 4  # Context window size

    # args_embeddings=open("enwiki_20180420_100d.pkl")
    # args_embeddings=open("enwiki_20180420_300d.txt")
    args_embeddings=open("wiki-news-300d-1M.vec")
    args_output_path="filenamefast.txt"


    # Load data
    print("Loading data...")
    tune_data=df_train.reset_index()[["sentence", "label"]].values.tolist()

    print("loading pretrained word vectors")

    pretrainedVectors = args_embeddings
    print("Fine-tuning word vectors")

    docs = [TaggedDocument(str(tweet[0]).strip('"\'').lower().split(' '), [tweet[1]]) for tweet in tune_data]
    doc2VecModel = Doc2Vec(docs, size=embedding_dim, window=context, min_count=min_word_count, sample=1e-5, workers=1,
                             hs=0, dm=0, negative=5, dbow_words=1, dm_concat=1, pretrained_emb=pretrainedVectors, iter=num_epochs)
    print("saving fine-tuned embeddings")
    doc2VecModel.save_word2vec_format(args_output_path)



    # converts the word embeddings in dictionary form

    ft=open("filenamefast.txt")
    wp=0
    embd_dict={}
    for line in ft.readlines():
        if wp==0:
            print(line) # no. of unique word embeddings we have from finetuning   
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

print(len(prof_dict))



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


# The dataset should be in specific format such that
# column names as "sentence" and "label" and there sould be no index column

# NITS SEM EVAL DATASET
# df_1=pd.read_csv('train_new1.csv', index_col=0)
# df_2=pd.read_csv("test_new.txt",sep=("\t"),names=['sentence','label'])

# IIITH CODEMIXED DATASET
df_3=pd.read_csv('/home/prarabdh/Downloads/IIITH_Codemixed_new.txt', sep='\t', index_col=0,names=['sentence','label'])
# split the data into train and test set
from sklearn.model_selection import train_test_split
df_1,df_2 = train_test_split(df_3, test_size=0.2, random_state=42, shuffle=True)
df_1=df_1.reset_index(drop=True)
df_2=df_2.reset_index(drop=True)


# # HEOT DATASET
# df_3 = pd.read_csv("/home/prarabdh/jupyterfilezz/latest_dataset.csv", names=['sentence','label'])
# df_3=df_3.iloc[1:]
# # split the data into train and test set
# from sklearn.model_selection import train_test_split
# df_1,df_2 = train_test_split(df_3, test_size=0.2, random_state=42, shuffle=True)
# df_1=df_1.reset_index(drop=True)
# df_2=df_2.reset_index(drop=True)

df_1=preprocessing_data(df_1)
df_2=preprocessing_data(df_2)
X_train=df_1['sentence']
X_test=df_2['sentence']
y_train=df_1['label']
y_test=df_2['label']



# df_1=preprocessing_data(df_1)
corpus_embeddings=corpus_word_embedding_dict(df_1)




print(len(corpus_embeddings))



# corpus_embeddings['modi'].shape



train_embeddings=corpus_sen_embedding_dict(embd_dict=corpus_embeddings, sen_data=X_train)
test_embeddings=corpus_sen_embedding_dict(embd_dict=corpus_embeddings, sen_data=X_test)



feature_matrix_train=train_embeddings.reshape(train_embeddings.shape[0], train_embeddings.shape[1], 1)
print(feature_matrix_train.shape)



feature_matrix_test=test_embeddings.reshape(test_embeddings.shape[0], test_embeddings.shape[1], 1)
print(feature_matrix_test.shape)



from keras.utils import to_categorical
y_train = to_categorical(y_train,num_classes=3)
y_test = to_categorical(y_test,num_classes=3)
y_train.shape



n_features=300
n_outputs=3



from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras.optimizers import SGD

model = Sequential()
model.add(Conv1D(filters=128,kernel_size=4, activation='relu', input_shape=(n_features,1)))
model.add(Conv1D(64, 4, activation='relu'))
model.add(Dropout(0.4))
model.add(Conv1D(64, 4, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
# model.add(Dropout(0.2))
model.add(Conv1D(32, 4, activation='relu'))
# model.add(Conv1D(32, 4, activation='relu'))
model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(GlobalAveragePooling1D())
model.add(Dense(n_outputs, activation='softmax'))
model.summary()



# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


# history=model.fit(feature_matrix_train, y_train, shuffle=True,
# batch_size=32, epochs=100,verbose=True)
history=model.fit(feature_matrix_train, y_train, validation_split=0.2,shuffle=True,
batch_size=64, epochs=200,verbose=True)

model.save("my_modelfast")
reconstructed_model = load_model("my_modelfast")

test_acc = reconstructed_model.evaluate(feature_matrix_test, y_test)



training_acc = reconstructed_model.evaluate(feature_matrix_train, y_train)



from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.show()



def check_custom_sen(custm_sent,words_embd_dic=corpus_embeddings,model_in=model):
    
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
# custom_sent="acha dost hai yaar tu"
# custom_sent="bhai aap best ho"
# custom_sent="thanku papa saving love alot plz paa"
# custom_sent="aap mujhe bahot pasand ho"
# custom_sent="mai aapse bahot pyar karta hun ho"
# custom_sent="chutiya hai kya"
# custom_sent="toh nahi hai tu"
# custom_sent="tune mere sath dhoka kiya"
custom_sent="jiyo mere laal"



out_class=check_custom_sen(custom_sent,model_in=model)
print(custom_sent)
print(out_class)



# NonOffensive = 0,1121
# Abusive = 1,1765
# Hateinducing = 2,303



# df_1[70:100]


# df_1['sentence'][86]

print("Time taken: {} seconds".format(time.time()-start_time))




from sklearn.metrics import classification_report
#Precision Recall and F1-Score, metrics for evaluation

y_pred_LSTM1 = reconstructed_model.predict(feature_matrix_test, batch_size=64, verbose=1)
y_pred_LSTM = np.argmax(y_pred_LSTM1, axis=1)

y_test_LSTM=np.argmax(y_test, axis=1)
#rounded_labels[1]

xyz=classification_report(y_test_LSTM, y_pred_LSTM)
print(xyz)


from keras.utils import plot_model
dot_img_file = 'model_1.png'
plot_model(model, to_file=dot_img_file, show_shapes=False)













