

import re
import fasttext
import numpy as np
import pandas as pd
from sklearn import utils
from sklearn.model_selection import train_test_split
from fasttext import load_model
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
", ".join(stopwords.words('english'))
#what kind of power mean you use
import sys
sys.path.append('/home/mayank/Downloads/pmeans')
import p_mean_FT as pmeanFT
meanlist=['mean','p_mean_2','p_mean_3']
#meanlist=['mean','max','min']
import sklearn
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier as RFC
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score


df = pd.read_csv("/home/mayank/Downloads/pmeans/IIITH_Codemixed_new.txt",sep = "\t",names=["sentence","label"]).dropna()


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


def preprocessing(df):

    df['sentence'] = df['sentence'].astype(str).map(clean_str,na_action=None)
    df['label'] = df['label'].astype(str)
    
    
    """Removing most frequent words,rare words and stop words"""
    
    from collections import Counter
    cnt = Counter()
    for text in df["sentence"].values:
        for word in text.split():
            cnt[word] += 1
            
    cnt.most_common(10)
    
    FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
    
    def remove_freqwords(text):
        """custom function to remove the frequent words"""
        return " ".join([word for word in str(text).split() if word not in FREQWORDS])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_freqwords(text))
    
    n_rare_words = 10
    RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
    
    def remove_rarewords(text):
        """custom function to remove the rare words"""
        return " ".join([word for word in str(text).split() if word not in RAREWORDS])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_rarewords(text))
    
    def remove_shorter_words(text):
      return " ".join([word for word in str(text).split() if len(word) >= 2])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_shorter_words(text))
    
    def remove_longer_words(text):
      """ remove words longer than 12 char """
      return " ".join([word for word in str(text).split() if len(word) <= 12])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_longer_words(text))
    
    def remove_words_digits(text):
      """ remove words with digits """
      return " ".join([word for word in str(text).split() if not any(c.isdigit() for c in word) ])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_words_digits(text))
    
    STOPWORDS = set(stopwords.words('english'))
    def remove_stopwords(text):
        """custom function to remove the stopwords"""
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])
    
    df["sentence"] = df["sentence"].apply(lambda text: remove_stopwords(text))
    return df


def create_feature_matrix(data):
  # data will be X_train and X_test
  """ data input -> will be in the form on sentences "hello world"
      Step1 -> format like -> ["hello","world"]
      step2-> using pmean create a feature matrix """
  temp = []
  for items in data:
    temp.append(list(items.split()))

  feature_matrix = []
  for sentences in temp:
    feature_matrix.append(pmeanFT.get_sentence_embedding(sentences, model,meanlist))
  
  return feature_matrix





model = fasttext.load_model('/home/mayank/Downloads/pmeans/current_model.bin')
# print(model.words)
# print(model.labels)

print("Breaking the dataset into training and test")
df = preprocessing(df)
X = df["sentence"]
y = df["label"]

print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=42)
print("splitting done")


"""### Function to create feature matrix based on pmeans method"""

print("Creating Feature Matrix")



feature_matrix_train = create_feature_matrix(X_train)
feature_matrix_test = create_feature_matrix(X_test)


feature_matrix_train = np.array(feature_matrix_train)
feature_matrix_test = np.array(feature_matrix_test)




print("Feature Matrix is created")
from sklearn.ensemble import RandomForestClassifier





# def train_classifier(X,y):
    
#     rfc=RandomForestClassifier(random_state=42)
#     param_grid = { 
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
#     }
#     clf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5,refit=True)
#     clf.fit(X,y)
#     return clf
# classifier = train_classifier(feature_matrix_train,y_train)
# print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
# print (classifier.score(feature_matrix_test,y_test))
# print(classifier.best_params_)

# rfc =RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 500, max_depth=8, criterion='gini')
# rfc.fit(feature_matrix_train,y_train)
# pred_y = rfc.predict(feature_matrix_test)
# print(sklearn.metrics.classification_report(y_test, pred_y))

import xgboost
from xgboost import XGBClassifier


# def train_classifier(X,y):
#     """ To perform grid search"""
#     estimator = XGBClassifier(
#     objective= 'multi:softmax' ,
#     nthread=4,
#     seed=42
#     )
#     parameters = {
#         'max_depth': range (2, 10, 1),
#         'n_estimators': range(60, 220, 40),
#         'learning_rate': [0.1, 0.01, 0.05]
#     }
#     clf = GridSearchCV(
#         estimator=estimator,
#         param_grid=parameters,
#         scoring = 'accuracy',
#         n_jobs = -1,
#         cv = 10,
#         verbose=True
#     )
#     clf.fit(X, y)
#     return clf
# classifier = train_classifier(feature_matrix_train,y_train)
# print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
# print (classifier.score(feature_matrix_test,y_test))
# print(classifier.best_params_)

xgb = XGBClassifier(learning_rate = 0.1,max_depth=8,n_estimators= 140,objective="multi:softmax")
# xgb = XGBClassifier(learning_rate = 0.1,max_depth=5,n_estimators= 60,objective="multi:softmax")
xgb.fit(feature_matrix_train,y_train)
pred_y = xgb.predict(feature_matrix_test)
print(sklearn.metrics.classification_report(y_test, pred_y))



# rfc =RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 500, max_depth=5, criterion='entropy')
# rfc.fit(feature_matrix_train,y_train)
# pred_y = rfc.predict(feature_matrix_test)
# print(sklearn.metrics.classification_report(y_test, pred_y))

# def train_classifier(X,y):
#     """ To perform grid search"""
#     param_grid = {'C': [0.1, 1, 10,100,1000],  
#               'gamma': [1, 0.1, 0.01], 
#               'kernel': ['rbf']}  
  
#     clf = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
#     clf.fit(X,y)
#     return clf

# classifier = train_classifier(feature_matrix_train,y_train)
# print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
# print (classifier.score(feature_matrix_test,y_test))
# print(classifier.best_params_)

#{'C': 1, 'gamma': 0.01, 'kernel': 'rbf'}

# SVM = svm.SVC(kernel='rbf',C=1,gamma = 0.01)
# SVM.fit(feature_matrix_train,y_train)
# import pickle
# print("Saving Model........")
# pkl_filename = "pmeans_model.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(SVM, file)
# print("Model Saved.")
# pred_y = SVM.predict(feature_matrix_test)
# print(sklearn.metrics.classification_report(y_test, pred_y))

###################custom sentence#############################


# def predict_the_sentiment(inp):
    
#     #inp = np.array(inp)
#     inp_emb = create_feature_matrix([inp])
#     temp = SVM.predict(inp_emb)
#     return int(temp)

        
# print(predict_the_sentiment("abe chuiya hai kya"))
# from sklearn.metrics import recall_score
# print(recall_score(y_test, pred_y, average='weighted'))
#-------------------------------NN-----------------------------------------






















