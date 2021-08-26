

#import nltk
#!pip install fasttext
#nltk.download('punkt')

# !git clone https://github.com/facebookresearch/fastText.git
# !cd fastText
# !pip install fastText

import re
import fasttext
import numpy as np
import pandas as pd
from fasttext import load_model
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
",".join(stopwords.words('english'))

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


"""### Training Fasttext"""

df1 = pd.read_csv("/home/mayank/Downloads/pmeans/train_new.txt",sep = "\t", names =["sentence","label"]).dropna()
df2 = pd.read_csv("/home/mayank/Downloads/pmeans/test_new.txt",sep = "\t", names =["sentence","label"]).dropna()
df3 = pd.read_csv("/home/mayank/Downloads/pmeans/IIITH_Codemixed_new.txt",sep = "\t", names =["sentence","label"]).dropna()
#df4 = pd.read_csv("/home/mayank/Downloads/pmeans/test_syllable.txt",sep = "\t", names =["sentence","label"]).dropna()
#df5 = pd.read_csv("/home/mayank/Downloads/pmeans/train_syllable.txt",sep = "\t", names =["sentence","label"]).dropna()
#df4 = pd.read_csv("/home/mayank/Downloads/pmeans/file15.csv").dropna()
#df4 = pd.read_csv('/home/mayank/Downloads/pmeans/twitter_mod.csv').dropna()


frames = [df1,df2,df3]
df_train = pd.concat(frames)
print("length of dataset on which model is being trained:",len(df_train))
print("Pre-processing the dataset")
df_train = preprocessing(df_train)
print("Preprocessing Done")


with open("/home/mayank/Downloads/pmeans/training.txt", "w") as train_file_handler:
    for X_train_entry, y_train_entry in zip(df_train['sentence'],df_train['label']):
        line_to_write = "__label__" + str(y_train_entry) + "\t" + str(X_train_entry) + "\n"
        try:
            train_file_handler.write(line_to_write)
        except:
            print(line_to_write)
            break
        

"""
Empty input or output path.

The following arguments are mandatory:
  -input              training file path
  -output             output file path

The following arguments are optional:
  -verbose            verbosity level [2]

The following arguments for the dictionary are optional:
  -minCount           minimal number of word occurrences [1]
  -minCountLabel      minimal number of label occurrences [0]
  -wordNgrams         max length of word ngram [1]
  -bucket             number of buckets [2000000]
  -minn               min length of char ngram [0]
  -maxn               max length of char ngram [0]
  -t                  sampling threshold [0.0001]
  -label              labels prefix [__label__]

The following arguments for training are optional:
  -lr                 learning rate [0.1]
  -lrUpdateRate       change the rate of updates for the learning rate [100]
  -dim                size of word vectors [100]
  -ws                 size of the context window [5]
  -epoch              number of epochs [5]
  -neg                number of negatives sampled [5]
  -loss               loss function {ns, hs, softmax} [softmax]
  -thread             number of threads [12]
  -pretrainedVectors  pretrained word vectors for supervised learning []
  -saveOutput         whether output params should be saved [0]

The following arguments for quantization are optional:
  -cutoff             number of words and ngrams to retain [0]
  -retrain            finetune embeddings if a cutoff is applied [0]
  -qnorm              quantizing the norm separately [0]
  -qout               quantizing the classifier [0]
  -dsub               size of each sub-vector [2]
"""


""" 
ft.trainUnsupervised(trainFile, modelname, args, trainCallback);
Hyper paramters of fastext
model: must be "cbow" or "skipgram"
lr                # learning rate [0.05]
dim               # size of word vectors [100]
ws                # size of the context window [5]
epoch             # number of epochs [5]
minCount          # minimal number of word occurences [5]
minn              # min length of char ngram [3]
maxn              # max length of char ngram [6]
neg               # number of negatives sampled [5]
wordNgrams        # max length of word ngram [1]
loss              # loss function {ns, hs, softmax, ova} [ns]
bucket            # number of buckets [2000000]
thread            # number of threads [number of cpus]
lrUpdateRate      # change the rate of updates for the learning rate [100]
t                 # sampling threshold [0.0001] """
print("Training Started")

#pretrainedVectors='/home/mayank/Downloads/pmeans/wiki.en.vec'

hyper_params = { 
    "lr": 0.001,         # Learning rate
    "epoch": 25,       # Number of training epochs to train for
    "wordNgrams": 4,    # Number of word n-grams to consider during training
    "dim": 300,         # Size of word vectors
    "ws": 5,            # Size of the context window for CBOW or skip-gram
    'lrUpdateRate': 150000,
    'bucket' : 100000,
    "loss" : 'ns',
    "maxn" : 5,
    "minn" : 2,
    "minCount":1
    
    
}

model = fasttext.train_supervised(input='/home/mayank/Downloads/pmeans/training.txt',**hyper_params,verbose = 10,pretrainedVectors='/home/mayank/Downloads/pmeans/wiki.en.vec')
print("/n/nTraining Ended")



print("/n/nSaving the model...........")
model.save_model('/home/mayank/Downloads/pmeans/current_model.bin')
print("/n/n","Model saved successfully")


"""Quering the model"""
print("/n/n20 Most common words:")
print(model.words[:20])



print(model.get_nearest_neighbors('modi'))





















































# """### Random Forest"""

# from sklearn.model_selection import GridSearchCV

# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [10,15],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3,4],
#     'min_samples_split': [3,4],
#     'n_estimators': [1150, 1200]
# }
# # Create a based model
# rf = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)

# # # Fit the grid search to the data
# grid_search.fit(feature_matrix_train,y_train)

# print(grid_search.best_params_)

#Fitting Random Forest Classification to the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(bootstrap = True,max_depth = 15,max_features = 3,min_samples_leaf = 3,min_samples_split =4,n_estimators = 1150)

# classifier.fit(feature_matrix_train,y_train)
# # Predicting the Test set results
# y_pred = classifier.predict(feature_matrix_test)

# import sklearn
# print(sklearn.metrics.classification_report(y_test, y_pred))





# def clean_dataset(df):
#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#     df.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return df[indices_to_keep].astype(np.str)

# df = clean_dataset(df)

# # import Word2Vec loading capabilities
# from gensim.models import KeyedVectors

# # Creating the model
# embed_lookup = KeyedVectors.load_word2vec_format('/content/drive/My Drive/GoogleNews-vectors-negative300.bin', 
#                                                  binary=True)

# store pretrained vocab
# pretrained_words = []
# for word in embed_lookup.vocab:
#     pretrained_words.append(word)

# row_idx = 1

# # get word/embedding in that row
# word = pretrained_words[row_idx] # get words by index
# embedding = embed_lookup[word] # embeddings by word

# # vocab and embedding info
# print("Size of Vocab: {}\n".format(len(pretrained_words)))
# print('Word in vocab: {}\n'.format(word))
# print('Length of embedding: {}\n'.format(len(embedding)))
# #print('Associated embedding: \n', embedding)

# # print a few common words
# for i in range(5):
#     print(pretrained_words[i])

#train, test = train_test_split(df, test_size=0.3, random_state=42)

# from nltk.corpus import stopwords
# def tokenize_text(text):
#     tokens = []
#     for sent in nltk.sent_tokenize(text):
#         for word in nltk.word_tokenize(sent):
#             if len(word) < 2:
#                 continue
#             tokens.append(word.lower())
#     return tokens

# train_tagged = train.apply(
#     lambda r: TaggedDocument(words=tokenize_text(r['Speech']), tags=[r.Labels]), axis=1)
# test_tagged = test.apply(
#     lambda r: TaggedDocument(words=tokenize_text(r['Speech']), tags=[r.Labels]), axis=1)

# train_tagged.values[1]

# model= Doc2Vec(dm=1, vector_size=200,min_count=5,window = 5,workers = 32,alpha = 0.1)
# model.build_vocab([x for x in tqdm(train_tagged.values)])

# for epoch in range(30):
#     model.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
#     model.alpha -= 0.002
#     model.min_alpha = model.alpha

# def vec_for_learning(model, tagged_docs):
#     sents = tagged_docs.values
#     targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
#     return targets, regressors
# def vec_for_learning(model, tagged_docs):
#     sents = tagged_docs.values
#     targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
#     return targets, regressors

# y_train, X_train = vec_for_learning(model, train_tagged)
# y_test, X_test = vec_for_learning(model, test_tagged)

# print(type(X_train))

# from sklearn.model_selection import cross_validate

# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier as RFC
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# import pickle

# def train_classifier(X,y):
#     param_grid = {'C': [0.1, 1, 10],  
#               'gamma': [1, 0.1, 0.01], 
#               'kernel': ['linear']}  
  
#     clf = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
#     clf.fit(X,y)
#     return clf 



#     # n_estimators = [200,400]
#     # min_samples_split = [2]
#     # min_samples_leaf = [1]
#     # bootstrap = [True]

#     # parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
#     #               'min_samples_split': min_samples_split}

#     # clf = GridSearchCV(svm.SVC(verbose=1,n_jobs=4), cv=4, param_grid=parameters)
#     # clf.fit(X, y)
#     # return clf


# classifier = train_classifier(X_train,y_train)
# print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
# print (classifier.score(X_test,y_test))

# SVM = svm.SVC(kernel='rbf',C=1, gamma=0.01)
# SVM.fit(X_train,y_train)

# pred_y = SVM.predict(X_test)

# import sklearn
# print(sklearn.metrics.classification_report(y_test, pred_y))

# print("SVM Accuracy Score -> ",accuracy_score(pred_y, y_test)*100)





# from sklearn.model_selection import cross_validate

# from sklearn.model_selection import GridSearchCV
# from sklearn.ensemble import RandomForestClassifier as RFC
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# import pickle

# def train_classifier(X,y):
#     param_grid = {'C': [0.1, 1, 10],  
#               'gamma': [1, 0.1, 0.01], 
#               'kernel': ['linear']}  
  
#     clf = GridSearchCV(svm.SVC(), param_grid, refit = True, verbose = 3)
#     clf.fit(X,y)
#     return clf
#     # n_estimators = [200,400]
#     # min_samples_split = [2]
#     # min_samples_leaf = [1]
#     # bootstrap = [True]

#     # parameters = {'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf,
#     #               'min_samples_split': min_samples_split}

#     # clf = GridSearchCV(RFC(verbose=1,n_jobs=4), cv=4, param_grid=parameters)
#     # clf.fit(X, y)
#     # return clf




# classifier = train_classifier(feature_matrix,y_train)
# print (classifier.best_score_, "----------------Best Accuracy score on Cross Validation Sets")
# print (classifier.score(feature_matrix_test,y_test))

















