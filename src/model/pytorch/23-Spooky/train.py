import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import keras
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize
from nltk.corpus import stopwords

import warnings

from lib.logger import Logger
from lib.training import BaseTrain
from lib.model.basic import Net1, Net2, Net3, Net4


warnings.filterwarnings('ignore')


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


stop_words = stopwords.words('english')


def Tfidf(train, val):
    # Always start with these features. They work (almost) everytime!
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

    # Fitting TF-IDF to both training and test sets (semi-supervised learning)
    tfv.fit(list(train) + list(val))
    xtrain_tfv = tfv.transform(train)
    xvalid_tfv = tfv.transform(val)
    return xtrain_tfv, xvalid_tfv, tfv


def CountV(xtrain, xvalid):
    ctv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}',
                          ngram_range=(1, 3), stop_words='english')

    # Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
    ctv.fit(list(xtrain) + list(xvalid))
    xtrain_ctv = ctv.transform(xtrain)
    xvalid_ctv = ctv.transform(xvalid)
    return xtrain_ctv, xvalid_ctv, ctv


def SVD(xtrain, xvalid):
    # Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.
    svd = decomposition.TruncatedSVD(n_components=120)
    svd.fit(xtrain)
    xtrain_svd = svd.transform(xtrain)
    xvalid_svd = svd.transform(xvalid)

    # Scale the data obtained from SVD. Renaming variable to reuse without scaling.
    scl = preprocessing.StandardScaler()
    scl.fit(xtrain_svd)
    xtrain_svd_scl = scl.transform(xtrain_svd)
    xvalid_svd_scl = scl.transform(xvalid_svd)
    return xtrain_svd_scl, xvalid_svd_scl, xtrain_svd, xvalid_svd, svd, scl


def LR_train(xtrain, ytrain, xvalid, yvalid):
    # Fitting a simple Logistic Regression on TFIDF
    clf = LogisticRegression(C=1.0)
    clf.fit(xtrain, ytrain)
    predictions = clf.predict_proba(xvalid)

    loss = multiclass_logloss(yvalid, predictions)
    print("LR logloss: %0.3f " % loss)
    return loss, clf


def MNB(xtrain, ytrain, xvalid, yvalid):
    # Fitting a simple Naive Bayes on TFIDF
    clf = MultinomialNB()
    clf.fit(xtrain, ytrain)
    predictions = clf.predict_proba(xvalid)

    loss = multiclass_logloss(yvalid, predictions)
    print("MNB logloss: %0.3f " % loss)
    return loss, clf


def SVC_train(xtrain, ytrain, xvalid, yvalid):
    # Fitting a simple SVM
    clf = SVC(C=1.0, probability=True)  # since we need probabilities
    clf.fit(xtrain, ytrain)
    predictions = clf.predict_proba(xvalid)

    loss = multiclass_logloss(yvalid, predictions)
    print("SVC logloss: %0.3f " % loss)
    return loss, clf


def XGB_train(xtrain, ytrain, xvalid, yvalid):
    # Fitting a simple xgboost on tf-idf
    clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8,
                            subsample=0.8, nthread=10, learning_rate=0.1)
    clf.fit(xtrain, ytrain)
    predictions = clf.predict_proba(xvalid)

    loss = multiclass_logloss(yvalid, predictions)
    print("XGB logloss: %0.3f " % loss)
    return loss, clf


def LR_GridSearch(xtrain, ytrain):
    mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)
    # Initialize SVD
    svd = TruncatedSVD()

    # Initialize the standard scaler
    scl = preprocessing.StandardScaler()

    # We will use logistic regression here..
    lr_model = LogisticRegression()

    # Create the pipeline
    clf = pipeline.Pipeline([('svd', svd),
                             ('scl', scl),
                             ('lr', lr_model)])

    param_grid = {'svd__n_components': [120, 180],
                  'lr__C': [0.1, 1.0, 10],
                  'lr__penalty': ['l1', 'l2']}
    # Initialize Grid Search Model
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                         verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    # Fit Grid Search Model
    model.fit(xtrain, ytrain)  # we can use the full data here but im only using xtrain
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def NB_GridSearch(xtrain, ytrain):
    mll_scorer = metrics.make_scorer(multiclass_logloss, greater_is_better=False, needs_proba=True)
    nb_model = MultinomialNB()

    # Create the pipeline
    clf = pipeline.Pipeline([('nb', nb_model)])

    # parameter grid
    param_grid = {'nb__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    # Initialize Grid Search Model
    model = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=mll_scorer,
                         verbose=10, n_jobs=-1, iid=True, refit=True, cv=2)

    # Fit Grid Search Model
    model.fit(xtrain, ytrain)  # we can use the full data here but im only using xtrain.
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def LR_predict(lr, tfv, test, sample, path="/home/webdev/ai/competition/spooky/result.csv"):
    text = tfv.transform(test.text.values)
    predictions = lr.predict_proba(text)
    sample[sample.columns.values[1]] = predictions[:, 0]
    sample[sample.columns.values[2]] = predictions[:, 1]
    sample[sample.columns.values[3]] = predictions[:, 2]
    sample.to_csv(path, index=False)


def load_GloVe():
    # load the GloVe vectors in a dictionary:
    embeddings_index = {}
    icount = 0
    f = open('/media/webdev/store/src/competition/spooky/glove.840B.300d.txt')
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            icount += 1
            print("error {0} {1}".format(icount, values))
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


# this function creates a normalized vector for the whole sentence
def sent2vec(s, embeddings_index):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


def GloVe(xtrain, xvalid, embeddings_index):
    # create sentence vectors using the above function for training and validation set
    xtrain_glove = [sent2vec(x, embeddings_index) for x in tqdm(xtrain)]
    xvalid_glove = [sent2vec(x, embeddings_index) for x in tqdm(xvalid)]
    xtrain_glove = np.array(xtrain_glove)
    xvalid_glove = np.array(xvalid_glove)
    return xtrain_glove, xvalid_glove


def to_categorical(y, num_classes=None):
    """ 1-hot encodes a tensor """
    if not num_classes:
        num_classes = len(np.unique(y))
    return np.eye(num_classes, dtype='uint8')[y]


if __name__ == "__main__":

    train = pd.read_csv('/media/webdev/store/src/competition/spooky/train.csv')
    test = pd.read_csv('/media/webdev/store/src/competition/spooky/test.csv')
    sample = pd.read_csv('/media/webdev/store/src/competition/spooky/sample_submission.csv')
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(train.author.values)

    xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y,
                                                      stratify=y,
                                                      random_state=42,
                                                      test_size=0.1, shuffle=True)

    y_train_enc = keras.utils.np_utils.to_categorical(ytrain)
    # y_train_enc1 = to_categorical(ytrain, 3)

    xtrain_tfv, xvalid_tfv, tfv = Tfidf(xtrain, xvalid)
    xtrain_ctv, xvalid_ctv, ctv = CountV(xtrain, xvalid)
    xtrain_svd_scl, xvalid_svd_scl, xtrain_svd, xvalid_svd, svd, scl = SVD(xtrain_tfv, xvalid_tfv)
    embedding_index = load_GloVe()
    xtrain_glove, xvalid_glove = GloVe(xtrain, xvalid, embedding_index)

    # loss, lr = LR_train(xtrain_tfv, ytrain, xvalid_tfv, yvalid)
    # loss, lr = LR_train(xtrain_ctv, ytrain, xvalid_ctv, yvalid)
    # loss, mnb = MNB(xtrain_tfv, ytrain, xvalid_tfv, yvalid)
    # loss, mnb = MNB(xtrain_ctv, ytrain, xvalid_ctv, yvalid)
    # loss, svc = SVC_train(xtrain_svd_scl, ytrain, xvalid_svd_scl, yvalid)
    # loss, svc = XGB_train(xtrain_tfv, ytrain, xvalid_tfv, yvalid)
    # loss, svc = XGB_train(xtrain_ctv, ytrain, xvalid_ctv, yvalid)
    # loss, svc = XGB_train(xtrain_svd, ytrain, xvalid_svd, yvalid)
    # loss, svc = XGB_train(xtrain_glove, ytrain, xvalid_glove, yvalid)
    # LR_GridSearch(xtrain_tfv, ytrain)
    # NB_GridSearch(xtrain_tfv, ytrain)
    # LR_predict(lr, tfv, test, sample)
    net1 = Net1()
    net2 = Net2()
    net3 = Net3()
    net4 = Net4()
    print(net1)
    print(net2)
    print(net3)
    print(net4)
    BaseTrain(net1)



