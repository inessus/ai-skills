import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize
from nltk.corpus import stopwords

import warnings

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
    return xtrain_svd_scl, xvalid_svd_scl, svd, scl


def LR_train(xtrain, ytrain, xvalid, yvalid):
    # Fitting a simple Logistic Regression on TFIDF
    clf = LogisticRegression(C=1.0)
    clf.fit(xtrain, ytrain)
    predictions = clf.predict_proba(xvalid)

    loss = multiclass_logloss(yvalid, predictions)
    print("logloss: %0.3f " % loss)
    return loss, clf


def MNB(xtrain, ytrain, xvalid, yvalid):
    # Fitting a simple Naive Bayes on TFIDF
    clf = MultinomialNB()
    clf.fit(xtrain, ytrain)
    predictions = clf.predict_proba(xvalid)

    loss = multiclass_logloss(yvalid, predictions)
    print("logloss: %0.3f " % loss)
    return loss, clf


def SVC_train(xtrain, ytrain, xvalid, yvalid):
    # Fitting a simple SVM
    clf = SVC(C=1.0, probability=True)  # since we need probabilities
    clf.fit(xtrain, ytrain)
    predictions = clf.predict_proba(xvalid)

    loss = multiclass_logloss(yvalid, predictions)
    print("logloss: %0.3f " % loss)
    return loss, clf


def LR_predict(lr, tfv, test, sample, path="/home/webdev/ai/competition/spooky/result.csv"):
    text = tfv.transform(test.text.values)
    predictions = lr.predict_proba(text)
    sample[sample.columns.values[1]] = predictions[:, 0]
    sample[sample.columns.values[2]] = predictions[:, 1]
    sample[sample.columns.values[3]] = predictions[:, 2]
    sample.to_csv(path, index=False)


if __name__ == "__main__":
    train = pd.read_csv('/home/webdev/ai/competition/spooky/train.csv')
    test = pd.read_csv('/home/webdev/ai/competition/spooky/test.csv')
    sample = pd.read_csv('/home/webdev/ai/competition/spooky/sample_submission.csv')
    lbl_enc = preprocessing.LabelEncoder()
    y = lbl_enc.fit_transform(train.author.values)

    xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y,
                                                      stratify=y,
                                                      random_state=42,
                                                      test_size=0.1, shuffle=True)

    xtrain_tfv, xvalid_tfv, tfv = Tfidf(xtrain, xvalid)
    xtrain_ctv, xvalid_ctv, ctv = CountV(xtrain, xvalid)
    xtrain_svd_scl, xvalid_svd_scl, svd, scl = SVD(xtrain_tfv, xvalid_tfv)
    loss, lr = LR_train(xtrain_tfv, ytrain, xvalid_tfv, yvalid)
    loss, lr = LR_train(xtrain_ctv, ytrain, xvalid_ctv, yvalid)
    loss, mnb = MNB(xtrain_tfv, ytrain, xvalid_tfv, yvalid)
    loss, mnb = MNB(xtrain_ctv, ytrain, xvalid_ctv, yvalid)
    loss, svc = SVC_train(xtrain_svd_scl, ytrain, xvalid_svd_scl, yvalid)
    # LR_predict(lr, tfv, test, sample)

