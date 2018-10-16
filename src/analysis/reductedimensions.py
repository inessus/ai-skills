import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


# tfidf 之后降维
def TruncatedSVD_arpack(full_tfidf, train_tfidf, test_tfidf):
    n_comp = 3
    svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    svd_obj.fit(full_tfidf)
    train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
    test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
    train_svd.columns = ['svd_title_'+str(i+i) for i in range(n_comp)]
    test_svd.columns = ['svd_title_'+str(i+i) for i in range(n_comp)]
    return train_svd, test_svd