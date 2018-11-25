import os
import re
import json
import shutil
import pandas as pd
from cytoolz import curry
from os.path import join
from datetime import datetime
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from library.utils.datasets.dictionary import VocV1
from library.utils.datasets.jsonfile import count_train_txt, count_txt


def convert_p2j(src_path, dest_path, voc_path=""):
    iCount = 0
    voc = VocV1(need_normal=True)

    for i in range(count_train_txt(src_path)):
        read_path = os.path.join(src_path, "bytecup.corpus.trainer.{}.txt".format(i))
        dw = pd.read_json(read_path, lines=True)
        for j in dw.index:
            data = {'article': dw.loc[j, 'content'].split('.'), 'abstract': [dw.loc[j, 'title']]}

            voc.addSentences(data['article'])
            voc.addSentences(data['abstract'])
            json.dump(data, open(os.path.join(dest_path, "{}.json".format(iCount)), 'w'))
            iCount += 1
            if iCount % 10000 == 0:
                print("{}-{}- lines: {} VOC:{}".format(i, j, iCount, len(voc.wc)))
        print("{}".format(read_path))

        if os.path.isdir(voc_path):
            print("saving the vocabulary {}".format(len(voc.wc)))
            voc.save_vc(voc_path)


def convert_txt_json(src_path, dest_path):
    for i in range(1, count_txt(src_path)+1):
        read_path = os.path.join(src_path, "{}.txt".format(i))
        with open(read_path, 'r') as fread:
            data = {'article': [fread.read()]}
            json.dump(data, open(os.path.join(dest_path, "{}.json".format(i-1)), 'w'))


def make_test_split(src_path, dest_pat):
    count = count_json(src_path)
    train_X, test_X = train_test_split(range(count), test_size=0.1)
    start = datetime.now()
    with mp.Pool() as pool:
        list(pool.imap_unordered(process_test(src_path, dest_pat), zip(test_X, list(range(len(test_X)))), chunksize=1024))
    print("end time {}".format(datetime.now()-start))

def count_json(path):
    """
        追加的统计文件个数的方法，计算json文件个数
        count number of data in the given path
    """
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


@curry
def process_test(src, dest, i):
    test, d_test = i
    shutil.copy(join(src, "{}.json".format(test)), join(dest, "{}.json".format(d_test)))

