import os
import json
import pandas as pd
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
