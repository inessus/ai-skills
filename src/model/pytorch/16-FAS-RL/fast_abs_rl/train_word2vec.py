""" pretrain a word2vec on the corpus"""
import argparse
<<<<<<< HEAD
import json
import logging
import os
=======
<<<<<<< HEAD
import json
import logging
import os
=======
import codecs
import json
import logging
import os
import pickle as pkl
import platform
from collections import Counter
>>>>>>> a6c58ab3456443573d54820cd38bc860739e28c8
>>>>>>> f0bd9a5b01fe49a55f538ece70dac34e89887f1f
from datetime import timedelta
from os.path import exists, join
from time import time

import gensim
<<<<<<< HEAD
from cytoolz import concatv

=======
<<<<<<< HEAD
from cytoolz import concatv

=======
import pandas as pd
from cytoolz import concatv

from utils import count_data
>>>>>>> a6c58ab3456443573d54820cd38bc860739e28c8
>>>>>>> f0bd9a5b01fe49a55f538ece70dac34e89887f1f
from data.data import JsonFileDataset


class Sentences(object):
    """ needed for gensim word2vec training"""
    def __init__(self, dataset):
        self._path = join(dataset.path, 'train')
        self._n_data = len(dataset)

    def __iter__(self):
        for i in range(self._n_data):
            with open(join(self._path, '{}.json'.format(i))) as f:
                data = json.loads(f.read())
            for s in concatv(data['article'], data['abstract']):
                yield ['<s>'] + s.lower().split() + [r'<\s>']


def main(args):
<<<<<<< HEAD
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
=======
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
>>>>>>> f0bd9a5b01fe49a55f538ece70dac34e89887f1f
    start = time()

    dataset = JsonFileDataset('train', 'cnndm', args.path)

    save_dir = join(args.path, "word2vec")
    if not exists(save_dir):
        os.makedirs(save_dir)

    sentences = Sentences(dataset)
<<<<<<< HEAD
    model = gensim.models.Word2Vec(size=args.dim, min_count=5, workers=16, sg=1)
=======
    model = gensim.models.Word2Vec(
        size=args.dim, min_count=5, workers=16, sg=1)
>>>>>>> f0bd9a5b01fe49a55f538ece70dac34e89887f1f
    model.build_vocab(sentences)
    print('vocab built in {}'.format(timedelta(seconds=time()-start)))
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

<<<<<<< HEAD
    model.save(join(save_dir, 'word2vec.{}d.{}k.bin'.format(args.dim, len(model.wv.vocab)//1000)))
    model.wv.save_word2vec_format(join(save_dir, 'word2vec.{}d.{}k.w2v'.format(args.dim, len(model.wv.vocab)//1000)))
=======
    model.save(join(save_dir, 'word2vec.{}d.{}k.bin'.format(
        args.dim, len(model.wv.vocab)//1000)))
    model.wv.save_word2vec_format(join(
        save_dir,
        'word2vec.{}d.{}k.w2v'.format(args.dim, len(model.wv.vocab)//1000)
    ))
>>>>>>> f0bd9a5b01fe49a55f538ece70dac34e89887f1f

    print('word2vec trained in {}'.format(timedelta(seconds=time()-start)))
    return model


if __name__ == '__main__':
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description='train word2vec embedding used for model initialization')
=======
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description='train word2vec embedding used for model initialization')
=======
    parser = argparse.ArgumentParser(
        description='train word2vec embedding used for model initialization'
    )
>>>>>>> a6c58ab3456443573d54820cd38bc860739e28c8
>>>>>>> f0bd9a5b01fe49a55f538ece70dac34e89887f1f
    parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--dim', action='store', type=int, default=128)
    args = parser.parse_args()

    main(args)
