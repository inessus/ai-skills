""" pretrain a word2vec on the corpus"""
import argparse
import json
import logging
import os
from datetime import timedelta
from os.path import exists, join
from time import time

import gensim
from cytoolz import concatv
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
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    start = time()

    # dataset = JsonFileDataset('train', 'cnndm', args.path)
    dataset = JsonFileDataset('train', args.path)

    save_dir = join(args.path, "word2vec")
    if not exists(save_dir):
        os.makedirs(save_dir)

    sentences = Sentences(dataset)
    model = gensim.models.Word2Vec(size=args.dim, min_count=5, workers=16, sg=1)
    model.build_vocab(sentences)
    print('vocab built in {}'.format(timedelta(seconds=time()-start)))
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

    model.save(join(save_dir, 'word2vec.{}d.{}k.bin'.format(args.dim, len(model.wv.vocab)//1000)))
    model.wv.save_word2vec_format(join(save_dir, 'word2vec.{}d.{}k.w2v'.format(args.dim, len(model.wv.vocab)//1000)))

    print('word2vec trained in {}'.format(timedelta(seconds=time()-start)))
    return model

"""
python 02_train_word2vec.py --path=/media/webdev/store/competition/cnndm
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='trainer word2vec embedding used for model initialization')
    parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--dim', action='store', type=int, default=128)
    args = parser.parse_args()
    main(args)
    # --path=/Users/oneai/ai/data/bytecup/