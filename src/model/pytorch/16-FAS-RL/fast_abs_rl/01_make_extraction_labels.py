"""produce the dataset with (psudo) extraction label"""
import os
from os.path import exists, join
import json
from time import time
from datetime import timedelta
import multiprocessing as mp

from cytoolz import curry, compose

from utils import count_data
from metric import compute_rouge_l
from data.data import JsonFileDataset

"""
val 13368
train 287227
test 11490

{
    "id": "9acb288aebea0b1d9c5c49e14cd992f07e34506c",
    "article": [
        "by daily mail reporter .",
        "10:40 est , 25 july 2012 .",
        "a karate instructor who murdered a churchgoing teenager as she babysat two young children was jailed for 25 years today .",
7       "tony bushby , 19 , tricked promising student catherine wynter into going out with him by using fake facebook friends and then stabbed her to death as she looked after her niece and nephew .",
        "her elder sister had warned miss wynter , 19 , to ` lock the door and not let anybody in ' .",
11      "bushby stabbed her 23 times and left her body locked in the house with the youngsters overnight .",
        "when her mother went to check on her the following morning , she was told by the children , aged three and four , ` grandma , katie 's dead . '",
        "` christmas should be a peaceful happy time but for my family its meaning has been changed forever , and we will never be able to celebrate in the same way .",
28      "` i have been robbed of a future with my daughter . ' mrs davies added : ` all the evidence shows that bushby had planned to kill katie -- why , what had she done to him ?",
        "judge bright added that bushby posed a ` very real danger to women . '"
    ],
    "abstract": [
        "tony bushby , 19 , created fake facebook friends to trick art student catherine wynter into going out with him .",
        "he stabbed her 23 times as she babysat her niece and nephew on boxing day last year then locked them in the house with her body .",
        "mother : ` i have been robbed of a future with my daughter '"
    ],
    "extracted": [ # 文章与标题相对应的rouge-L最大的索引
        7,
        11,
        28
    ],
    "score": [
        0.6666666666666666,
        0.4074074074074074,
        0.8571428571428571
    ]
}%                                                                                                                                                                                                          
"""


# DATA_DIR = "/Users/oneai/ai/data/cnndm"
DATA_DIR = "/media/webdev/store/competition/cnndm/"


def _split_words(texts):
    return map(lambda t: t.split(), texts)


def get_extract_label(art_sents, abs_sents):
    """
        greedily match summary sentences to article sentences
        针对每个标题，与文章对应的句子计算rouge—L，选一个最大值，记录索引和分数
    """
    extracted = []
    scores = []
    indices = list(range(len(art_sents)))
    for abst in abs_sents:
        rouges = list(map(compute_rouge_l(reference=abst, mode='r'), art_sents))
        ext = max(indices, key=lambda i: rouges[i]) # 本篇文章所有句子索引，按照的rouge-L值排序
        indices.remove(ext)
        extracted.append(ext)
        scores.append(rouges[ext])
        if not indices:
            break
    return extracted, scores


@curry
def process(split, i):
    data_dir = join(DATA_DIR, split)
    with open(join(data_dir, '{}.json'.format(i))) as f:
        data = json.loads(f.read())
    tokenize = compose(list, _split_words)
    art_sents = tokenize(data['article'])
    abs_sents = tokenize(data['abstract'])
    if art_sents and abs_sents: # some data contains empty article/abstract
        extracted, scores = get_extract_label(art_sents, abs_sents)
    else:
        extracted, scores = [], []
    data['extracted'] = extracted
    data['score'] = scores
    with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
        json.dump(data, f, indent=4)


def label_mp(split):
    """ process the data split with multi-processing"""
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    with mp.Pool() as pool:
        list(pool.imap_unordered(process(split), list(range(n_data)), chunksize=1024))
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def label(split):
    start = time()
    print('start processing {} split...'.format(split))
    data_dir = join(DATA_DIR, split)
    n_data = count_data(data_dir)
    for i in range(n_data):
        print('processing {}/{} ({:.2f}%%)\r'.format(i, n_data, 100*i/n_data),
              end='')
        with open(join(data_dir, '{}.json'.format(i))) as f:
            data = json.loads(f.read())
        tokenize = compose(list, _split_words)
        art_sents = tokenize(data['article'])
        abs_sents = tokenize(data['abstract'])
        extracted, scores = get_extract_label(art_sents, abs_sents)
        data['extracted'] = extracted
        data['score'] = scores
        with open(join(data_dir, '{}.json'.format(i)), 'w') as f:
            json.dump(data, f, indent=4)
    print('finished in {}'.format(timedelta(seconds=time()-start)))


def main():
    for split in ['val', 'train']:  # no need of extraction label when testing
        label_mp(split)


if __name__ == '__main__':
    main()