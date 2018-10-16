from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import random
import re
import unicodedata
import codecs
import math
import itertools
from io import open
import torch
import torchvision
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import csv
from cytoolz import map, filter, take

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')

corpus = r'/Users/oneai/ai/data/cmdc/'


def printLines(file, n=10):
    """
        打印文件的函数。相当于head命令
    :param file: 要读入的文件
    :param n:  打印的最大长度
    :return:
    """
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)


def fileFilter(file, n=10):
    with open(file, 'rb') as datafile:
        lines = datafile.readlines()
        return list(take(n, lines))


printLines(os.path.join(corpus, "movie_lines.txt"))
fileFilter(os.path.join(corpus, "movie_lines.txt"))


# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    """
        把文件按照关键字，组成二级字典
    :param fileName: 要处理的文件
    :param fields:  文件中的字典关键字
    :return:
    """
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            value = line.split(" +++$+++")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = value[i]

            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines form readLines into conversations based on movie_conversations.txt
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            lineIds = eval(convObj['utteranceIDs'])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the line of the conversation
        for i in range(len(conversation["lines"]) - 1):  #
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i + 1]["text"].strip()
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


# Define path to new file
datafile = os.path.join(corpus, "formatted_movie_lines.txt")


# Print a sample of lines
print("\nSample lines from file:")
printLines(datafile)


delimiter = '\t'
# Unescape the delimiter
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

# Initialize lines dict, conversations list, and field ids
lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

# Load lines and process conversations
print("\nProcessing corpus...")
lines = loadLines(os.path.join(corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)

print("\nLoading conversations...")
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"), lines, MOVIE_CONVERSATIONS_FIELDS)

# Write new csv file
print("\nWriting newly formatted file..")
with open(datafile, "w", encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter)
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)


import unicodedata
def shave_marks(txt):
    """去掉全部变音符号"""
    norm_txt = unicodedata.normalize('NFD', txt) # 把所有字符分解成基字符和组合记号
    shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c)) # 过滤掉所有组合记号。
    return unicodedata.normalize('NFC', shaved) # 重组所有字符

if __name__ == '__main__':
    print("test")