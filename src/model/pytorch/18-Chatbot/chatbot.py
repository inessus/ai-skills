from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import codecs
from io import open
import torch
import matplotlib
import torch.nn as nn
import csv
import torch.optim as optim

matplotlib.use('TkAgg')


from data.prepare import printLines, fileFilter, loadLines, \
    loadConversations, extractSentencePairs, loadPrepareData, \
    trimRareWords, batch2TrainData

from model.seq2seq import LuongAttnDecoderRNN, GreedySearchDecoder, EncoderRNN
from model.trainer import trainIters, \
    learning_rate,decoder_learning_ratio, n_iteration, \
    print_every, save_every, clip


def extract_data(corpus, datafile):
    # 首先开始查看设备类型
    USE_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if USE_CUDA else 'cpu')

    corpus = r'/Users/oneai/ai/data/cmdc/'
    printLines(os.path.join(corpus, "movie_lines.txt"))
    fileFilter(os.path.join(corpus, "movie_lines.txt"))

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
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"), lines,
                                      MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file..")
    with open(datafile, "w", encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter)
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)


if __name__ == '__main__':

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    corpus = r'/Users/oneai/ai/data/cmdc/'
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")
    # extract_data(corpus, datafile)

    corpus_name = "cornell movie-dialogs corpus"
    # Load/Assemble voc and pairs
    save_dir = os.path.join("data", "save")
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    MIN_COUNT = 3  # Minimum word count threshold for trimming
    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, MIN_COUNT)

    # Example for validation
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable) # size() = (longest, batch_size)
    print("lengths:", lengths) #(batchsize)
    print("target_variable:", target_variable) # size() = (longest, batch_size)
    print("mask:", mask) # size() = (longest, batch_size)
    print("max_target_len:", max_target_len) # (longest)

    # Configure models
    model_name = 'cb_model'
    attn_model = 'dot'
    # attn_model = 'general'
    # attn_model = 'concat'
    hidden_size = 500
    encoder_n_layers = 2
    decoder_n_layers = 2
    dropout = 0.1
    batch_size = 64

    # Set checkpoint to load from; set to None if starting from scratch
    loadFilename = None
    checkpoint_iter = 4000
    # loadFilename = os.path.join(save_dir, model_name, corpus_name,
    #                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                            '{}_checkpoint.tar'.format(checkpoint_iter))

    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Starting Training!")
    trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename)

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    # Begin chatting (uncomment and run the following line to begin)
    # evaluateInput(encoder, decoder, searcher, voc)


    # T is the max time steps,
    # B is the batch size
    # N is the hidden size