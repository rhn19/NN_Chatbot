import os
import json
import random
import torch
import itertools
from vocab import Vocab

def load(voc_path, pair_path):
    """
    Load vocabulary from JSON dump. Load pairs from saved file
    @param voc_path (str): file path to vocab file
    @param pair_path (str): file path to pairs file
    @returns voc (Vocab) : Vocab object loaded from JSON dump
    @returns pairs (List[List(str)]) : Sentence Pairs loaded from file
    """
    # Only word2idx is used further. Loading other fields is optional
    voc = Vocab("Cornell_Movie")
    voc.trimmed = True

    entry = json.load(open(voc_path, 'r'))
    voc.word2idx = entry['word2idx']
    voc.word2cnt = entry['word2cnt']
    voc.idx2word = entry['idx2word']
    voc.num_words = len(voc.idx2word)

    #adding constants
    voc.word2idx['<PAD>'] = 0
    voc.word2idx['<SOS>'] = 1
    voc.word2idx['<EOS>'] = 2

    lines = open(pair_path, encoding='utf-8').\
        read().strip().split('\n')
    pairs = [[s for s in l.split('\t')] for l in lines]
    return voc, pairs

def indexesFromSentence(voc, sentence):
    """
    Convert words to indexes from vocab
    @param voc (Vocab) : Vocabulary object
    @param sentence (List[str]) : Tokenized sentence
    @returns (List[int]) : Indexed sentence
    """
    return [voc.word2idx[word] for word in sentence.split(' ')] + [voc.word2idx['<EOS>']]

def zeroPadding(l, fillvalue=0):
    """
    Pads seqences to the longest sequence length
    @param l (List[List[int]]) : Batch of sequences
    @param fillvalue (int) : Padding value (Default : 0)
    @returns (List[List[int]]) : Padded batch of sequences
    """
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=0):
    """
    Create masking matrix for packing sequences
    @param l (List[List[int]]) : Batch of sequences
    @param value (int) : Padding value (Default : 0)
    @returns m (List[List[int]]) : Matrix of masks
    """
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc):
    """
    Returns padded input sequence tensor and lengths
    @param l (List[List[str]]) : Batch of sequences
    @param voc (Vocab) : Vocabulary object
    @returns padVar (torch.LongTensor) : Padded input sequences
    @returns lengths (List[int]) : Lengths of inputs in batch
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    """
    Returns padded target sequence tensor, padding mask, and max target length
    @param l (List[List[str]]) : Batch of sequences
    @param voc (Vocab) : Vocabulary object
    @returns padVar (torch.LongTensor) : Padded output sequences
    @returns mask (List[List[int]]) : Matrix of masks
    @returns max_target_len (int) : Max of target sequences length
    """
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList, voc.word2idx['<PAD>'])
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    """
    Returns all items for a given batch of pairs
    @param voc (Vocab) : Vocabulary object
    @param pair_batch (List[List[List[str]]]) : Pairs of input output sequences
    @returns inp (torch.LongTensor) : Padded input sequences
    @returns lengths (List[int]) : Lengths of inputs in batch
    @returns output (torch.LongTensor) : Padded output sequences
    @returns mask (List[List[int]]) : Matrix of masks
    @returns max_target_len (int) : Max of target sequences length
    """
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

if __name__ == "__main__":
    voc_path = r"/content/generated/vocab.json"
    pair_path = r"/content/generated/processed_pairs.txt"
    voc, pairs = load(voc_path, pair_path)
    print(pairs[:5])
    print(voc.word2idx['knee'])

    #Sanity Checks
    small_batch_size = 5
    batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
    #batches = batch2TrainData(voc, pairs[:5])
    input_variable, lengths, target_variable, mask, max_target_len = batches

    print("input_variable:", input_variable)
    print("lengths:", lengths)
    print("target_variable:", target_variable)
    print("mask:", mask)
    print("max_target_len:", max_target_len)