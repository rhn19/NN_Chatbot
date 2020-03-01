import re
import os
import csv
import json
import codecs
import unicodedata

MAX_SENT_LENGTH = 10    # Maximum sentence length to consider
MIN_WORD_COUNT = 3  # Minimum word count threshold for trimming

class Vocab:
    def __init__(self, corpus_name):
        self.name = corpus_name
        self.trimmed = False
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>"}
        self.num_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.num_words
            self.word2cnt[word] = 1
            self.idx2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2cnt[word] += 1

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2cnt.items():
            if v >= min_count:
                keep_words.append(k)
        
        print("Vocab trimmed to {}/{} = {:.4f}".format(len(keep_words), len(self.word2cnt), len(keep_words)/ len(self.word2cnt)))

        #reinit vocab
        self.word2idx = {}
        self.word2cnt = {}
        self.idx2word = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>"}
        self.num_words = 3

        for word in keep_words:
            self.addWord(word)

def unicodeToAscii(s):
    """
    Converts Unicode string to plain ASCII
    @param s (str) : Unicode String
    @returns (str) : ASCII String
    """
    #Better solution at https://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)  #Normal Form Decomposed for combined decomposed characters
        if unicodedata.category(c) != 'Mn'  #Mn is the character category (Nonspacing Mark)
    )

def normalizeString(s):
    """
    Lowercase, trim and remove non-letter characters
    @param s (str) : Unicode String
    @returns s (str) : Normalized ASCII String
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readPairs(datafile, corpus_name):
    """
    Reads Data Pairs & creates a Vocab object
    @param datafile (str) : Path to the data pairs file
    @param corpus_name (str) : Vocabulary Name for init
    @returns voc (Vocab) : Vocab object
    @returns pairs (List[List(str)]) : List of normalized sentence pairs
    """
    print("Reading lines...")
    # Read the file and split into lines
    lines = open(datafile, encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    voc = Vocab(corpus_name)
    return voc, pairs

def filterPair(p):
    """
    True if both sentences are under MAX_SENT_LENGTH
    @param p (List(str)) : Sentence Pair
    @returns (bool) : Answer for condition
    """
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_SENT_LENGTH and len(p[1].split(' ')) < MAX_SENT_LENGTH

def filterPairs(pairs):
    """
    Filter Pairs according to MAX_SENT_LENGTH condition
    @param pairs (List[List(str)]) : List of sentence pairs
    @returns (List[List(str)]) : List of sentence pairs under MAX_SENT_LENGTH
    """
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(datafile, corpus_name):
    """
    Prepare Data & Populate Vocab
    @param datafile (str) : Path to the data pairs file
    @param corpus_name (str) : Vocabulary Name for init
    @returns voc (Vocab) : Populated Vocab object
    @returns pairs (List[List(str)]) : List of normalized & filtered sentence pairs
    """
    print("Start preparing training data ...")
    voc, pairs = readPairs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

def trimRareWords(voc, pairs, MIN_COUNT):
    """
    Trim Words below threshold
    @param voc (Vocab) : Populated Vocab object
    @param pairs (List[List(str)]) : List of normalized & filtered sentence pairs
    @param MIN_COUNT (int) : Minimum Threshold for word count
    @returns keep_pairs (List[List(str)]) : Sentence Pairs satisfying the condition
    """
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2idx:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2idx:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def save(voc, pairs, base_path):
    """ 
    Save Vocab to file as JSON dump.
    @param base_path (str): directory path to generated saves
    """
    vocab_file = os.path.join(base_path, "vocab.json")
    pairs_file = os.path.join(base_path, "processed_pairs.txt")
    # Only word2idx is used after processing (saving other 2 are optional)
    # word2idx won't contain <SOS>, <EOS>, <PAD> due to the way in which Vocab was built
    json.dump(dict(word2idx=voc.word2idx, word2cnt=voc.word2cnt, idx2word=voc.idx2word), open(vocab_file, 'w'), indent=2)

    delimiter = '\t'
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    with open(pairs_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter, lineterminator='\n')
        for pair in pairs:
            writer.writerow(pair)

def buildVocab(datafile, corpus_name, base_path, MIN_COUNT):
    """
    Builds & saves Vocab to a JSON dump & Pairs to a file
    @param datafile (str) : Path to the data pairs file
    @param corpus_name (str) : Vocabulary Name for init
    @param base_path (str): directory path to generated saves
    @param MIN_COUNT (int) : Minimum Threshold for word count
    """
    voc, pairs = prepareData(datafile, corpus_name)
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    print("Saving Vocab & Processed Pairs...")
    save(voc, pairs, base_path)

if __name__ == "__main__":
    base_path = r"./generated"
    datafile = os.path.join(base_path, "formatted_pairs.txt")
    buildVocab(datafile, "Cornell_Movie", base_path, MIN_WORD_COUNT)