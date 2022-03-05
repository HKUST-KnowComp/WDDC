import numpy as np
from util.word_dict import WordDict
from collections import defaultdict
import os

def load_corpus(config, path, name, word2id, asp_word2id, filter_null=False):

    with open(path, "r", encoding="iso-8859-1") as fh:
        lines = fh.readlines()

    segs = [line.strip().split('\t\t\t') for line in lines]
    tmp_x = [ seg[2].split() for seg in segs ]
    asp_senti = [ seg[1].split('\t') for seg in segs ]

    senti = []
    weight = []
    valid = []
    senti_words = WordDict()

    for idx, sample in enumerate(asp_senti):
        sample_weight = []
        sample_senti = []
        sample_valid = False
        for i in range(len(sample) // 2):
            asp_ = sample[2 * i]
            senti_ = sample[2 * i + 1]
            wei_ = 1.
            if " no" in senti_:
                senti_ = senti_.split()[0].strip()
                wei_ = -1.

            if senti_ in asp_word2id: 
                sample_senti.append(asp_word2id[senti_])
                senti_words.add(asp_word2id[senti_])
                sample_weight.append(wei_)
                sample_valid = True

        senti.append(sample_senti)
        weight.append(sample_weight)
        valid.append(sample_valid)

    corpus_x = [[word2id[word] for word in doc] for doc in tmp_x]
    corpus_y = [ int(seg[0])-1 for seg in segs]
    assert len(corpus_x) == len(corpus_y)

    if filter_null:
        corpus_x = [corpus_x[i] for i, v in enumerate(valid) if v is True]
        corpus_y = [corpus_y[i] for i, v in enumerate(valid) if v is True]
        senti = [senti[i] for i, v in enumerate(valid) if v is True]
        weight = [weight[i] for i, v in enumerate(valid) if v is True]
    print(len(corpus_x))
    return corpus_x, corpus_y, senti, weight, senti_words


def load_embedding(config, path, normalized=False):
    word2idx_dict = defaultdict(int)
    embedding = [np.zeros(config.emb_dim).tolist()]
    with open(path, "r", encoding="iso-8859-1") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip().split()
            word = " ".join(line[:-config.emb_dim])
            word2idx_dict[word] = i
            vec = list(map(float, line[-config.emb_dim:]))
            embedding.append(vec)
    if normalized:
        embedding = np.array(embedding, dtype=np.float32)
        embedding = embedding / (np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)+1e-12)
    return word2idx_dict, embedding



