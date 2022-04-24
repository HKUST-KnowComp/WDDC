import glob
import os
import torch
import numpy as np
import string
import argparse
from tqdm import tqdm
import copy
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, roc_curve, auc
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
import random
from nltk.stem import WordNetLemmatizer
import math
from os import listdir
from os.path import isfile, join, exists
import json
from scipy.sparse import csr_matrix
import scipy.sparse
import copy
from transformers import BertTokenizer
from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence


class Extract():
    def __init__(self, args, file_name, label_name, data_dir, output_dir, vws_dir, esa_dir, input_dir, top=10):
        self.file_name = file_name
        self.data_dir = data_dir
        self.input_dir = input_dir
        self.output_dir = output_dir
        if not exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.esa_dir = esa_dir

        self.vws_dir = vws_dir
        if not exists(self.vws_dir):
            os.makedirs(self.vws_dir)
        self.label_name = label_name
        self.num_class = len(label_name)

        self.stop_words = set(['than', 'to', 'for', 'about', 'same', 'by', 'where',
                               'been', 'being', 'mightn', "shan't", 'wouldn', 'me', 'us', 'yours', 'you', 'he',
                               'here', "she's", 'she', 'i', "mustn't", 'y', 'our', 'those', 'haven', 'too',
                               'don', 'because', "won't", 'on', 'against', 'has', 'as', 'doing', "that'll",
                               'below', 'how', 'up', 'they', 'won', 'that', "aren't", 'some', 'so', 'theirs',
                               'didn', 'should', 'weren', 'having', 'an', 'nor', "needn't", 'of', 'yourselves',
                               'had', 'then', 'from', 'myself', 'few', "weren't", 'ours', 'couldn', 'will',
                               'needn', 'doesn', 'whom', 'themselves', "didn't", 'more', 'yourself', 'after',
                               'ain', 'are', 'does', "hasn't", 'ma', 'have', 'but', 'who', 'were', 'out',
                               'not', 'only', 'very', 'd', 'hers', 'what', 'my', 'and', "isn't", 'is',
                               'until', 'such', 'or', 't', 's', 'do', 'while', "you'll", 'it', 'their', 'am',
                               'was', 'be', 'shan', "couldn't", 'over', 'its', 'in', 'these', 've', "doesn't",
                               'we', 'can', 'hadn', 'his', "it's", 'other', 're', 'at', 'you', 'this', 'hasn',
                               'the', 'further', 'both', "you'd", 'your', "should've", 'a', 'any', 'why',
                               "shouldn't", "haven't", 'isn', 'her', "you're", 'again', "wasn't", 'did', "hadn't",
                               'own', "mightn't", 'down', 'herself', 'o', 'aren', 'shouldn', 'him', 'once', 'there',
                               'most', 'mustn', 'off', 'ourselves', 'each', 'above', 'now', 'before', 'with', 'under',
                               "don't", "wouldn't", 'which', 'if', 'when', 'himself', 'wasn', 'all', 'just', 'through',
                               'them', 'between', 'would', 'she', 'm', 'during', 'no', 'itself', "you've", 'into', 'll',
                               '.', '...', '?', '!', 'something', 'everything', 'nothing', 'wikipedia'])

        self.emb_dim = 768

        self.get_POS_input()
        self.get_POS_output()

        self.corpus = {}
        self.lemmatizer, self.lemmatizer_inv = self.lemmatize()

        self.get_corpus('train')
        self.get_noun('train')

        self.get_corpus('dev')
        self.get_noun('dev')

        self.get_corpus('test')
        self.get_noun('test')

        self.threshold = 0.6

        print('threshold', self.threshold)
        self.lemmatizer, self.lemmatizer_inv = self.lemmatize()
        self.get_pseudo_train()

        self.lemmatizer, self.lemmatizer_inv = self.lemmatize()
        self.get_common_words()

        self.get_contextualized_words()
        self.get_corpus_for_VWS()
        self.get_keyword_emb()

    def is_contain_digit_or_punctuation(self, word):
        for char in word:
            if char.isdigit():
                return True
            elif char in string.punctuation:
                return True
        return False

    def get_corpus_for_VWS(self):

        if not exists(self.vws_dir):
            os.makedirs(self.vws_dir)

        keywords = set()

        with open(join(self.output_dir, 'contextualized_words_{}'.format(self.threshold))) as fin:
            for line in fin:
                keywords.add(line.strip())

        with open(join(self.output_dir, 'common_words_across_class_{}.txt'.format(self.threshold))) as fin:
            for line in fin:
                word = line.strip()
                if word in keywords:
                    keywords.remove(word)

        with open(join(self.vws_dir, 'train'), 'w') as fo:
            with open(join(self.output_dir, 'new_train')) as fin:
                for line in fin:
                    label, tuples, text = line.strip().split('\t\t\t')
                    new_tuples = set()
                    for w in tuples.split():
                        w = w.lower()
                        if w in keywords:
                            new_tuples.add(w)
                        elif w in self.lemmatizer and self.lemmatizer[w] in keywords:
                            new_tuples.add(self.lemmatizer[w])
                    fo.write(label + '\t\t\t' + '\t'.join(new_tuples) + '\t\t\t' + text + '\n')

        with open(join(self.vws_dir, 'dev'), 'w') as fo:
            with open(join(self.output_dir, 'new_dev')) as fin:
                for line in fin:
                    label, tuples, text = line.strip().split('\t\t\t')
                    new_tuples = set()
                    for w in tuples.split():
                        w = w.lower()
                        if w in keywords:
                            new_tuples.add(w)
                        elif w in self.lemmatizer and self.lemmatizer[w] in keywords:
                            new_tuples.add(self.lemmatizer[w])
                    fo.write(label + '\t\t\t' + '\t'.join(new_tuples) + '\t\t\t' + text + '\n')

        with open(join(self.vws_dir, 'test'), 'w') as fo:
            with open(join(self.output_dir, 'new_test')) as fin:
                for line in fin:
                    label, tuples, text = line.strip().split('\t\t\t')
                    new_tuples = set()
                    for w in tuples.split():
                        w = w.lower()
                        if w in keywords:
                            new_tuples.add(w)
                        elif w in self.lemmatizer and self.lemmatizer[w] in keywords:
                            new_tuples.add(self.lemmatizer[w])
                    fo.write(label + '\t\t\t' + '\t'.join(new_tuples) + '\t\t\t' + text + '\n')

    def get_keyword_emb(self):
        word2idx, idx2word, label_word2idx, label_idx2word, label_embedding = self.get_vocab_emb_mean()
        keywords = []
        self.word_embedding = TransformerWordEmbeddings('bert-base-uncased', layers='-1')

        with open(join(self.output_dir, 'contextualized_words_{}'.format(self.threshold))) as fin:
            for line in fin:
                keywords.append(line.strip())

        with open(join(self.output_dir, 'common_words_across_class_{}.txt'.format(self.threshold))) as fin:
            for line in fin:
                word = line.strip()
                if word in keywords:
                    keywords.remove(word)

        with open(join(self.output_dir, 'op_emb_bert_sup_{}'.format(self.threshold)), 'w') as fo:
            for w in keywords:
                if w in word2idx:
                    if w in self.lemmatizer_inv and self.lemmatizer_inv[w] in word2idx:
                        mean_emb = np.mean(
                            [self.embedding[word2idx[w]], self.embedding[word2idx[self.lemmatizer_inv[w]]]], axis=0)
                        fo.write(w + ' ' + ' '.join(list(map(str, mean_emb))) + '\n')
                    else:
                        fo.write(w + ' ' + ' '.join(list(map(str, self.embedding[word2idx[w]]))) + '\n')
                else:
                    sentence = Sentence(w)
                    self.word_embedding.embed(sentence)
                    for tk in sentence:
                        word = tk.text
                        word = word.lower()
                        emb = tk.embedding.detach().cpu().numpy()
                        emb = list(map(str, emb))
                        fo.write(w + ' ' + ' '.join(emb) + '\n')

    def get_contextualized_words(self):
        word2idx, idx2word, label_word2idx, label_idx2word, label_embedding = self.get_vocab_emb_mean()
        keywords = Counter()
        num_keyword_token = 0

        common_words = set()
        with open(join(self.output_dir, 'common_words_across_class_{}.txt'.format(self.threshold))) as fin:
            for line in fin:
                common_words.add(line.strip())

        with open(join(self.output_dir, 'contextualized_words_{}'.format(self.threshold)), 'w') as fo:
            with open(join(self.output_dir, 'pseudo_train_{}'.format(self.threshold))) as fin:
                for line in fin:
                    words = set(line.strip().split('\t\t\t')[1].split('\t'))
                    words = words - set(self.stop_words)
                    for w in words:
                        w = w.lower()
                        if len(w) > 1:
                            if self.is_contain_digit_or_punctuation(w):
                                continue
                            if w in word2idx and w in self.lemmatizer and self.lemmatizer[w] in word2idx:
                                keywords.update([self.lemmatizer[w]])
                            else:
                                keywords.update([w])
                    num_keyword_token += len(words)

            for (w, f) in keywords.most_common(len(keywords)):
                if w not in common_words:
                    if f >= 10:
                        print("{:15} {:.6f} {:5}".format(w, f / num_keyword_token, f))
                    if f / num_keyword_token >= 0.0003:
                        fo.write(w + '\n')

    def get_common_words(self):
        label2idx = {label.split('/')[0]: i + 1 for i, label in enumerate(self.label_name)}
        idx2label = {i + 1: label.split('/')[0] for i, label in enumerate(self.label_name)}

        total_num_words = {label.split('/')[0]: 0 for label in self.label_name}
        total_num_docs = {label.split('/')[0]: 0 for label in self.label_name}
        all_words = {label.split('/')[0]: Counter() for label in self.label_name}
        all_words_dist = {}

        word_vocab = Counter()

        with open(join(self.output_dir, 'pseudo_train_{}'.format(self.threshold))) as fin:
            for line in fin:
                label, tuples, text = line.strip().split('\t\t\t')
                tuples = [w.lower() for w in tuples.split('\t')]
                total_num_docs[idx2label[int(label)]] += 1
                total_num_words[idx2label[int(label)]] += len(tuples)
                all_words[idx2label[int(label)]].update(tuples)
                word_vocab.update(tuples)

        for ii, label in enumerate(self.label_name):
            label = label.split('/')[0]
            for (w, f) in all_words[label].most_common(len(all_words[label])):
                if w not in all_words_dist:
                    all_words_dist[w] = {l.split('/')[0]: 0 for l in self.label_name}
                all_words_dist[w][label] = f / total_num_words[label] * 1e5

        print(total_num_words)
        total_num_words_all_labels = 0
        for l in total_num_words:
            total_num_words_all_labels += total_num_words[l]
        # for (w,f) in word_vocab.most_common(len(word_vocab)):
        # 	print(w,f)

        with open(join(self.output_dir, 'common_words_across_class_{}.txt'.format(self.threshold)), 'w') as fo:
            for (word, freq) in word_vocab.most_common(len(word_vocab)):
                freq_list = [all_words_dist[word][l] for l in all_words_dist[word]]
                freq_list = sorted(freq_list)
                if freq_list[-2] != 0:
                    first_sec_diff = freq_list[-1] / freq_list[-2]
                else:
                    first_sec_diff = 10000
                if freq_list[-3] != 0:
                    sec_third_diff = freq_list[-1] / freq_list[-3]
                else:
                    sec_third_diff = 10000

                print("{:15} {:.6f}".format(word, freq / total_num_words_all_labels), end=' ')
                for l in all_words_dist[word]:
                    print("{:10} {:8.2f}".format(l, all_words_dist[word][l]), end='         ')
                print("{:8.2f} {:8.2f}".format(first_sec_diff, sec_third_diff), end='\n\n')

                # if first_sec_diff <= 2.0:
                # 	if sec_third_diff >= 50.0:
                # 		pass
                # 	else:
                # 		fo.write(word+'\n')

                # if first_sec_diff > 2.0 and first_sec_diff <= 10.0 and sec_third_diff <= 25.0:
                # 	fo.write(word+'\n')

                if first_sec_diff <= 2.0:
                    fo.write(word + '\n')

    def get_vocab_emb_mean(self):
        # use the vocab_emb_mean generated by agnews_mask.py
        emb_file = join(self.data_dir, 'vocab_emb_mean')
        word2idx = {}
        idx2word = {}
        embedding = []

        with open(emb_file, "r", encoding="iso-8859-1") as fin:
            for i, line in enumerate(fin):
                line = line.strip().split()
                word = " ".join(line[:-self.emb_dim])
                word2idx[word] = i
                idx2word[i] = word
                vec = list(map(float, line[-self.emb_dim:]))
                embedding.append(vec)
        self.embedding = np.array(embedding)

        label_word2idx = {ln: i for i, ln in enumerate(self.label_name)}
        label_idx2word = {i: ln for i, ln in enumerate(self.label_name)}
        label_embedding = []

        for i, lns in enumerate(self.label_name):
            lns = lns.lower()
            lns = lns.split('/')
            emb_of_lns = []
            for ln in lns:
                if ln in word2idx:
                    if ln in self.lemmatizer and self.lemmatizer[ln] in word2idx:
                        mean = np.mean([self.embedding[word2idx[ln]], self.embedding[word2idx[self.lemmatizer[ln]]]],
                                       axis=0)
                        emb_of_lns.append(mean)
                    elif ln in self.lemmatizer_inv and self.lemmatizer_inv[ln] in word2idx:
                        mean = np.mean(
                            [self.embedding[word2idx[ln]], self.embedding[word2idx[self.lemmatizer_inv[ln]]]], axis=0)
                        emb_of_lns.append(mean)
                    else:
                        emb_of_lns.append(self.embedding[word2idx[ln]])
            label_embedding.append(emb_of_lns)

        return word2idx, idx2word, label_word2idx, label_idx2word, label_embedding

    def get_pseudo_train(self):

        word2idx, idx2word, label_word2idx, label_idx2word, label_embedding = self.get_vocab_emb_mean()

        num_correct = 0
        num_total = 0

        pseudo = []
        golds = []

        print('{} classes'.format(len(label_word2idx)))

        with open(join(self.output_dir, 'pseudo_train_{}'.format(self.threshold)), 'w') as fo:
            for file_name in ['train', 'test']:
                with open(join(self.output_dir, 'new_{}'.format(file_name))) as fin:
                    for line_id, line in tqdm(enumerate(fin)):
                        ground_true, words, text = line.strip().split('\t\t\t')
                        words = [w.lower() for w in words.split('\t') if w.lower() not in self.stop_words]
                        words.reverse()
                        words_after_lemmatize = []
                        keyword_emb = []
                        for w in words:
                            if w in word2idx:
                                if w in self.lemmatizer and self.lemmatizer[w] in word2idx:
                                    if self.lemmatizer[w] not in words_after_lemmatize:
                                        mean_emb = np.mean(
                                            [self.embedding[word2idx[w]], self.embedding[word2idx[self.lemmatizer[w]]]],
                                            axis=0)
                                        keyword_emb.append(mean_emb)
                                        words_after_lemmatize.append(self.lemmatizer[w])
                                elif w in self.lemmatizer_inv and self.lemmatizer_inv[w] in word2idx:
                                    if w not in words_after_lemmatize:
                                        mean_emb = np.mean([self.embedding[word2idx[w]],
                                                            self.embedding[word2idx[self.lemmatizer_inv[w]]]], axis=0)
                                        keyword_emb.append(mean_emb)
                                        words_after_lemmatize.append(w)
                                else:
                                    if w not in words_after_lemmatize:
                                        keyword_emb.append(self.embedding[word2idx[w]])
                                        words_after_lemmatize.append(w)
                            else:
                                words_after_lemmatize.append(w)

                        if len(keyword_emb) != 0:
                            keyword_emb = np.array(keyword_emb)
                            mean_sim = []
                            for label_i in range(len(self.label_name)):
                                one_label_embedding = np.array(label_embedding[label_i])
                                sim_matrix = cosine_similarity(keyword_emb, one_label_embedding)
                                # sim_matrix = np.sort(sim_matrix,axis=0)
                                # sim_matrix = sim_matrix[:10,:]
                                mean_value = np.amax(np.mean(sim_matrix, axis=0))
                                mean_sim.append(mean_value)
                            mean_ = np.argmax(mean_sim)
                            mean_value = np.max(mean_sim)
                            pseudo_label = mean_ + 1
                            if mean_value >= self.threshold:
                                pseudo.append(pseudo_label)
                                golds.append(int(ground_true))
                                if int(ground_true) == pseudo_label:
                                    num_correct += 1
                                fo.write(str(pseudo_label) + '\t\t\t' + "\t".join(
                                    words_after_lemmatize) + '\t\t\t' + text + '\n')
                                num_total += 1

        print(metrics.confusion_matrix(golds, pseudo))
        print("{}/{} {:.4f}".format(num_correct, num_total, num_correct / num_total))

    def lemmatize(self):
        lemmatizer = {}
        lemmatizer_inv = {}
        with open(join(self.data_dir, 'lemmatize')) as fin:
            for line in fin:
                before, after = line.strip().split()
                lemmatizer[before] = after
                lemmatizer_inv[after] = before
        return lemmatizer, lemmatizer_inv

    def get_POS_input(self):
        for file_name in ['train', 'dev', 'test']:
            with open(join(self.input_dir, file_name)) as fin:
                with open(join(self.output_dir, '{}_POS'.format(file_name)), 'w') as fo:
                    for line_id, line in enumerate(fin):
                        label, text = line.strip().split('\t\t\t')

                        if len(text.strip()) >= 5:
                            if text.strip()[-5:] == '. . .':
                                text = text.strip()[:-5] + ' .'
                            elif text.strip()[-5:] == '!!!!!':
                                text = text.strip()[:-5] + ' .'
                        if len(text.strip()) >= 3:
                            if text.strip()[-3:] == '...':
                                text = text.strip()[:-3] + ' .'
                            elif text.strip()[-3:] == '?!?':
                                text = text.strip()[:-3] + ' .'
                            elif text.strip()[-3:] == '!!!':
                                text = text.strip()[:-3] + ' .'
                            elif text.strip()[-3:] == '?!!':
                                text = text.strip()[:-3] + ' .'
                        if len(text.strip()) >= 2:
                            if text.strip()[-2:] == '?!':
                                text = text.strip()[:-2] + ' .'
                            elif text.strip()[-2:] == '!?':
                                text = text.strip()[:-2] + ' .'
                            elif text.strip()[-2:] == '!!':
                                text = text.strip()[:-2] + ' .'
                        if text.strip()[-1] not in ['.', '?', '!']:
                            text += ' .'
                        fo.write(text + '\n')
                        fo.write('\nThis is the end of document.\n\n')

    def get_POS_output(self):
        command = "java -Xmx6G -cp /path_to_stanford-parser-4.0.0/stanford-parser.jar:path_to_stanford-parser-4.0.0/stanford-parser-4.0.0-models.jar edu.stanford.nlp.parser.lexparser.LexicalizedParser -retainTMPSubcategories -outputFormat \"wordsAndTags\" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz {}/train_POS 2>&1 | tee {}/train_POS_out".format(self.output_dir, self.output_dir)
        os.system(command)

        command = "java -Xmx6G -cp /path_to_stanford-parser-4.0.0/stanford-parser.jar:/path_to_stanford-parser-4.0.0/stanford-parser-4.0.0-models.jar edu.stanford.nlp.parser.lexparser.LexicalizedParser -retainTMPSubcategories -outputFormat \"wordsAndTags\" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz {}/dev_POS 2>&1 | tee {}/dev_POS_out".format(self.output_dir, self.output_dir)
        os.system(command)

        command = "java -Xmx6G -cp /path_to_stanford-parser-4.0.0/stanford-parser.jar:/path_to_stanford-parser-4.0.0/stanford-parser-4.0.0-models.jar edu.stanford.nlp.parser.lexparser.LexicalizedParser -retainTMPSubcategories -outputFormat \"wordsAndTags\" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz {}/test_POS 2>&1 | tee {}/test_POS_out".format(self.output_dir, self.output_dir)
        os.system(command)

    def get_corpus(self, file_name):
        if file_name not in self.corpus:
            self.corpus[file_name] = []
        with open(join(self.input_dir, file_name)) as fin:
            for line_id, line in enumerate(fin):
                self.corpus[file_name].append(line)

    def finish_parse_doc_level(self, file_name, line_no, tuples_list_doc_level):
        # global cnt
        # cnt += 1

        unique_tuples = []
        for opinion in tuples_list_doc_level:
            if opinion in self.lemmatizer:
                opinion = self.lemmatizer[opinion]
            if opinion not in self.stop_words:
                unique_tuples.append(opinion)

        tuples_list_doc_level = set(unique_tuples)

        with open(join(self.output_dir, 'new_{}'.format(file_name)), 'a+') as fo:
            line = self.corpus[file_name][line_no]
            label, text = line.strip().split('\t\t\t')
            fo.write(label + '\t\t\t' + '\t'.join(tuples_list_doc_level) + '\t\t\t' + text + '\n')

    def get_noun(self, file_name):
        with open(join(self.output_dir, 'new_{}'.format(file_name)), 'w') as fo:
            pass
        actual_cnt = 0
        nn_pos = set(['NN', 'NNS', 'NNP', 'NNPS'])
        word_list = []
        pos_list = []
        tuples_list_sent_level = []
        word_pos_tag = False

        tuples_list_doc_level = []
        end_of_doc = 'This/DT is/VBZ the/DT end/NN of/IN document/NN ./.'

        with open(join(self.output_dir, '{}_POS_out'.format(file_name)), 'r') as fin:
            for i, line in enumerate(fin):
                if line.strip()[:len(end_of_doc)] == end_of_doc:

                    tuples_list_doc_level.extend(tuples_list_sent_level)

                    self.finish_parse_doc_level(file_name, actual_cnt, tuples_list_doc_level)

                    tuples_list_doc_level = []

                    word_pos_tag = False

                    tuples_list_sent_level = []

                    actual_cnt += 1


                elif line[:len('Parsing [sent.')] == 'Parsing [sent.':
                    word_pos_tag = True

                elif line == '\n':
                    continue

                elif word_pos_tag:
                    tuples_list_doc_level.extend(tuples_list_sent_level)
                    word_pos_tag = False
                    word_list = []
                    pos_list = []
                    tuples_list_sent_level = []

                    word_tag_pairs = line.strip('\n').split(' ')
                    for j in range(len(word_tag_pairs)):
                        pos = word_tag_pairs[j].split('/')[-1]
                        word = word_tag_pairs[j].rstrip('/' + pos)

                        pos_list.append(pos)
                        word_list.append(word.lower())
                        if pos in nn_pos:
                            tuples_list_sent_level.append(word.lower())
                else:
                    continue
        print(actual_cnt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='train',
                        help='file name')
    parser.add_argument('--data_dir', type=str, default=0,
                        help='data directory')
    parser.add_argument('--seg_no', type=int, default=0,
                        help='data directory')
    args = parser.parse_args()

    label_name = ['politics','sports','business','technology']
    data_dir = 'baseline_data/agnews'
    output_dir = 'output/agnews_extract'
    vws_dir = output_dir
    esa_dir = ''
    input_dir = data_dir
    print(output_dir)
    Extract(args, 'train', label_name, data_dir, output_dir, vws_dir, esa_dir, input_dir)




