import glob
import os
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from transformers import BertTokenizer, BertForMaskedLM
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
class Masked_Bert():
	def __init__ (self, args, file_name, label_name, data_dir, output_dir, vws_dir, esa_dir, top = 10):
		self.file_name = file_name
		self.data_dir = data_dir
		self.output_dir = output_dir
		self.esa_dir = esa_dir
		self.vws_dir = vws_dir
		if not exists(self.output_dir):
			os.makedirs(self.output_dir)
		self.dump_dir = join(output_dir,'bert_dump')
		if not exists(self.dump_dir):
			os.makedirs(self.dump_dir)
		self.label_name = label_name
		self.num_class = len(label_name)
		self.top = top
		self.emb_dim = 768
		self.wordnet_lemmatizer = WordNetLemmatizer()
		self.stop_words = set(['than', 'to', 'for', 'about', 'same', 'by', 'where', 
		'been', 'being', 'mightn', "shan't", 'wouldn', 'me', 'us','yours', 'you' ,'he', 
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
		'them', 'between', 'would', 'she', 'm', 'during', 'no', 'itself', "you've", 'into', 'll','.','...','?','!','something','everything','nothing','wikipedia'])
		
		self.threshold = 0.6
		print('pseudo threshold',self.threshold)
		self.lemmatizer, self.lemmatizer_inv = self.lemmatize()

		
		# 0 step
		# self.esa_similarity()
		# 1 step
		self.generate_masked_candidate()
		# 2 step
		self.lemmatizer, self.lemmatizer_inv = self.lemmatize()
		self.refine_lemmatizer()
		# 3 step
		self.get_freq_words()
		self.get_freq_words_on_masked_words()
		self.diff_freq_words()
		# 4 step
		self.static_rep()
		# 5 step
		self.mean_static_rep()
		# 6 step
		# self.lemmatizer, self.lemmatizer_inv = self.lemmatize()
		# self.infer()
		# 7 step
		self.lemmatizer, self.lemmatizer_inv = self.lemmatize()
		self.get_pseudo_train()
		# 8 step
		self.get_common_words()
		# 9 step
		self.lemmatizer, self.lemmatizer_inv = self.lemmatize()
		self.get_contextualized_words()
		self.get_corpus_for_VWS()
		self.get_keyword_emb()


	def lemmatize(self):
		lemmatizer = {}
		lemmatizer_inv = {}
		with open(join(self.data_dir,'lemmatize')) as fin:
			for line in fin:
				before, after = line.strip().split()
				lemmatizer[before] = after
				lemmatizer_inv[after] = before
		return lemmatizer, lemmatizer_inv

	def refine_lemmatizer(self):
		new_lemmatizer = set()
		wordnet_lemmatizer = WordNetLemmatizer()
		with open(join(self.output_dir,'new_train_top{}'.format(self.top))) as fin:
			for line_id, line in enumerate(fin):
				ground_true, words, text = line.strip().split('\t\t\t')
				for w in words.split('\t'):
					if w not in self.lemmatizer:
						if w != wordnet_lemmatizer.lemmatize(w):
							new_lemmatizer.add(" ".join([w,wordnet_lemmatizer.lemmatize(w)]))


	def generate_masked_candidate(self):
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		model = BertForMaskedLM.from_pretrained('bert-base-uncased')
		model.to('cuda')
		model.eval()

		for file_name in ['train','dev','test']:
			with open(join(self.output_dir,'new_{}_top{}'.format(file_name,self.top)),'w') as fo:
				with open(join(self.data_dir,file_name)) as fin:
					for line_id, line in tqdm(enumerate(fin)):
						label, original_text = line.strip().split('\t\t\t')
						if original_text.strip()[-1] not in ['.','!','?']:
							text = original_text + '.'
						else:
							text = copy.deepcopy(original_text)
						
						text = '[CLS] ' + text + ' This article is talking about [MASK]. [SEP]'

						tokenized_text = tokenizer.tokenize(text)
						
						masked_index = tokenized_text.index('[MASK]')
						
						tokens_index = tokenizer.convert_tokens_to_ids(tokenized_text)
						
						tokens_tensor = torch.tensor([tokens_index])
						tokens_tensor = tokens_tensor.to('cuda')

						with torch.no_grad():
							outputs = model(tokens_tensor)
							predictions = outputs[0]

							predictions = predictions.detach().cpu()
							predicted_index = np.argsort(predictions[0, masked_index])

							predicted_token = tokenizer.convert_ids_to_tokens(predicted_index[-self.top:])

						fo.write(label+'\t\t\t'+'\t'.join(predicted_token)+'\t\t\t'+original_text+'\n')


	def get_vocab_emb_mean(self):
		emb_file = join(self.data_dir,'vocab_emb_mean')
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

		label_word2idx = { ln:i for i, ln in enumerate(self.label_name) }
		label_idx2word ={ i:ln for i, ln in enumerate(self.label_name)} 
		label_embedding = []

		for i, lns in enumerate(self.label_name):
			lns = lns.lower()
			lns = lns.split('/')
			emb_of_lns = []
			for ln in lns:
				if ln in word2idx:
					if ln in self.lemmatizer and self.lemmatizer[ln] in word2idx:
						mean = np.mean( [self.embedding[word2idx[ln]] , self.embedding[word2idx[self.lemmatizer[ln]]] ] ,axis=0)
						emb_of_lns.append(mean)
					elif ln in self.lemmatizer_inv and self.lemmatizer_inv[ln] in word2idx:
						mean = np.mean( [self.embedding[word2idx[ln]] , self.embedding[word2idx[self.lemmatizer_inv[ln]]] ],axis=0)
						emb_of_lns.append(mean)
					else:
						emb_of_lns.append(self.embedding[word2idx[ln]])
			label_embedding.append(emb_of_lns)

		return word2idx, idx2word, label_word2idx, label_idx2word, label_embedding


	def infer(self,verbose=True):
		word2idx, idx2word, label_word2idx, label_idx2word, label_embedding = self.get_vocab_emb_mean()

		print(join(self.output_dir,'new_test_top{}'.format(self.top)))
		
		with open(join(self.output_dir,'new_test_top{}'.format(self.top))) as fin:
			max_preds = []
			mean_preds = []
			golds = []
			for line in fin:
				label, words, _ = line.strip().split('\t\t\t')
				golds.append(int(label)-1)
				words = words.split('\t')
				words = set(words) - self.stop_words
				keyword_emb = []
				for w in words:
					if w in word2idx:
						if w in self.lemmatizer and self.lemmatizer[w] in word2idx:
							mean_emb = np.mean( [ self.embedding[word2idx[w]],self.embedding[word2idx[self.lemmatizer[w]]] ], axis=0 )
							keyword_emb.append(mean_emb)
						elif w in self.lemmatizer_inv and self.lemmatizer_inv[w] in word2idx:
							mean_emb = np.mean( [ self.embedding[word2idx[w]],self.embedding[word2idx[self.lemmatizer_inv[w]]] ], axis=0 )
							keyword_emb.append(mean_emb)
						else:
							keyword_emb.append(self.embedding[word2idx[w]])
				
				if len(keyword_emb) != 0:
					keyword_emb = np.array(keyword_emb)
					max_sim = []
					mean_sim = []
					for label_i in range(len(self.label_name)):
						one_label_embedding = np.array(label_embedding[label_i])
						sim_matrix = cosine_similarity(keyword_emb,one_label_embedding)
						max_value = np.amax(np.amax(sim_matrix, axis=0))
						mean_value = np.amax(np.mean(sim_matrix, axis=0))
						max_sim.append(max_value)
						mean_sim.append(mean_value)
					max_preds.append(np.argmax(max_sim))
					mean_preds.append(np.argmax(mean_sim))
				else:
					pred = random.randint(0,self.num_class-1)
					max_preds.append(pred)
					mean_preds.append(pred)
			
			golds = np.array(golds)
			max_preds = np.array(max_preds)
			mean_preds = np.array(mean_preds)

			
			print('MAX')
			if verbose:
				print(metrics.confusion_matrix(golds, max_preds))
				print('\n')
			precision = precision_score(golds,max_preds,average='macro')
			recall = recall_score(golds,max_preds,average='macro')
			f1_macro = f1_score(golds,max_preds,average='macro')
			f1_micro = f1_score(golds,max_preds,average='micro')
			accuracy = accuracy_score(golds, max_preds)
			print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(precision, recall, f1_micro, f1_macro, accuracy))

			print('\n\n')
			print('MEAN')
			if verbose:
				print(metrics.confusion_matrix(golds, mean_preds))
				print('\n')
			precision = precision_score(golds,mean_preds,average='macro')
			recall = recall_score(golds,mean_preds,average='macro')
			f1_macro = f1_score(golds,mean_preds,average='macro')
			f1_micro = f1_score(golds,mean_preds,average='micro')
			accuracy = accuracy_score(golds, mean_preds)
			print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(precision, recall, f1_micro, f1_macro, accuracy))


	def get_pseudo_train(self):

		word2idx, idx2word, label_word2idx, label_idx2word, label_embedding = self.get_vocab_emb_mean()
		
		num_correct = 0
		num_total = 0

		pseudo = []
		golds = []

		print('{} classes'.format(len(label_word2idx)))

		with open(join(self.output_dir,'pseudo_train_{}'.format(self.threshold)),'w') as fo:
			for file_name in ['train','test']:
				with open(join(self.output_dir,'new_{}_top{}'.format(file_name, self.top))) as fin:
					for line_id, line in tqdm(enumerate(fin)):
						ground_true, words, text = line.strip().split('\t\t\t')
						words = [ w.lower() for w in words.split('\t') if w.lower() not in self.stop_words]
						words.reverse()
						words_after_lemmatize = []
						keyword_emb = []
						for w in words:
							if w in word2idx:
								if w in self.lemmatizer and self.lemmatizer[w] in word2idx:
									if self.lemmatizer[w] not in words_after_lemmatize:
										mean_emb = np.mean( [ self.embedding[word2idx[w]],self.embedding[word2idx[self.lemmatizer[w]]] ], axis=0 )
										keyword_emb.append(mean_emb)
										words_after_lemmatize.append(self.lemmatizer[w])
								elif w in self.lemmatizer_inv and self.lemmatizer_inv[w] in word2idx:
									if w not in words_after_lemmatize:
										mean_emb = np.mean( [ self.embedding[word2idx[w]],self.embedding[word2idx[self.lemmatizer_inv[w]]] ], axis=0 )
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
								sim_matrix = cosine_similarity(keyword_emb,one_label_embedding)
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
								fo.write(str(pseudo_label)+'\t\t\t'+"\t".join(words_after_lemmatize)+'\t\t\t'+text+'\n')
								num_total += 1
		
		print(metrics.confusion_matrix(golds, pseudo))
		print("{}/{} {:.4f}".format(num_correct,num_total,num_correct/num_total))


	def get_keyword_emb(self):
		word2idx, idx2word, label_word2idx, label_idx2word, label_embedding = self.get_vocab_emb_mean()
		keywords = []
		self.word_embedding = TransformerWordEmbeddings('bert-base-cased',layers='-1')
		
		with open(join(self.output_dir,'contextualized_words_{}'.format(self.threshold))) as fin:
			for line in fin:
				keywords.append(line.strip())

		with open(join(self.output_dir,'common_words_across_class_{}.txt'.format(self.threshold))) as fin:
			for line in fin:
				word = line.strip()
				if word in keywords:
					keywords.remove(word)
		
		with open(join(self.output_dir,'op_emb_bert_sup_{}'.format(self.threshold)),'w') as fo:
			for w in keywords:
				if w in word2idx:
					if w in self.lemmatizer_inv and self.lemmatizer_inv[w] in word2idx:
						mean_emb = np.mean( [ self.embedding[word2idx[w]],self.embedding[word2idx[self.lemmatizer_inv[w]]] ], axis=0 )
						fo.write(w+' '+' '.join(list(map(str,mean_emb)))+'\n')
					else:
						fo.write(w+' '+' '.join(list(map(str,self.embedding[word2idx[w]])))+'\n')
				else:
					sentence = Sentence( w )
					self.word_embedding.embed(sentence)
					for tk in sentence:
						word = tk.text
						word = word.lower()	
						emb = tk.embedding.detach().cpu().numpy()
						emb = list(map(str,emb))
						fo.write(w+' '+' '.join(emb)+'\n')


	def get_contextualized_words(self):
		word2idx, idx2word, label_word2idx, label_idx2word, label_embedding = self.get_vocab_emb_mean()
		keywords = Counter()
		num_keyword_token = 0

		common_words = set()
		with open(join(self.output_dir,'common_words_across_class_{}.txt'.format(self.threshold))) as fin:
			for line in fin:
				common_words.add(line.strip())

		with open(join(self.output_dir,'contextualized_words_{}'.format(self.threshold)),'w') as fo:
			with open(join(self.output_dir,'pseudo_train_{}'.format(self.threshold))) as fin:
				for line in fin:
					words = set(line.strip().split('\t\t\t')[1].split('\t'))
					words = words - self.stop_words
					for w in words:
						w = w.lower()
						if len(w) > 1:
							if w in word2idx and w in self.lemmatizer and self.lemmatizer[w] in word2idx:
								keywords.update([self.lemmatizer[w]])
							else:
								keywords.update([w])
					num_keyword_token += len(words)
			
			for (w,f) in keywords.most_common(len(keywords)):
				if w not in common_words:
					if f > 2:
						print("{:15} {:.6f} {:5}".format(w,f/num_keyword_token,f))	
					if f/num_keyword_token >= 0.00005:
						fo.write(w+'\n')


	def get_common_words(self):
		
		label2idx = { label.split('/')[0]:i+1 for i, label in enumerate(self.label_name)}
		idx2label = { i+1:label.split('/')[0] for i, label in enumerate(self.label_name)}
		
		total_num_words = {label.split('/')[0]:0 for label in self.label_name}
		total_num_docs = {label.split('/')[0]:0 for label in self.label_name}
		all_words = {label.split('/')[0]:Counter() for label in self.label_name}
		all_words_dist = {}

		word_vocab = Counter()
		
		with open(join(self.output_dir,'pseudo_train_{}'.format(self.threshold))) as fin:
			for line in fin:
				label, tuples, text = line.strip().split('\t\t\t')
				tuples = [ w.lower() for w in tuples.split('\t')]
				total_num_docs[idx2label[int(label)]] += 1
				total_num_words[idx2label[int(label)]] += len(tuples)
				all_words[idx2label[int(label)]].update(tuples)
				word_vocab.update(tuples)

		for ii, label in enumerate(self.label_name):
			label = label.split('/')[0]
			for (w,f) in all_words[label].most_common(len(all_words[label])):
				if w not in all_words_dist:
					all_words_dist[w] = {l.split('/')[0]:0 for l in self.label_name}
				all_words_dist[w][label] = f/total_num_words[label]*1e5


		print(total_num_words)
		total_num_words_all_labels = 0
		for l in total_num_words:
			total_num_words_all_labels += total_num_words[l]
		# for (w,f) in word_vocab.most_common(len(word_vocab)):
		# 	print(w,f)

		with open(join(self.output_dir,'common_words_across_class_{}.txt'.format(self.threshold)),'w') as fo:
			for (word,freq) in word_vocab.most_common(len(word_vocab)):
				freq_list = [ all_words_dist[word][l] for l in all_words_dist[word] ]
				freq_list = sorted(freq_list)
				if freq_list[-2] != 0:
					first_sec_diff = freq_list[-1] / freq_list[-2]
				else:
					first_sec_diff = 10000
				if freq_list[-3] != 0:
					sec_third_diff = freq_list[-1] / freq_list[-3]
				else:
					sec_third_diff = 10000

				print("{:15} {:.6f}".format(word,freq/total_num_words_all_labels),end=' ')
				for l in all_words_dist[word]:
					print("{:10} {:8.2f}".format(l, all_words_dist[word][l]),end='         ')
				print("{:8.2f} {:8.2f}".format(first_sec_diff,sec_third_diff),end='\n\n')


				# if first_sec_diff <= 2.0:
				# 	if sec_third_diff >= 50.0:
				# 		pass
				# 	else:
				# 		fo.write(word+'\n')
				
				# if first_sec_diff > 2.0 and first_sec_diff <= 10.0 and sec_third_diff <= 25.0:
				# 	fo.write(word+'\n')


				if first_sec_diff <= 2.0:
					fo.write(word+'\n')


	def get_corpus_for_VWS(self):

		des_dir = self.output_dir
		data_dir = self.output_dir

		keywords = set()
		with open(join(data_dir,'contextualized_words_{}'.format(self.threshold))) as fin:
			for line in fin:
				keywords.add(line.strip())

		with open(join(data_dir,'common_words_across_class_{}.txt'.format(self.threshold))) as fin:
			for line in fin:
				word = line.strip()
				if word in keywords:
					keywords.remove(word)

		with open(join(des_dir, 'train'),'w') as fo:
			with open(join(data_dir,'new_train_top{}'.format(self.top))) as fin:
				for line in fin:
					label, tuples, text = line.strip().split('\t\t\t')
					new_tuples = set()
					for w in tuples.split():
						if w in keywords:
							new_tuples.add(w)
						elif w in self.lemmatizer and self.lemmatizer[w] in keywords:
							new_tuples.add(self.lemmatizer[w])
					fo.write(label+'\t\t\t'+'\t'.join(new_tuples)+'\t\t\t'+text+'\n')

		with open(join(des_dir, 'dev'),'w') as fo:
			with open(join(data_dir,'new_dev_top{}'.format(self.top))) as fin:
				for line in fin:
					label, tuples, text = line.strip().split('\t\t\t')
					new_tuples = set()
					for w in tuples.split():
						if w in keywords:
							new_tuples.add(w)
						elif w in self.lemmatizer and self.lemmatizer[w] in keywords:
							new_tuples.add(self.lemmatizer[w])
					fo.write(label+'\t\t\t'+'\t'.join(new_tuples)+'\t\t\t'+text+'\n')

		with open(join(des_dir, 'test'),'w') as fo:
			with open(join(data_dir,'new_test_top{}'.format(self.top))) as fin:
				for line in fin:
					label, tuples, text = line.strip().split('\t\t\t')
					new_tuples = set()
					for w in tuples.split():
						if w in keywords:
							new_tuples.add(w)
						elif w in self.lemmatizer and self.lemmatizer[w] in keywords:
							new_tuples.add(self.lemmatizer[w])
					fo.write(label+'\t\t\t'+'\t'.join(new_tuples)+'\t\t\t'+text+'\n')


	def get_freq_words(self, min_cut=3):
		tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

		if not os.path.exists(join(self.output_dir, 'freq_words.txt')):
			with open(join(self.output_dir, 'freq_words.txt'), 'w') as fo:
				with open(join(self.output_dir, 'new_train_top{}'.format(self.top))) as fin:
					freq_cnt = Counter()
					for line_id, line in tqdm(enumerate(fin)):
						label, words, text = line.strip().split('\t\t\t')

						tokenized_text = tokenizer.tokenize(text)

						tokens = []
						for word in tokenized_text:
							word = word.lower()
							if word[-1] == '.' or word[-1] == '\'':
								word = word[:-1]
							tokens.append(word)

						freq_cnt.update(tokens)

				with open(join(self.output_dir, 'new_test_top{}'.format(self.top))) as fin:
					freq_cnt = Counter()
					for line_id, line in tqdm(enumerate(fin)):
						label, words, text = line.strip().split('\t\t\t')

						tokenized_text = tokenizer.tokenize(text)

						tokens = []
						for word in tokenized_text:
							word = word.lower()
							if word[-1] == '.' or word[-1] == '\'':
								word = word[:-1]
							tokens.append(word)

						freq_cnt.update(tokens)

				with open(join(self.output_dir, 'new_dev_top{}'.format(self.top))) as fin:
					freq_cnt = Counter()
					for line_id, line in tqdm(enumerate(fin)):
						label, words, text = line.strip().split('\t\t\t')

						tokenized_text = tokenizer.tokenize(text)

						tokens = []
						for word in tokenized_text:
							word = word.lower()
							if word[-1] == '.' or word[-1] == '\'':
								word = word[:-1]
							tokens.append(word)

						freq_cnt.update(tokens)

				punc = set(string.punctuation)
				punc.add('..')
				punc.add('...')

				self.freq_words = set()
				for (w, f) in freq_cnt.most_common(len(freq_cnt)):
					if w in self.stop_words or "/" in w or len(w) == 0:
						continue
					if len(set(w) & punc) > 0:
						continue
					if w[:2] == '##':
						continue
					if len(set(w) & set('0123456789')) > 0:
						continue
					if f >= min_cut:
						self.freq_words.add(w)
						print("{:10} {}".format(w, f))
				self.freq_words = self.freq_words - punc
				for w in self.freq_words:
					fo.write(w + '\n')
				print('{} frequent words'.format(len(self.freq_words)))
		else:
			self.freq_words = set()
			with open(os.path.join(self.output_dir, 'freq_words.txt')) as fin:
				for line in fin:
					self.freq_words.add(line.strip())

	def get_freq_words_on_masked_words(self, min_cut=5):
		if not os.path.exists(join(self.output_dir, 'freq_words_on_lm_generated_words.txt')):
			with open(join(self.output_dir, 'freq_words_on_lm_generated_words.txt'), 'w') as fo:
				for file_name in ['train', 'dev', 'test']:
					with open(join(self.output_dir, 'new_{}_top{}'.format(file_name, self.top))) as fin:
						freq_cnt = Counter()
						for line_id, line in enumerate(fin):
							label, words, text = line.strip().split('\t\t\t')
							words = words.split('\t')
							tokens = []
							for word in words:
								word = word.lower()
								if word[-1] == '.' or word[-1] == '\'':
									word = word[:-1]
								tokens.append(word)

							freq_cnt.update(tokens)

				punc = set(string.punctuation)
				punc.add('..')
				punc.add('...')

				self.freq_words = set()
				for (w, f) in freq_cnt.most_common(len(freq_cnt)):
					if w in self.stop_words or "/" in w or len(w) == 0:
						continue
					if len(set(w) & punc) > 0:
						continue
					if w[:2] == '##':
						continue
					if len(set(w) & set('0123456789')) > 0:
						continue
					if f >= min_cut:
						self.freq_words.add(w)
				self.freq_words = self.freq_words - punc
				for w in self.freq_words:
					fo.write(w + '\n')
				print('{} frequent words'.format(len(self.freq_words)))
		else:
			self.freq_words = set()
			with open(os.path.join(self.output_dir, 'freq_words_on_lm_generated_words.txt')) as fin:
				for line in fin:
					self.freq_words.add(line.strip())

	def diff_freq_words(self):
		text_freq = set()
		gen_lm_freq = set()
		with open(join(self.output_dir,'freq_words.txt')) as fin:
			for line in fin:
				text_freq.add(line.strip())
		with open(join(self.output_dir,'freq_words_on_lm_generated_words.txt')) as fin:
			for line in fin:
				gen_lm_freq.add(line.strip())

		print('text has, lm-generated do not')
		print(text_freq-gen_lm_freq)

		print('lm-generated has, text do not')
		print(gen_lm_freq-text_freq)

	def static_rep(self, seg_no=0, num_seg=1):
		self.get_freq_words()
		self.word_embedding = TransformerWordEmbeddings('bert-base-cased', layers='-1')

		if not os.path.exists(self.dump_dir):
			os.makedirs(self.dump_dir)

		file_names = ['dev']
		for file_name in file_names:
			with open(os.path.join(self.data_dir, file_name), "r", encoding="iso-8859-1") as fin:
				for line_id, line in enumerate(fin):
					pass
			num_samples = line_id + 1
			batch_size = math.ceil(num_samples / num_seg)
			print('dump dir {}'.format(self.dump_dir))
			print('{} samples {} batches'.format(num_samples, num_seg))
			with open(os.path.join(self.data_dir, file_name), "r", encoding="iso-8859-1") as fin:
				for line_id, line in tqdm(enumerate(fin)):
					if line_id >= batch_size * (seg_no) and line_id < batch_size * (seg_no + 1):
						ground_true, text = line.strip().split('\t\t\t')
						if len(text.split()) > 64:
							num_sents = math.ceil(len(text.split()) / 64)
							for sent_id in range(num_sents):
								small_text = text.split()[sent_id * 64:(sent_id + 1) * 64]
								sentence = Sentence(" ".join(small_text))
								self.word_embedding.embed(sentence)
								for tk in sentence:
									word = tk.text
									word = word.lower()
									if word in self.freq_words:
										emb = tk.embedding.detach().cpu().numpy()
										emb = list(map(str, emb))
										with open(os.path.join(self.dump_dir, word), 'a+') as fo:
											fo.write(" ".join(emb) + '\n')
						else:
							sentence = Sentence(text)
							self.word_embedding.embed(sentence)
							for tk in sentence:
								word = tk.text
								word = word.lower()
								if word in self.freq_words:
									emb = tk.embedding.detach().cpu().numpy()
									emb = list(map(str, emb))
									with open(os.path.join(self.dump_dir, word), 'a+') as fo:
										fo.write(" ".join(emb) + '\n')

	def mean_static_rep(self):
		word2idx = {}
		idx2word = {}
		embedding = []

		files = [f for f in listdir(self.dump_dir) if isfile(join(self.dump_dir, f))]
		word_cnt = 0
		for ii, fn in tqdm(enumerate(files)):
			path = join(self.dump_dir,fn)
			vec = []
			with open(path) as fin:
				for line in fin:
					emb = list(map(float,line.strip().split()))
					vec.append(emb)
			if len(vec) > 0:
				word2idx[fn] = word_cnt
				idx2word[word_cnt] = fn
				word_cnt += 1
				embedding.append(np.mean(vec,axis=0))

		embedding = np.array(embedding, dtype=np.float32)
		embedding = embedding / (np.linalg.norm(embedding, ord=2, axis=1, keepdims=True))
		with open(os.path.join(self.data_dir,'vocab_emb_mean'),'w') as fo:
			for i in range(len(embedding)):
				fo.write(idx2word[i]+' '+' '.join(list(map(str,embedding[i])))+'\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--file_name', type=str, default='train',
						help='file name') 
	parser.add_argument('--data_dir', type=str, default=0,
						help='data directory') 
	args = parser.parse_args()
	label_name = ['politics','sports','business','technology']
	data_dir = 'baseline_data/agnews'
	output_dir = 'output/agnews_bert'
	vws_dir = ''
	esa_dir = ''
	Masked_Bert(args,'train', label_name, data_dir, output_dir, vws_dir, esa_dir)

