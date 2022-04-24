import os
import argparse
from main import train

parser = argparse.ArgumentParser(description='VWS')

parser.add_argument('--unsupervised', type=bool, default=False,
                    help='whether train model unsupervisedly')
parser.add_argument('--lm_name', type=str, default="bert-base-uncased",
                    help='training set')

parser.add_argument('--train', type=str, default="pseudo_train",
                    help='training set') # pseudo_train
parser.add_argument('--dev', type=str, default="dev",
                    help='development set')
parser.add_argument('--test', type=str, default="test",
                    help='test set')

parser.add_argument('--num_chunk', type=int, default=2,
                    help='number of chunk')
parser.add_argument('--eval_chunk_size', type=int, default=5000,
                    help='size of chunk when evaluate')

parser.add_argument('--save_dir', type=str, default="../saved_model/20news_bert/",
                    help='path to model saving directory')
parser.add_argument('--data_dir', type=str, default="../data/20news_bert",
                    help='path to data directory')
parser.add_argument('--esa_dir', type=str, default="../data/20news_bert",
                    help='path to data directory')
parser.add_argument('--log_dir', type=str, default="../log/20news_bert",
                    help='path to log directory')
parser.add_argument('--log_file', type=str, default="pretrain_hyper_params.txt",
                    help='file to record f1 score')

parser.add_argument('--keyword_emb', type=str, default="op_emb",
                    help='path to pre-trained keyword embedding')
parser.add_argument('--reg_type', type=str, default='word',
                    help='regularization type')

parser.add_argument('--alpha', type=float, default=0.0,
                    help='coefficient of H(q(c|x))')
parser.add_argument('--beta', type=float, default=0.0,
                    help='coefficient of regularization')

parser.add_argument('--gamma_positive', type=float, default=0.0,
                    help='similarity threshold')
parser.add_argument('--gamma_negative', type=float, default=0.0,
                    help='dissimilarity threshold')
parser.add_argument('--score_scale', type=int, default=6,
                    help='number of classes of labels')
parser.add_argument('--num_senti', type=int, default=5,
                    help='number of keywords')
parser.add_argument('--num_neg', type=int, default=50,
                    help='number of negative samples')
parser.add_argument('--min_count', type=int, default=1,
                    help='words that less than min_count will be filtered out in lexicon learner')

parser.add_argument('--max_len', type=int, default=128,
                    help='maximum document length in CNN')
parser.add_argument('--num_filters', type=int, default=100,
                    help='number of filters in CNN')
parser.add_argument('--emb_dim', type=int, default=768,
                    help='dimension of embedding in CNN')
parser.add_argument('--emb_trainable', type=bool, default=False,
                    help='whether word embeddings is trainable')

parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--num_epochs', type=int, default=5,
                    help='maximum number of epochs')
parser.add_argument('--eval_period', type=int, default=100,
                    help='evaluate on dev every period')

parser.add_argument('--keep_prob', type=float, default=0.7,
                    help='keep probability in dropout')
parser.add_argument('--lr', type=float, default=1,
                    help='learning rate for Adadelta')
parser.add_argument('--lr_decay', type=float, default=0.95,
                    help='learning rate decay')
parser.add_argument('--l2_reg', type=float, default=0.0001,
                    help='l2 regularization of weights')

parser.add_argument('--verbose', type=bool, default=True,
                    help='whether print details')
parser.add_argument('--use_cuda', type=bool, default=True,
                    help='whether use cuda')

args = parser.parse_args()

print('alpha {:.2f} beta {:.2f} gamma_positive {:.2f} gamma_negative {:.2f}'.format(args.alpha, args.beta, args.gamma_positive, args.gamma_negative))
print('dataset {} train {} dev {} test {}'.format(args.data_dir, args.train, args.dev, args.test))
print('unsupervised {}'.format(args.unsupervised))
print('language model name {}'.format(args.lm_name))
# print('regularization type {}'.format(args.reg_type))
print('num_filters {} max_len {} emb_dim {} emb_trainable {}'.format(args.num_filters, args.max_len, args.emb_dim, args.emb_trainable))
print('score_scale {} num_senti {} num_neg {}'.format(args.score_scale, args.num_senti, args.num_neg))
print('num_epochs {}'.format(args.num_epochs))  
train(args)
