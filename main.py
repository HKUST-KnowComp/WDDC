import os
from os.path import join
import random
import numpy as np
from tqdm import tqdm
import time
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Model
from util.batch_gen import VWSDataset
from util.load import load_embedding

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, roc_curve, auc
from sklearn import metrics


def collate_fn(x):
    bz = len(x)
    num_item = len(x[0])
    
    new_x = [ [] for _ in range(num_item) ] 
    for i in range(bz):
        for j in range(num_item):
            new_x[j].append(x[i][j])
    x = []
    for i, item in enumerate(new_x):
        if i == 0:
            item = torch.stack(item,dim=0)
        else:
            item = torch.tensor(item)
        x.append(item)

    return x


def evaluate(args, name, asp_word2idx, model, device, dataloader, verbose): # file_name = join(args.data_dir,args.dev)
    eval_chunk_size = args.eval_chunk_size
    goldens = []
    preds = []

    file_name = join(args.data_dir,getattr(args, name))
    if dataloader is None:
        num_sample = get_num_sample(file_name)
        num_chunk = math.ceil(num_sample / eval_chunk_size)
        random_idx = np.arange(num_sample)
        
        for chunk in range(num_chunk):
            chunk_random_idx = random_idx[chunk*eval_chunk_size:(chunk+1)*eval_chunk_size]
            data_set = VWSDataset(args, name, asp_word2idx, chunk_random_idx, False)
            dataloader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            for batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight, batch_doc_id in dataloader:
                batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight, batch_doc_id = \
                batch_x.to(device), batch_y.to(device), batch_senti.to(device), batch_neg_senti.to(device), batch_weight.to(device), batch_doc_id.to(device)
                model.eval()
                loss = model(batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight, batch_doc_id, False)
                pred = model.pred.detach()
                golden = batch_y
                goldens.append(golden.cpu().numpy())
                preds.append(pred.cpu().numpy())
            del data_set
            del dataloader
    else:  
        for batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight, batch_doc_id in dataloader:
            batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight, batch_doc_id = \
            batch_x.to(device), batch_y.to(device), batch_senti.to(device), batch_neg_senti.to(device), batch_weight.to(device), batch_doc_id.to(device)
            model.eval()
            loss = model(batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight, batch_doc_id, False)
            pred = model.pred.detach()
            golden = batch_y
            goldens.append(golden.cpu().numpy())
            preds.append(pred.cpu().numpy())
            
    golden = np.concatenate(goldens, axis=0)
    pred = np.concatenate(preds, axis=0)



    if verbose:
        print(metrics.confusion_matrix(golden, pred))
        print('\n')
    
    precision = precision_score(golden,pred,average='macro')
    recall = recall_score(golden,pred,average='macro')
    f1_macro = f1_score(golden,pred,average='macro')
    f1_micro = f1_score(golden,pred,average='micro')
    accuracy = accuracy_score(golden, pred)

    return precision, recall, f1_macro, f1_micro, accuracy, golden, pred


def result_analysis(args,name,pred,golden):
    with open(join(args.data_dir,'label_to_index.json')) as fin:
        label_to_index = json.load(fin)

    index_to_label = { label_to_index[l]:l for l in label_to_index}
    extracted_keyword = []
    text = []
    with open(join(args.data_dir,getattr(args,name))) as fin:
        for line in fin:
            extracted_words = line.strip().split('\t\t\t')[1]
            extracted_words = set(extracted_words.split('\t'))
            extracted_keyword.append(" , ".join(list(extracted_words)))
            text.append(line.strip().split('\t\t\t')[-1])
    
    with open(join(args.log_dir,name+'_new_analysis'),'w') as fo:
        fo.write('{:10} {:10} extracted_keywords original text\n\n'.format('prediction','ground_true'))
        cnt = 0
        for p, g in zip(pred,golden):
            if p != g and len(extracted_keyword) > 0:
                fo.write("{:6} {:2} {:10} {:2} {:10} * {} * TEXT_START {}\n\n".format(cnt, p, index_to_label[p], g, index_to_label[g],extracted_keyword[cnt],text[cnt]))
            cnt += 1


def get_num_sample(file_name):
    with open(file_name, encoding="utf-8") as fin:
        for line_id, line in enumerate(fin):
            pass
    return line_id + 1


def train(args):

    time_stamp = time.time()
    asp_word2idx, keyword_emb = load_embedding(args, os.path.join(args.data_dir,args.keyword_emb), True)
    
    asp_idx2word = { asp_word2idx[word]:word for word in asp_word2idx }
    asp_idx2word[0] = 'NULL'
    print("Building Batches")

    need_neg_senti = args.unsupervised
    
    num_train_sample = get_num_sample(join(args.data_dir,args.train))
    num_dev_sample = get_num_sample(join(args.data_dir,args.dev))
    num_test_sample = get_num_sample(join(args.data_dir,args.test))

    if num_dev_sample > args.eval_chunk_size:
        dev_set, dev_dataloader = None, None
    else:
        dev_set = VWSDataset(args, 'dev', asp_word2idx)
        dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if num_test_sample > args.eval_chunk_size:
        test_set, test_dataloader = None, None
    else:
        test_set = VWSDataset(args, 'test', asp_word2idx)
        test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    num_chunk = args.num_chunk
    if num_chunk == 1:
        train_set = VWSDataset(args, 'train', asp_word2idx, need_neg_senti=need_neg_senti)
        train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        # num_train_sample = len(train_dataloader)
    device = torch.device("cuda" if args.use_cuda else "cpu")
    args.device = device
    print('device',args.device)
    model = Model(args, torch.tensor(keyword_emb))
    model.to(device)
    print('Trainable Parameters')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    if args.unsupervised:
        try:
            with open(os.path.join(args.save_dir, "record"),'r') as fin:
                for line in fin:
                    best_step = line.strip()
            file_name = os.path.join(args.save_dir, "model_pretrain_{}.pt".format(best_step))
        except:
            with open(os.path.join(args.default_save_dir, "record"),'r') as fin:
                for line in fin:
                    best_step = line.strip()
            file_name = os.path.join(args.default_save_dir, "model_pretrain_{}.pt".format(best_step))
        
        pretrained_dict = torch.load(file_name)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v.to(device) for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)


        # dev_precision, dev_recall, dev_f1_macro, dev_f1_micro, dev_accuracy, dev_golden, dev_pred = evaluate(args, 'dev', asp_word2idx, model, device, dev_dataloader, True)
        # test_precision, test_recall, test_f1_macro, test_f1_micro, test_accuracy, test_golden, test_pred = evaluate(args, 'test', asp_word2idx, model, device, test_dataloader, True)


        # print('Pretrain Result')
        
        # print('Dev\n')
        # print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(dev_precision, dev_recall, dev_f1_micro, dev_f1_macro, dev_accuracy))

        # result_analysis(args,'dev',dev_pred,dev_golden)
        # print('Test\n')
        # print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(test_precision, test_recall, test_f1_micro, test_f1_macro,test_accuracy)) 
        # raise SystemExit

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)


    best_dev_precision, best_dev_recall, best_dev_f1_macro, best_dev_f1_micro, best_dev_accuracy = 0., 0., 0., 0., 0.
    best_test_precision, best_test_recall, best_test_f1_macro, best_test_f1_micro, best_test_accuracy = 0., 0., 0., 0., 0.
    best_epoch = 0
    best_global_step = 0
    global_step = 0
 
    
    chunk_size = math.ceil(num_train_sample / num_chunk)
    for epoch in range(1, args.num_epochs + 1):
        random_idx = np.random.permutation(num_train_sample)
        for chunk in range(num_chunk):
            if num_chunk > 1:
                chunk_random_idx = random_idx[chunk*chunk_size:(chunk+1)*chunk_size]
                train_set = VWSDataset(args, 'train', asp_word2idx, chunk_random_idx, need_neg_senti=need_neg_senti)
                train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
                
            for batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight, batch_doc_id in train_dataloader:
                # model.train()
                global_step += 1
                batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight, batch_doc_id = \
                batch_x.to(device), batch_y.to(device), batch_senti.to(device), batch_neg_senti.to(device), batch_weight.to(device), batch_doc_id.to(device)

                optimizer.zero_grad()
                loss = model(batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight, batch_doc_id)

                reg_loss = None
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if reg_loss is None:
                            reg_loss = 0.5 * torch.sum(param**2)
                        else:
                            reg_loss = reg_loss + 0.5 * torch.sum(param**2)

                total_loss = loss + args.l2_reg*reg_loss

                total_loss.backward()

                optimizer.step()

                if global_step % args.eval_period == 0:
                    print('Start Evaluating ...')
                    dev_precision, dev_recall, dev_f1_macro, dev_f1_micro, dev_accuracy, dev_golden, dev_pred = evaluate(args, 'dev', asp_word2idx, model, device, dev_dataloader, True)
                    if num_test_sample <= args.eval_chunk_size:
                        test_precision, test_recall, test_f1_macro, test_f1_micro, test_accuracy, test_golden, test_pred = evaluate(args, 'test', asp_word2idx, model, device, dev_dataloader, True)
            
                    if dev_f1_macro > best_dev_f1_macro: #if dev_f1_micro > best_dev_f1_micro:
                        best_epoch = epoch
                        best_global_step = global_step
                        best_dev_precision, best_dev_recall, best_dev_f1_macro, best_dev_f1_micro, best_dev_accuracy = dev_precision, dev_recall, dev_f1_macro, dev_f1_micro, dev_accuracy
                        if num_test_sample <= args.eval_chunk_size:
                            best_test_precision, best_test_recall, best_test_f1_macro, best_test_f1_micro, best_test_accuracy = test_precision, test_recall, test_f1_macro, test_f1_micro, test_accuracy 
                        # result_analysis(args,'dev',dev_pred,dev_golden)

                        if not args.unsupervised:
                            if not os.path.exists(args.save_dir):
                                os.makedirs(args.save_dir)
                            filename = os.path.join(args.save_dir, "model_pretrain_{}.pt".format(global_step))
                            torch.save(model.state_dict(), filename)
                            with open(os.path.join(args.save_dir, "record"),'w') as fo:
                                fo.write("{}\n".format(global_step))
                        else:
                            if not os.path.exists(args.save_dir):
                                os.makedirs(args.save_dir)
                            filename = os.path.join(args.save_dir, "model_vws_{}.pt".format(time_stamp))
                            torch.save(model.state_dict(), filename)
                            

                            # asp_emb = model.keyword_emb.weight.detach().cpu().numpy()
                            # with open(join(args.data_dir,'op_emb_after_vws'),'w') as fo:
                            #     for idx in asp_idx2word.keys():
                            #         if idx != 0:
                            #             fo.write( str(asp_idx2word[idx]) + ' ' ) 
                            #             fo.write( " ".join(list(map(str,asp_emb[idx]))) +'\n')

                    if args.verbose:
                        print('alpha {:.2f} beta {:.2f} gamma_positive {:.2f} gamma_negative {:.2f}'.format(args.alpha, args.beta, args.gamma_positive, args.gamma_negative))
                        print('dataset {} train {} dev {} test {}'.format(args.data_dir, args.train, args.dev, args.test))
                        print('unsupervised {}'.format(args.unsupervised))
                        print('language model name {}'.format(args.lm_name))
                        print('regularization type {}'.format(args.reg_type))
                        print('num_filters {} max_len {} emb_dim {} emb_trainable {}'.format(args.num_filters, args.max_len, args.emb_dim, args.emb_trainable))
                        print('score_scale {} num_senti {} num_neg {}'.format(args.score_scale, args.num_senti, args.num_neg))
                        print('num_epochs {}'.format(args.num_epochs))

                        print('Epoch {} Step {}\n'.format(epoch, global_step))
                        print('Dev\n')
                        print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(dev_precision, dev_recall, dev_f1_micro, dev_f1_macro, dev_accuracy))

                        if num_test_sample <= args.eval_chunk_size:
                            print('Test\n')
                            print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(test_precision, test_recall, test_f1_micro, test_f1_macro,test_accuracy)) 
            
            if num_chunk > 1:
                del train_set
                del train_dataloader
        
        if epoch >= 10:
            lr=args.lr*args.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    if args.unsupervised:
        filename = os.path.join(args.save_dir, "model_vws_{}.pt".format(time_stamp))
    else:
        filename = os.path.join(args.save_dir, "model_pretrain_{}.pt".format(best_global_step))
    
    model.load_state_dict(torch.load(filename))
    model.eval()
    
    dev_precision, dev_recall, dev_f1_macro, dev_f1_micro, dev_accuracy, dev_golden, dev_pred = evaluate(args, 'dev', asp_word2idx, model, device, dev_dataloader, True)
    test_precision, test_recall, test_f1_macro, test_f1_micro, test_accuracy, test_golden, test_pred = evaluate(args, 'test', asp_word2idx, model, device, test_dataloader, True)

    best_dev_precision, best_dev_recall, best_dev_f1_macro, best_dev_f1_micro, best_dev_accuracy = dev_precision, dev_recall, dev_f1_macro, dev_f1_micro, dev_accuracy
    best_test_precision, best_test_recall, best_test_f1_macro, best_test_f1_micro, best_test_accuracy = test_precision, test_recall, test_f1_macro, test_f1_micro, test_accuracy 

    print('alpha {:.2f} beta {:.2f} gamma_positive {:.2f} gamma_negative {:.2f}'.format(args.alpha, args.beta, args.gamma_positive, args.gamma_negative))
    print('dataset {} train {} dev {} test {}'.format(args.data_dir, args.train, args.dev, args.test))
    print('unsupervised {}'.format(args.unsupervised))
    print('language model name {}'.format(args.lm_name))
    print('regularization type {}'.format(args.reg_type))
    print('num_filters {} max_len {} emb_dim {} emb_trainable {}'.format(args.num_filters, args.max_len, args.emb_dim, args.emb_trainable))
    print('score_scale {} num_senti {} num_neg {}'.format(args.score_scale, args.num_senti, args.num_neg))
    print('num_epochs {}'.format(args.num_epochs))

    print('Best Epoch at {:2}'.format(best_epoch))
    print('Best Global Step at {:2}'.format(best_global_step))
    
    print('Dev\n')
    print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(best_dev_precision, best_dev_recall, best_dev_f1_micro, best_dev_f1_macro, best_dev_accuracy))

    
    print('Test\n')
    print('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(best_test_precision, best_test_recall, best_test_f1_micro, best_test_f1_macro, best_test_accuracy))

    print('='*100 + '\n')


    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        if not os.path.exists(os.path.join(args.log_dir,args.log_file)):
            with open(os.path.join(args.log_dir,args.log_file),'w') as fo:
                pass
            

    with open(os.path.join(args.log_dir,args.log_file),'a') as fo:
        fo.write('alpha {:.2f} beta {:.2f} gamma_positive {:.2f} gamma_negative {:.2f}\n'.format(args.alpha, args.beta, args.gamma_positive, args.gamma_negative))
        fo.write('dataset {} train {} dev {} test {}\n'.format(args.data_dir, args.train, args.dev, args.test))
        fo.write('unsupervised {}\n'.format(args.unsupervised))
        fo.write('language model name {}\n'.format(args.lm_name))
        fo.write('regularization type {}\n'.format(args.reg_type))
        fo.write('num_filters {} max_len {} emb_dim {} emb_trainable {}\n'.format(args.num_filters, args.max_len, args.emb_dim, args.emb_trainable))
        fo.write('score_scale {} num_senti {} num_neg {}\n'.format(args.score_scale, args.num_senti, args.num_neg))
        fo.write('num_epochs {}\n'.format(args.num_epochs))
        
        fo.write('Best Epoch at {}\n'.format(best_epoch))
        fo.write('Best Global Step at {:2}\n'.format(best_global_step))
        fo.write('Dev\n')
        fo.write('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(best_dev_precision, best_dev_recall, best_dev_f1_micro, best_dev_f1_macro, best_dev_accuracy))

        fo.write('Test\n')
        fo.write('Precision {:.5f}, Recall {:.5f}, Micro F1 {:.5f}, Macro F1 {:.5f}, ACC {:.5f} \n'.format(best_test_precision, best_test_recall, best_test_f1_micro, best_test_f1_macro, best_test_accuracy))

        fo.write('='*100 + '\n')
    
    

