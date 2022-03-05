import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from os.path import join


class Model(nn.Module):
    def __init__(self, args, asp_word_mat):
        super(Model, self).__init__()
        self.args = args
        self.bert_emb_dim = 768


        self.alpha = torch.nn.Parameter(torch.tensor(args.alpha), requires_grad=False)
        self.beta = torch.nn.Parameter(torch.tensor(args.beta), requires_grad=False)

        self.loss, self.total_loss = None, None
        self.vae_loss, self.entropy_term_loss, self.opinion_reg_loss = None, None, None

        self.filter_sizes = [2, 3, 4, 5]
        self.num_filters = args.num_filters
        self.emb_dim = args.emb_dim
        self.score_scale = args.score_scale

        if args.unsupervised:
            self.keyword_emb = nn.Embedding(asp_word_mat.size(0), asp_word_mat.size(1))
            self.keyword_emb.weight = nn.Parameter(asp_word_mat)
            self.keyword_emb.weight.requires_grad = True

            W_weight = torch.empty(self.emb_dim, self.score_scale)
            torch.nn.init.xavier_uniform_(W_weight)
            self.W = torch.nn.Parameter(W_weight, requires_grad=True)

            self.scale = torch.nn.Parameter(torch.tensor(0.1), requires_grad=True)

        self.cnn_convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (f_size, self.bert_emb_dim), padding=(f_size - 1, 0)) for f_size in
             self.filter_sizes])
        self.cnn_fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.score_scale)
        self.loss_func = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=1 - args.keep_prob)
        self.reg_type = args.reg_type
        if self.reg_type == 'doc':
            self.doc_esa_sim = torch.tensor(np.load(open(join(self.args.esa_dir, self.args.esa_file), 'rb'))).to(
                self.args.device)

    def dropouted_emb(self, embed, words, dropout=0.1):
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
            embed.weight) / (1 - dropout)
        masked_embed_weight = mask * embed.weight
        X = torch.nn.functional.embedding(words, masked_embed_weight, embed.padding_idx,
                                          embed.max_norm, embed.norm_type,
                                          embed.scale_grad_by_freq, embed.sparse
                                          )
        return X

    def distance(self, probs):
        r_p_matrix = torch.unsqueeze(probs, dim=1).repeat(1, probs.size(0), 1)
        r_p_matrix_T = r_p_matrix.permute(1, 0, 2)

        r_p_matrix_distance = torch.pow((r_p_matrix - r_p_matrix_T), 2)

        epsilon = 1e-12 * torch.ones(r_p_matrix_distance.size(0), r_p_matrix_distance.size(1))
        r_p_distance = torch.sqrt(torch.max(torch.sum(r_p_matrix_distance, 2), epsilon.to(self.args.device)))

        return r_p_distance

    def block_wise_operator(self, mat, batch_size, num_senti):
        mat = torch.reshape(mat, (batch_size, num_senti, -1))
        mat = mat.permute(0, 2, 1)
        mat = torch.reshape(mat, (batch_size, batch_size, num_senti, num_senti))
        mat = mat.permute(0, 1, 3, 2)
        return mat

    def get_emb_similarity(self, senti_emb):  # cosine_mean
        args = self.args
        batch_size = senti_emb.size(0)
        num_senti = senti_emb.size(1)

        # negation_ = torch.unsqueeze(negation, dim=2).repeat(1, 1, senti_emb.size(-1))
        # senti_emb = senti_emb*negation_

        senti_emb = torch.reshape(senti_emb, (-1, senti_emb.size(-1)))
        senti_emb = F.normalize(senti_emb, dim=1, p=2)

        bn_x_bn_cos_sim = torch.matmul(senti_emb, senti_emb.permute(1, 0))

        bn_x_bn_cos_sim = self.block_wise_operator(bn_x_bn_cos_sim, batch_size, num_senti)

        b_x_b_mean = torch.mean(bn_x_bn_cos_sim, dim=(2, 3))

        mask_positive = torch.gt(b_x_b_mean, args.gamma_positive).type(torch.float32)
        mask_negative = torch.lt(b_x_b_mean, args.gamma_negative).type(torch.float32)

        mask = torch.eye(mask_positive.size(0)).type('torch.BoolTensor').to(args.device)
        mask_positive.masked_fill_(mask, 0)
        mask_negative.masked_fill_(mask, 0)

        similarity = b_x_b_mean * mask_positive - (0.5 - b_x_b_mean) * mask_negative

        return similarity, mask_positive + mask_negative

    def get_esa_similarity(self, senti_emb, doc_id):
        args = self.args
        b_x_b = self.doc_esa_sim[doc_id]
        b_x_b = b_x_b[:, doc_id]

        mask_positive = torch.gt(b_x_b, args.gamma_positive).type(torch.float32)
        mask_negative = torch.lt(b_x_b, args.gamma_negative).type(torch.float32)

        mask = torch.eye(mask_positive.size(0)).type('torch.BoolTensor').to(args.device)
        mask_positive.masked_fill_(mask, 0)
        mask_negative.masked_fill_(mask, 0)

        similarity = b_x_b * mask_positive - (0.5 - b_x_b) * mask_negative

        return similarity, mask_positive + mask_negative

    def get_regularizer_score(self, probs, senti_emb, doc_id, reg_type):
        """
        sent_emb b x num_senti x embd_size
        probs b x num_class

        """
        r_p_distance = self.distance(probs)

        if reg_type == 'doc':
            similarity, mask = self.get_esa_similarity(senti_emb, doc_id)
        elif reg_type == 'word':
            similarity, mask = self.get_emb_similarity(senti_emb)
        else:
            raise NotImplementedError

        pair_cnt = torch.sum(mask)

        sum_d = torch.sum(r_p_distance * similarity)

        regularizer_score = torch.true_divide(torch.sum(r_p_distance * similarity),
                                              (pair_cnt + torch.tensor(1e-6, dtype=torch.float32)))
        return regularizer_score

    def selectional_preference(self, senti_emb, neg_senti_emb, negation, probs, score_scale):

        # W_norm = torch.norm(self.W.clone(), dim=0, keepdim=True)
        # self.W_after_scale = self.W / W_norm * self.scale

        negation_ = torch.unsqueeze(negation, dim=2)  # negation_ => batch_size x num_senti_ x 1
        u = torch.sigmoid(torch.matmul(senti_emb, self.W) * negation_)  # u => batch_size x num_senti x score_scale
        u_neg_sample = torch.sigmoid(
            -torch.matmul(neg_senti_emb, self.W))  # u_neg_sample => batch_size x num_senti x score_scale

        log_u = torch.mean(torch.log(u), dim=1)  # log_u => batch_size x score_scale
        log_u_neg_sample = torch.mean(torch.log(u_neg_sample), dim=1)  # log_u_neg_sample => batch_size x score_scale

        entropy = - torch.mean(torch.sum(probs * torch.log(probs), dim=1))
        vae_loss = torch.mean(torch.sum((log_u + log_u_neg_sample) * probs, dim=1))

        return -vae_loss, -entropy

    def text_cnn(self, x, y):
        x = torch.unsqueeze(x, 1)
        x = self.dropout(x)

        # x = self.cnn_word_emb(x)
        # x = torch.unsqueeze(x,1)
        # x = self.dropout(x)

        pooled_outputs = []
        for conv_layer in self.cnn_convs:
            conv = F.relu(conv_layer(x))  # [B, F, T, 1]
            conv = torch.squeeze(conv, -1)  # [B, F, T]
            pool = F.max_pool1d(conv, conv.size(2))  # [B, F, 1]
            pooled_outputs.append(pool)

        h_pool = torch.cat(pooled_outputs, 2)  # [B, F, window]
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)])  # [B, F * window]
        h_drop = self.dropout(h_pool_flat)
        h_drop = h_pool_flat

        logit = self.cnn_fc(h_drop)
        self.probs = F.softmax(logit, dim=-1)
        self.pred = torch.argmax(self.probs, dim=-1)
        self.loss = self.loss_func(logit, y)

    def lexicon_learner(self, senti, neg_senti, negation, doc_id):
        senti_emb = self.keyword_emb(senti)
        neg_senti_emb = self.keyword_emb(neg_senti)

        self.vae_loss, entropy_term = self.selectional_preference(senti_emb, neg_senti_emb, negation, self.probs,
                                                                  self.score_scale)
        self.entropy_term_loss = torch.mul(entropy_term, self.alpha)

        senti_emb_copy = Variable(senti_emb.detach().clone(), requires_grad=False).to(self.args.device)
        opinion_reg = self.get_regularizer_score(self.probs, senti_emb_copy, doc_id, self.reg_type)
        self.opinion_reg_loss = torch.mul(opinion_reg, self.beta)

    def forward(self, x, y, senti, neg_senti, negation, doc_id, is_train=True):
        args = self.args
        self.text_cnn(x, y)
        if args.unsupervised and is_train:
            self.lexicon_learner(senti, neg_senti, negation, doc_id)
            self.total_loss = self.vae_loss + self.entropy_term_loss + self.opinion_reg_loss
        else:
            self.total_loss = self.loss
        return self.total_loss
