#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author : "Yifeng Tao", "Xiaojun Ma"
# Last update: March 2021
# =============================================================================
""" 
Implementation of CITRUS model and its variants.

"""
import os
from BioLayer import MaskedBioLayer
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import get_minibatch, evaluate, EarlyStopping, shuffle_data

from base import ModelBase

from sklearn.metrics import accuracy_score

##[(320, 1387), (1387, 1066), (1066, 447), (447, 147), (147, 26), (26, 1)]

from utils import Data

data = Data(
    fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
    fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
    fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
    fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv',
)


class weightConstraint(object):
    """ Define weight constraint for a model layer """
    def __call__(self, module):
        """Set the constraint for correcting  parameters to positive value 
        during training.

        """        
        if hasattr(module, "weight"):
            w = module.weight.data
            w = torch.abs(w)
            module.weight.data = w


class CITRUS(ModelBase):
    """CITRUS model and its variants."""

    def __init__(self, args, masks, **kwargs):
        """Initialize the model.

        Parameters
        ----------
        args: arguments for initializing the model.

        """

        super(CITRUS, self).__init__(args, **kwargs)
        self.masks = masks
        self.analysis_mode = False
        # torch.manual_seed(0)
        
    def analysis(self):
        self.analysis_mode = True

    def build(self, device='cpu'):
        """Define modules of the model."""
        self.device = device
        self.layer_sga_emb = nn.Embedding(
            num_embeddings=self.sga_size + 1,
            embedding_dim=self.embedding_size,
            padding_idx=0,
            scale_grad_by_freq=False
        )

        self.layer_can_emb = nn.Embedding(
            num_embeddings=self.can_size + 1,
            embedding_dim=self.embedding_size,
            padding_idx=0,
            scale_grad_by_freq=False
        )
        
        self.pik_emb = nn.Embedding(
            num_embeddings = 2,
            embedding_dim = self.embedding_size
        )

        self.layer_w_0 = nn.Linear(
            in_features=self.embedding_size, out_features=self.attention_size, bias=True
        )

        self.layer_beta = nn.Linear(
            in_features=self.attention_size, out_features=self.attention_head, bias=True
        )

        self.layer_dropout_1 = nn.Dropout(p=self.dropout_rate)

        self.layer_w_1 = nn.Linear(
            in_features=self.embedding_size, out_features=447, bias=True
        )

        self.layer_dropout_2 = nn.Dropout(p=self.dropout_rate)

        self.layer_w_2 = nn.Linear(
            in_features=320, out_features=self.gep_size, bias=True
        )
        
        # self.processes = nn.Linear(447, 1066, bias=True)
        # mask_2 = torch.from_numpy(self.masks[2].values.T).float().to(device)
        # self.processes.weight.data = self.processes.weight.data * mask_2.T.cpu()
        # self.processes.weight.register_hook(lambda grad: grad.mul_(mask_2.T))
        
        # self.pathways = nn.Linear(1066, 1387, bias=True)
        # mask_1 = torch.from_numpy(self.masks[1].values.T).float().to(device)
        # self.pathways.weight.data = self.pathways.weight.data * mask_1.T.cpu()
        # self.pathways.weight.register_hook(lambda grad: grad.mul_(mask_1.T))
        
        # self.genes = nn.Linear(1387, 320, bias=True)
        # mask_0 = torch.from_numpy(self.masks[0].values.T).float().to(device)
        # # mask_0 = torch.ones_like(mask_0) ## make it fully connected
        # self.genes.weight.data = self.genes.weight.data * mask_0.T.cpu()
        # self.genes.weight.register_hook(lambda grad: grad.mul_(mask_0.T))
        
        path_tf_mask = np.load('pathway_tf_mask2.npy', allow_pickle=True)
        # path_tf_mask = np.load('mm.npy', allow_pickle=True)
        self.genes = nn.Linear(447, 320, bias=True)
        path_tf_mask = torch.from_numpy(path_tf_mask.T).float().to(device)
        self.genes.weight.data = self.genes.weight.data * path_tf_mask.T.cpu()
        self.genes.weight.register_hook(lambda grad: grad.mul_(path_tf_mask.T))
        
        # self.tf_gene = np.where(self.tf_gene>0, 1, 0)
        

        mask_value = torch.FloatTensor(self.tf_gene.T).to(self.device)
        
        # define layer weight clapped by mask
        self.layer_w_2.weight.data = self.layer_w_2.weight.data * torch.FloatTensor(
            self.tf_gene.T
        )
        # register a backford hook with the mask value
        self.layer_w_2.weight.register_hook(lambda grad: grad.mul_(mask_value))
        
        
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        self.loss = nn.MSELoss().to(self.device)

    def forward(self, sga_index, can_index):
        """Forward process.

        Parameters
        ----------
        sga_index: list of SGA index vectors.
        can_index: list of cancer type indices.

        Returns
        -------
        preds: 2D array of float in [0, 1]
          predicted gene expression
        hid_tmr: 2D array of float
          hidden layer of MLP
        emb_tmr: 2D array of float
          tumor embedding
        emb_sga: 2D array of float
          stratified tumor gene embedding
        attn_wt: 2D array of float
          attention weights of SGAs

        """

        sga_index = sga_index.to(self.device).long()
        can_index = can_index.to(self.device).long()
        
        is_mut = torch.from_numpy(np.array([int(212 in sga_index[i]) for i in range(len(sga_index))])).to(self.device).long()
        
        pik = self.pik_emb(is_mut).view(-1, self.embedding_size)
        
        # cancer type embedding
        emb_can = self.layer_can_emb(can_index)
        emb_can = emb_can.view(-1, self.embedding_size)
        
        # dropout input 
        if self.training:
            sga_index2 = (
                torch.empty_like(sga_index).bernoulli_(1 - self.input_dropout_rate)
                * sga_index
            )
        else:
            sga_index2 = sga_index
        E_t = self.layer_sga_emb(sga_index2)

        # squeeze and tanh-curve the gene embeddings
        E_t_flatten = E_t.view(-1, self.embedding_size)
        E_t1_flatten = torch.tanh(self.layer_w_0(E_t_flatten))
        
        # multiplied by attention heads
        E_t2_flatten = self.layer_beta(E_t1_flatten)
        E_t2 = E_t2_flatten.view(-1, self.num_max_sga, self.attention_head)

        # normalize by softmax
        E_t2 = E_t2.permute(1, 0, 2)
        A = F.softmax(E_t2, dim=0)
        A = A.permute(1, 0, 2)
        
        if self.attention:
            # multi-head attention weighted sga embedding:
            emb_sga = torch.sum(torch.bmm(A.permute(0, 2, 1), E_t), dim=1)
            emb_sga = emb_sga.view(-1, self.embedding_size)
        else:
            # if not using attention, simply sum up SGA embeddings
            emb_sga = torch.sum(E_t, dim=1)
            emb_sga = emb_sga.view(-1, self.embedding_size)
        
        # if use cancer type input, add cancer type embedding
        if self.cancer_type:
            emb_tmr = emb_can + emb_sga + pik
        else:
            emb_tmr = emb_sga
        
        # MLP decoder
        emb_tmr_relu = self.layer_dropout_1(self.activation(emb_tmr))
        hid_tmr = self.layer_w_1(emb_tmr_relu)
        hid_tmr_relu = self.layer_dropout_2(self.activation(hid_tmr))
        
        x = hid_tmr_relu
                
        # x = self.processes(self.activation(x))
        # x = self.pathways(self.activation(x))
        x = self.genes(self.activation(x))
        hid_tmr_relu = self.activation(x)
        
        
        preds = self.layer_w_2(hid_tmr_relu)
        

        # attention weights
        attn_wt = torch.sum(A, dim=2)
        attn_wt = attn_wt.view(-1, self.num_max_sga)
        
        if self.analysis_mode:
            return preds
        
        return preds, hid_tmr, x, emb_sga, attn_wt

    def fit(
        self,
        train_set,
        test_set,
        batch_size=None,
        test_batch_size=None,
        max_iter=None,
        max_fscore=None,
        test_inc_size=None,
        **kwargs
    ):
        """Train the model until max_iter or max_fscore reached.

        Parameters
        ----------
        train_set: dict
          dict of lists, including SGAs, cancer types, GEPs, patient barcodes
        test_set: dict
        batch_size: int
        test_batch_size: int
        max_iter: int
          max number of iterations that the training will run
        max_fscore: float
          max test F1 score that the model will continue to train itself
        test_inc_size: int
          interval of running a test/evaluation

        """


        if self.patience != 0:
            es = EarlyStopping(patience=self.patience)
        constraints = weightConstraint()

        for iter_train in range(0, max_iter * len(train_set["can"]), batch_size):

            batch_set = get_minibatch(
                train_set, iter_train, batch_size, batch_type="train"
            )

            preds, hid_tmr, _, _, _ = self.forward(batch_set["sga"], batch_set["can"])
            labels = batch_set["gep"].to(self.device)


            self.optimizer.zero_grad()
            
            loss = self.loss(preds, labels) 
            loss.backward()
            self.optimizer.step()

            self.layer_w_2.apply(constraints)
            # self.genes.apply(constraints)
            

            if (iter_train % len(train_set["can"])) == 0:
                train_set = shuffle_data(train_set)
            if test_inc_size and (iter_train % test_inc_size == 0):
                labels, preds, _, _, _, _, _ = self.test(test_set, test_batch_size)
                corr_spearman, corr_pearson = evaluate(
                    labels, preds, epsilon=self.epsilon
                )
                if corr_spearman is None:
                    corr_spearman = 0
                    
                if corr_pearson is None:
                    corr_pearson = 0
                    
                print(
                    "[%d,%d], spearman correlation: %.6f, pearson correlation: %.6f"
                    % (
                        iter_train // len(train_set["can"]),
                        iter_train % len(train_set["can"]),
                        corr_spearman,
                        corr_pearson,
                    )
                )
                if self.patience != 0:
                    if es.step(corr_pearson) and iter_train > 180 * test_inc_size:

                        # self.save_model(os.path.join(self.output_dir, "trained_model.pth"))
                        break
        self.save_model(os.path.join(self.output_dir, "trained_model.pth"))

    def test(self, test_set, test_batch_size, **kwargs):
        """Run forward process over the given whole test set.

        Parameters
        ----------
        test_set: dict
          dict of lists, including SGAs, cancer types, GEPs, patient barcodes
        test_batch_size: int

        Returns
        -------
        labels: 2D array of 0/1
          groud truth of gene expression
        preds: 2D array of float in [0, 1]
          predicted gene expression
        hid_tmr: 2D array of float
          hidden layer of MLP
        emb_tmr: 2D array of float
          tumor embedding
        emb_sga: 2D array of float
          stratified tumor embedding
        attn_wt: 2D array of float
          attention weights of SGAs
        tmr: list of str
          barcodes of patients/tumors

        """

        labels, preds, hid_tmr, emb_tmr, emb_sga, attn_wt, tmr = (
            [], [], [], [], [], [], [],
        )
        
        self.eval()
        
        for iter_test in range(0, len(test_set["can"]), test_batch_size):
            
            batch_set = get_minibatch(
                test_set, iter_test, test_batch_size, batch_type="test"
            )
            (
                batch_preds,
                batch_hid_tmr,
                batch_emb_tmr,
                batch_emb_sga,
                batch_attn_wt,
            ) = self.forward(batch_set["sga"], batch_set["can"])
            
            batch_labels = batch_set["gep"]

            labels.append(batch_labels.cpu().data.numpy())
            preds.append(batch_preds.cpu().data.numpy())
            hid_tmr.append(batch_hid_tmr.cpu().data.numpy())
            emb_tmr.append(batch_emb_tmr.cpu().data.numpy())
            emb_sga.append(batch_emb_sga.cpu().data.numpy())
            attn_wt.append(batch_attn_wt.cpu().data.numpy())
            tmr = tmr + batch_set["tmr"]
            
        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        hid_tmr = np.concatenate(hid_tmr, axis=0)
        emb_tmr = np.concatenate(emb_tmr, axis=0)
        emb_sga = np.concatenate(emb_sga, axis=0)
        attn_wt = np.concatenate(attn_wt, axis=0)
        
        self.train()
                
        return labels, preds, hid_tmr, emb_tmr, emb_sga, attn_wt, tmr
