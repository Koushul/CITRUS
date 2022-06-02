#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author : "Koushul Ramjattun"
# =============================================================================

import os
from typing import Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from MaskedBioLayer import MaskedBioLayer

from utils import logger, get_minibatch, evaluate, EarlyStopping, shuffle_data
from base import ModelBase
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'


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


class BioCitrus(nn.Module):

    def __init__(self, args, biomask, init_weights=None):
      super(BioCitrus, self).__init__()


      ## Hyperparameters
      self.epsilon = 1e-4
      self.dropout_input = True

      self.embedding_size = args.embedding_size
      self.can_size = args.can_size
      self.tf_size = args.tf_gene.shape[0]
      self.gep_size = args.gep_size
      self.tf_gene = args.tf_gene
      self.patience = args.patience
      self.cancer_type = args.cancer_type

      self.learning_rate = args.learning_rate
      self.dropout_rate = args.dropout_rate
      self.weight_decay = args.weight_decay
      if args.activation == "relu":
          self.activation = torch.relu
      if args.activation == "tanh":
          self.activation = torch.tanh

      self.nclusters = biomask.shape[1]
      
      ## Simple Layers
      self.biolayer = nn.Sequential(
        MaskedBioLayer(biomask, bias=None, init_weights=init_weights),
        nn.Tanh()
      )


      if self.cancer_type:
        ## cancer embedding
        self.layer_can_emb = nn.Embedding(
            num_embeddings = self.can_size + 1,
            embedding_dim = self.embedding_size,
            padding_idx = 0,
        )

        self.tf_layer = nn.Sequential(
          nn.Linear(self.nclusters+self.embedding_size, self.tf_size, bias=True),
          nn.Tanh(),
          nn.Dropout(p=self.dropout_rate)
        )

      
      else:
        self.tf_layer = nn.Sequential(
          # nn.Linear(self.tf_size, self.tf_size, bias=True),
          nn.Linear(self.nclusters, self.tf_size, bias=True),
          nn.Tanh(),
          nn.Dropout(p=self.dropout_rate)
        )

      self.gep_output_layer = nn.Linear(
          in_features=self.tf_size, out_features=self.gep_size, bias=True
      ) ## gene expression output layer

      ## TODO: Refactor this to use the BioLayerMaskFunction instead
      # define layer weight clapped by mask
      mask_value = torch.FloatTensor(self.tf_gene.T).to(device)
      self.gep_output_layer.weight.data = self.gep_output_layer.weight.data * torch.FloatTensor(
          self.tf_gene.T
      )
      # register a hook with the mask value
      self.gep_output_layer.weight.register_hook(lambda grad: grad.mul_(mask_value))

      self.optimizer = optim.Adam(
          self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
      )

      self.criterion = nn.MSELoss()

      for name, param in self.named_parameters():
        logger.debug(f'{name} | {param.requires_grad} | {tuple(param.shape)}')

  
    def forward(self, sga: torch.Tensor, can: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
      
      sga = sga.to(device)

      ppi = self.biolayer(sga)

      if self.cancer_type:
          can = can.to(device)
          emb_can = self.layer_can_emb(can)
          emb_can = emb_can.view(-1, self.embedding_size)
          emb_tmr = torch.cat((emb_can, ppi), dim=1)
      else:
          emb_tmr = ppi

      tf = self.tf_layer(emb_tmr)      
      gexp = self.gep_output_layer(tf)

      return gexp, tf


    def fit(
      self,
      train_set: dict,
      test_set: dict,
      batch_size: int = None,
      test_batch_size: int = None,
      max_iter: int = None,
      max_fscore: float = None,
      test_inc_size: int = None,
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

      self.verbose = kwargs.get('verbose', True)


      if self.patience != 0:
          es = EarlyStopping(patience=self.patience)
      
      constraints = weightConstraint()
      print('')


      r = range(0, max_iter * len(train_set["can"]), batch_size)
      if not self.verbose: pbar = tqdm(total = len(r))

      for iter_train in r:

        batch_set = get_minibatch(
            train_set, iter_train, batch_size, batch_type="train"
        )

        preds, hid_tmr  = self.forward(batch_set["sga"], batch_set["can"])
        labels = batch_set["gep"].to(device)

        self.optimizer.zero_grad()
        loss = self.criterion(preds, labels)
        loss.backward()
        self.optimizer.step()

        # self.biolayer.apply(constraints)
        self.gep_output_layer.apply(constraints)

        if not self.verbose: pbar.update()

        if (iter_train % len(train_set["can"])) == 0:
          train_set = shuffle_data(train_set)
        
        if iter_train == 0 or (test_inc_size and (iter_train % test_inc_size == 0)):
          labels, preds, _ = self.test(test_set, test_batch_size)
          corr_spearman, corr_pearson = evaluate(
              labels, preds, epsilon=self.epsilon)

          if not self.verbose:
            pbar.set_description(f'CORR: {corr_spearman:.3f} | MSE: {loss.cpu().detach().item():.3f} | {self.biolayer.weight.data.sum():.3f}')

          else:
            print('\x1b[38;5;196m correlation: %.5f, loss: %.5f | w_: %.5f \x1b[0m' % 
                (corr_spearman, loss.cpu().detach().item(), self.biolayer[0].weight.data.sum()))

          if self.patience != 0:
            if es.step(corr_pearson) and iter_train > 180 * test_inc_size:
              break
            
      if not self.verbose: pbar.close()

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

        """

        labels, preds, hid_tmr = ([], [], [])
        
        self.eval()
        
        for iter_test in range(0, len(test_set["can"]), test_batch_size):
            
            batch_set = get_minibatch(
                test_set, iter_test, test_batch_size, batch_type="test"
            )
            
            batch_preds, batch_hid_tmr = self.forward(batch_set["sga"], batch_set["can"])
            batch_labels = batch_set["gep"]

            labels.append(batch_labels.cpu().data.numpy())
            preds.append(batch_preds.cpu().data.numpy())
            hid_tmr.append(batch_hid_tmr.cpu().data.numpy())
                        
        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        hid_tmr = np.concatenate(hid_tmr, axis=0)

        self.train()
        
        return labels, preds, hid_tmr


    # def init_weights(self):
    #   torch.nn.init.xavier_uniform(self.biolayer.weight)
    #   torch.nn.init.xavier_uniform(self.linear.weight)


    def load_model(self, path="data/trained_model.pth"):
      """Load trained parameters of the model."""

      print("Loading model from " + path)
      self.load_state_dict(torch.load(path))

    def save_model(self, path="data/trained_model.pth"):
        """Save learnable parameters of the trained model."""

        print("Saving model to " + path)
        torch.save(self.state_dict(), path)
