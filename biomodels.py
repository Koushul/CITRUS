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
from BioLayer import BioLayer

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
      self.hidden_size = args.hidden_size
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
      self.biolayer = BioLayer(biomask, bias=None)
      if init_weights is not None:
        logger.debug('Using predefined bio weights')
        self.biolayer.weight.data = init_weights

      self.tumor_hidden_layer = nn.Linear(self.nclusters, self.hidden_size, bias=True)
      self.tf_layer = nn.Linear(self.hidden_size+self.embedding_size, self.hidden_size, bias=True) 


      self.gep_output_layer = nn.Linear(
          in_features=self.hidden_size, out_features=self.gep_size, bias=True
      ) ## gene expression output layer

      ## TODO: Refactor this to use the BioLayerMaskFunction instead
      # define layer weight clapped by mask
      mask_value = torch.FloatTensor(self.tf_gene.T).to(device)
      self.gep_output_layer.weight.data = self.gep_output_layer.weight.data * torch.FloatTensor(
          self.tf_gene.T
      )
      # register a hook with the mask value
      self.gep_output_layer.weight.register_hook(lambda grad: grad.mul_(mask_value))

      ## cancer embedding
      self.layer_can_emb = nn.Embedding(
          num_embeddings = self.can_size + 1,
          embedding_dim = self.embedding_size,
          padding_idx = 0,
      )


      self.layer_dropout_1 = nn.Dropout(p=self.dropout_rate)
      self.layer_dropout_2 = nn.Dropout(p=self.dropout_rate)

      self.optimizer = optim.Adam(
          self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
      )

      self.loss = nn.MSELoss().to(device)

  
    def forward(self, sga: torch.Tensor, can: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
      
      sga = sga.to(device)
      can = can.to(device)

      emb_can = self.layer_can_emb(can)
      emb_can = emb_can.view(-1, self.embedding_size)

      h_relu = torch.relu(self.biolayer(sga))
      h = torch.relu(self.tumor_hidden_layer(h_relu))

      if self.cancer_type:
          emb_tmr = torch.cat((emb_can, h), dim=1)
      else:
          emb_tmr = h

      emb_tmr_relu = self.layer_dropout_1(self.activation(emb_tmr))
      hid_tmr = self.tf_layer(emb_tmr_relu)
      hid_tmr_relu = self.layer_dropout_2(self.activation(hid_tmr))
      
      preds = self.gep_output_layer(hid_tmr_relu)

      return preds, hid_tmr



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


      if self.patience != 0:
          es = EarlyStopping(patience=self.patience)
      
      constraints = weightConstraint()

      r = range(0, max_iter * len(train_set["can"]), batch_size)
      # pbar = tqdm(total = len(r))

      for iter_train in r:

        batch_set = get_minibatch(
            train_set, iter_train, batch_size, batch_type="train"
        )

        preds, hid_tmr  = self.forward(batch_set["sga"], batch_set["can"])
        labels = batch_set["gep"].to(device)

        self.optimizer.zero_grad()
        loss = self.loss(preds, labels)
        loss.backward()
        self.optimizer.step()

        self.biolayer.apply(constraints)
        self.gep_output_layer.apply(constraints)

        # pbar.update()


        if (iter_train % len(train_set["can"])) == 0:
          train_set = shuffle_data(train_set)
        
        if test_inc_size and (iter_train % test_inc_size == 0):
          labels, preds, _ = self.test(test_set, test_batch_size)
          corr_spearman, corr_pearson = evaluate(
              labels, preds, epsilon=self.epsilon
          )

          # pbar.set_description(f'CORR: {corr_spearman:.3f} | MSE: {loss.cpu().detach().item():.3f}')


          print(
            "correlation: %.5f, mse: %.5f"
            % (
                corr_spearman,
                loss.cpu().detach().item(),
            )
          )

          # if self.patience != 0:
          #   if es.step(corr_pearson) and iter_train > 180 * test_inc_size:

          #     # self.save_model(os.path.join(self.output_dir, "trained_model.pth"))
          #     break
      
      #self.save_model(os.path.join(self.output_dir, "trained_model.pth"))
      # pbar.close()

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
