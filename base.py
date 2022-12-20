#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author : "Yifeng Tao", "Xiaojun Ma"
# Finalized Date: March 2021
# =============================================================================
""" 
Base model of CITRUS and its variants.

"""
import torch
import torch.nn as nn



class ModelBase(nn.Module):
    """Base models for all models."""

    def __init__(self, args):
        """Initialize the hyperparameters of model.

        Parameters
        ----------
        args: arguments for initializing the model.

        """

        super(ModelBase, self).__init__()

        self.epsilon = 1e-4

        self.input_dir = args.input_dir
        self.output_dir = args.output_dir

        self.sga_size = args.sga_size
        self.gep_size = args.gep_size
        self.can_size = args.can_size

        self.num_max_sga = args.num_max_sga

        self.embedding_size = args.embedding_size
        self.hidden_size = args.hidden_size
        self.attention_size = args.attention_size
        self.attention_head = args.attention_head

        self.learning_rate = args.learning_rate
        self.dropout_rate = args.dropout_rate
        self.weight_decay = args.weight_decay
        self.input_dropout_rate = args.input_dropout_rate

        self.attention = args.attention
        self.cancer_type = args.cancer_type

        if args.activation == "relu":
            self.activation = torch.relu
        if args.activation == "tanh":
            self.activation = torch.tanh
            
        self.tf_gene = args.tf_gene
        self.mask01 = args.mask01
        self.gep_normalization = args.gep_normalization
        
        self.patience = args.patience
        self.run_count = args.run_count
        
        self.tag = args.tag
       
    def build(self):
        """Define modules of the model."""

        raise NotImplementedError

    def forward(self):
        """Define the data flow across modules of the model."""

        raise NotImplementedError

    def fit(self):
        """Train the model using training set."""

        raise NotImplementedError

    def test(self):
        """Test the model using test set."""
        raise NotImplementedError

    def load_model(self, path, device):
        """Load trained parameters of the model."""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['state_dict'])
                
        self.performance = checkpoint['performance']
        self.pval_corr = checkpoint['pval_corr']
        self.cancers = checkpoint['cancers']

    def save_model(self, path="data/trained_model.pth"):
        """Save learnable parameters of the trained model."""

        print("Saving model to " + path)
        torch.save(
            {
                'state_dict': self.state_dict(),
                'performance': self.performance,
                'pval_corr': self.pval_corr,
                'cancers': self.cancers,
                'idd': self.idd,
                'iter_corr': self.iter_corr,
                'uuid': self.uuid
            }, path)
