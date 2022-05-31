#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author : "Koushul Ramjattun"
# =============================================================================

from collections import defaultdict
import os
import argparse
from utils import Data, bool_ext, generate_mask_from_ppi, load_dataset, load_sga, split_dataset, evaluate, checkCorrelations
from biomodels import BioCitrus
import pickle
import pandas as pd
import torch
import numpy as np
import logging
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    device_name = torch.cuda.get_device_name(0)
else:
    device_name = 'cpu'

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir", 
    help="directory of input files", 
    type=str, 
    default="./data"
)
parser.add_argument(
    "--output_dir",
    help="directory of output files",
    type=str,
    default="./output",
)
parser.add_argument(
    "--embedding_size",
    help="embedding dimension of genes and tumors",
    type=int,
    default=512,
)
parser.add_argument(
    "--hidden_size", 
    help="hidden layer dimension of MLP decoder", 
    type=int, 
    default=320
)

parser.add_argument(
    "--algo", 
    help="clustering algorithm to use on the portein-protein network", 
    type=str, 
    default='DPCLUS'
)


parser.add_argument(
    "--learning_rate", 
    help="learning rate for Adam", 
    type=float, 
    default=1e-3
)
parser.add_argument(
    "--max_iter", 
    help="maximum number of training iterations", 
    type=int, 
    default=1000
)
parser.add_argument(
    "--max_fscore",
    help="Max F1 score to early stop model from training",
    type=float,
    default=0.7
)
parser.add_argument(
    "--batch_size", 
    help="training batch size", 
    type=int, 
    default=100
)
parser.add_argument(
    "--test_batch_size", 
    help="test batch size", 
    type=int, 
    default=100
)
parser.add_argument(
    "--test_inc_size",
    help="increment interval size between log outputs",
    type=int,
    default=256
)
parser.add_argument(
    "--dropout_rate", 
    help="dropout rate", 
    type=float, 
    default=0.2
)
parser.add_argument(
    "--input_dropout_rate", 
    help="dropout rate", 
    type=float, 
    default=0.2
)
parser.add_argument(
    "--weight_decay", 
    help="coefficient of l2 regularizer", 
    type=float, 
    default=1e-5
)
parser.add_argument(
    "--activation",
    help="activation function used in hidden layer",
    type=str,
    default="tanh",
)
parser.add_argument(
    "--patience", 
    help="earlystopping patience", 
    type=int, 
    default=30
)
parser.add_argument(
    "--mask01",
    help="wether to ignore the float value and convert mask to 01",
    type=bool_ext,
    default=True,
)
parser.add_argument(
    "--gep_normalization", 
    help="how to normalize gep", 
    type=str, 
    default="scaleRow"
)

parser.add_argument(
    "--cancer_type",
    help="whether to use cancer type or not",
    type=bool_ext,
    default=True,
)
parser.add_argument(
    "--train_model",
    help="whether to train model or load model",
    type=bool_ext,
    default=True,
)
parser.add_argument(
    "--dataset_name",
    help="the dataset name loaded and saved",
    type=str,
    default="dataset_CITRUS",
)
parser.add_argument(
    "--tag", 
    help="a tag passed from command line", 
    type=str, 
    default=""
)
parser.add_argument(
    "--run_count", 
    help="the count for training", 
    type=str, 
    default="1"
)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logger.info('Training BioCITRUS on %s' % device_name)

data = Data(
    fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
    fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
    fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
    fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv'
)

train_set, test_set = data.get_train_test()
args.gep_size = train_set['gep'].shape[1]
args.tf_gene = data.gene_tf_sga.values.T
args.can_size = len(np.unique(data.cancer_types))


biomask = generate_mask_from_ppi(sga = data.sga_sga, clust_algo=args.algo).to(device)

# biomask = torch.zeros_like(biomask)
# idx = torch.randperm(biomask.nelement())
# biomask = biomask.view(-1)[idx].view(biomask.size())

model = BioCitrus(args, biomask=biomask)  
model.to(device)

if args.train_model:  # train from scratch

    model.fit(
        train_set,
        test_set,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        max_iter=args.max_iter,
        max_fscore=args.max_fscore,
        test_inc_size=args.test_inc_size,
    )
    model.save_model(os.path.join(args.output_dir, "trained_model.pth"))
else:  # or directly load trained model
    model.load_model(os.path.join(args.input_dir, "trained_model.pth"))
    
# evaluation
print("Evaluating...")
labels, preds, _  = model.test(
    test_set, test_batch_size=args.test_batch_size
)
print("\nPerformance on validation set:\n")
checkCorrelations(labels, preds)

# print("\nPredicting on main dataset...\n")
# get training attn_wt and others
# labels, preds, hid_tmr, emb_tmr, emb_sga, attn_wt, tmr = model.test(
#     dataset, test_batch_size=args.test_batch_size
# )
# print("Predicting on test dataset...\n")
# predict on holdout and evaluate the performance
labels_test, preds_test, _  = model.test(test_set, test_batch_size=args.test_batch_size)

print("\nPerformance on holdout test set:\n")
checkCorrelations(labels_test, preds_test)

np.save('biolayer_weights', model.biolayer.weight.data.cpu().numpy())

# dataset_out = {
#     "labels": labels,         # measured exp 
#     "preds": preds,           # predicted exp
#     "hid_tmr": hid_tmr,       # TF activity
#     "emb_tmr": emb_tmr,       # tumor embedding
#     "tmr": tmr,               # tumor list
#     "emb_sga": emb_sga,       # straitified tumor embedding
#     "attn_wt": attn_wt,       # attention weight
#     "can": dataset["can"],    # cancer type list
#     "gene_emb": gene_emb,     # gene embedding
#     "tf_gene": model.gep_output_layer.weight.data.cpu().numpy(),  # trained weight of tf_gene constrains
#     "labels_test":labels_test,      # measured exp on test set
#     "preds_test":preds_test,        # predicted exp on test set
#     "tmr_test":tmr_test,            # tumor list on test set
#     "can_test":dataset_test["can"]  # cancer type list on test set
# }

# with open(os.path.join(args.output_dir, "output_{}_{}{}.pkl".format(args.dataset_name, args.run_count, args.tag)), "wb") as f:
#   pickle.dump(dataset_out, f, protocol=2)
