#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author : "Koushul Ramjattun"
# =============================================================================

import os
import argparse
import random
from utils import cfh, logger, Data, bool_ext, checkCorrelations, generate_masks_from_ppi
from biomodels import BioCitrus
import torch
import numpy as np
import sys
import pandas as pd

import warnings
warnings.filterwarnings("ignore") ##This is bad but temporary

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
    "--algo", 
    help="clustering algorithm to use on the portein-protein network (DPCLUS, MCODE, COACH)", 
    type=str, 
    default='COACH'
)


parser.add_argument(
    "--learning_rate", 
    help="learning rate for Adam", 
    type=float, 
    default=1e-2
)
parser.add_argument(
    "--max_iter", 
    help="maximum number of training iterations", 
    type=int, 
    default=100
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
    default=64
)
parser.add_argument(
    "--dropout_rate", 
    help="dropout rate", 
    type=float, 
    default=0.1
)

parser.add_argument(
    "--weight_decay", 
    help="coefficient of l2 regularizer", 
    type=float, 
    default=1e-2
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
    default=50
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
    default=False,
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

parser.add_argument(
    "--ppi_weights", 
    help="", 
    type=bool_ext, 
    default=False
)

parser.add_argument(
    "--verbose", 
    help="", 
    type=bool_ext, 
    default=False
)

parser.add_argument(
    "--constrain", 
    help="force weight and biases to be strictly non-negative", 
    type=bool_ext, 
    default=True
)

parser.add_argument(
    "--biases", 
    help="enable all nn.Linear biases", 
    type=bool_ext, 
    default=True
)

parser.add_argument(
    "--sparse", 
    help="only use SIGNOR data, resulting in sparser connections", 
    type=bool_ext, 
    default=False
)

parser.add_argument(
    "--cancer", 
    help="", 
    type=str, 
    default=''
)




args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logger.info('Training BioCITRUS on %s' % device_name)

cancer_type = args.cancer


args.cancer_type = cancer_type

collected_metrics = []

for i in range(1):
    # logger.info(f'Cancer Type: {cancer_type}')
    
    data = Data(
        fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
        fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
        fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
        fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv',
        # cancer_type='BRCA'
    )
    
    
    train_set, test_set = data.get_train_test()
    args.gep_size = train_set['gep'].shape[1]
    args.tf_gene = data.gene_tf_sga.values.T
    args.can_size = len(np.unique(data.cancer_types))
    args.sga_size = 11998
    

    sga_mask, sga_weights, tf_mask, tf_weights = generate_masks_from_ppi(sga = data.sga_sga, tf = data.gene_tf_sga, clust_algo=args.algo, sparse=args.sparse)

    # np.save(f'experiments/init_weights', weights)

    sga_mask = sga_mask.to(device)
    sga_weights = sga_weights.t().to(device)
    tf_mask = tf_mask.t().to(device)
    tf_weights = tf_weights.to(device)

    if not args.ppi_weights:
        ppi_weights = None
        tf_weights = None

    # sga_mask = torch.ones_like(sga_mask)
    # tf_mask = torch.ones_like(tf_mask)

    # biomask = torch.zeros_like(biomask)
    # idx = torch.randperm(biomask.nelement())
    # biomask = biomask.view(-1)[idx].view(biomask.size())


    model = BioCitrus(
        args = args, 
        sga_ppi_mask = sga_mask, 
        ppi_tf_mask = tf_mask, 
        sga_ppi_weights = None, 
        ppi_tf_weights = None,
        enable_bias = args.biases
    ).to(device)

    # sys.exit(1)

    try:
        model.fit(
            train_set,
            test_set,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            max_iter=args.max_iter,
            max_fscore=args.max_fscore,
            test_inc_size=args.test_inc_size, 
            verbose = args.verbose,
            # cancer_type = 'BRCA'
        )
        
    except KeyboardInterrupt: # exit gracefully
        logger.warning('Training Interrupted by User')
        
    # collected_metrics.append(model.metrics)
    
    # torch.save(model.state_dict(), f'/ix/hosmanbeyoglu/kor11/CITRUS_models/embedded_model.pth')
    

# torch.save(model.state_dict(), f'/ix/hosmanbeyoglu/kor11/CITRUS_models/{cancer_type}_{i}.pth')

# sys.exit(1)

# # Evaluating the model on the test set.
# logger.info("Evaluating on test set...")
# labels, preds, _  = model.test(
#     test_set, test_batch_size=args.test_batch_size
# )

# checkCorrelations(labels, preds)


# logger.info("Evaluating on test set...")
# get training attn_wt and others
# labels, preds, hid_tmr, emb_tmr, emb_sga, attn_wt, tmr = model.test(
#     dataset, test_batch_size=args.test_batch_size
# )
# print("Predicting on test dataset...\n")
# predict on holdout and evaluate the performance
# labels_test, preds_test, _  = model.test(test_set, test_batch_size=args.test_batch_size)

# print("\nPerformance on holdout test set:\n")
# checkCorrelations(labels_test, preds_test)

# np.save(f'experiments/sga_layer_weights_{random.randint(0, 99999)}', model.sga_layer[0].weight.data.cpu().numpy())

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
