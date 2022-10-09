#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author : "Yifeng Tao", "Xiaojun Ma"
# Last update: March 2021
# =============================================================================
""" 
Demo of training and evaluating CITRUS model and its variants.

"""
import os
import argparse
from utils import bool_ext, load_dataset, split_dataset, evaluate, checkCorrelations
from models import CITRUS
import pickle
import torch
import numpy as np
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
    "--embedding_size",
    help="embedding dimension of genes and tumors",
    type=int,
    default=512,
)
parser.add_argument(
    "--hidden_size", 
    help="hidden layer dimension of MLP decoder", 
    type=int, 
    default=400
)
parser.add_argument(
    "--attention_size", 
    help="size of attention parameter beta_j", 
    type=int, 
    default=256
)
parser.add_argument(
    "--attention_head", 
    help="number of attention heads", 
    type=int, 
    default=8
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
    default=75
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
    default=10
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
    "--attention",
    help="whether to use attention mechanism or not",
    type=bool_ext,
    default=True,
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

parser.add_argument(
    "--label", 
    help="model label", 
    type=str, 
    default="untitled"
)

parser.add_argument(
    "--ppi", 
    help="", 
    type=int, 
    default=0
)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

experiment = args.label


print("Loading dataset...")
dataset, dataset_test = load_dataset(
    input_dir=args.input_dir,
    mask01=args.mask01,
    dataset_name=args.dataset_name,
    gep_normalization=args.gep_normalization,
)


train_set, test_set = split_dataset(dataset, ratio=0.66)

args.can_size = dataset["can"].max()  # cancer type dimension
args.sga_size = dataset["sga"].max()  # SGA dimension
args.gep_size = dataset["gep"].shape[1]  # GEP output dimension
args.num_max_sga = dataset["sga"].shape[1]  # maximum number of SGAs in a tumor

args.hidden_size = dataset["tf_gene"].shape[0]
print("Hyperparameters:")
print(args)
args.tf_gene = dataset["tf_gene"]


# masks = np.load('./pnet_prostate_paper/train/maps.npy', allow_pickle=True)


model = CITRUS(args)  # initialize CITRUS model
model.build(device=device)  # build CITRUS model
model.to(device)

print(model)

# exit()



# np.random.shuffle(train_set['sga'])
# np.random.shuffle(train_set['can'])
# np.random.shuffle(train_set['gep'])


if args.train_model:  # train from scratch
    print(f"Training on {device_name}...")
    model.fit(
        train_set,
        test_set,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        max_iter=args.max_iter,
        max_fscore=args.max_fscore,
        test_inc_size=args.test_inc_size,
    )
    # model.save_model(os.path.join(args.output_dir, f'model_{args.run_count}_{args.tag}.pth'))
    model.save_model(os.path.join(args.output_dir, f'model0909.pth'))
else:  # or directly load trained model
    model.load_model(os.path.join(args.input_dir, "trained_model.pth"))
    
# evaluation
print("Evaluating...")
labels, preds, _, _, _, _, _, _ = model.test(
    test_set, test_batch_size=args.test_batch_size
)
print("\nPerformance on validation set:")
checkCorrelations(labels, preds)

# print("\nPredicting on main dataset...\n")
# get training attn_wt and others
labels, preds, tf, hid_tmr, emb_tmr, emb_sga, attn_wt, tmr = model.test(
    dataset, test_batch_size=args.test_batch_size
)
# print("Predicting on test dataset...\n")
# predict on holdout and evaluate the performance
labels_test, preds_test, _, _, _, _, _, tmr_test = model.test(dataset_test, test_batch_size=args.test_batch_size)

print("\nPerformance on holdout test set:")
checkCorrelations(labels_test, preds_test)



gene_emb = model.layer_sga_emb.weight.data.cpu().numpy()
dataset_out = {
    "labels": labels,         # measured exp 
    "preds": preds,           # predicted exp
    "hid_tmr": tf,            # TF activity
    "pathways": hid_tmr,      # pathways activity
    "emb_tmr": emb_tmr,       # tumor embedding
    "tmr": tmr,               # tumor list
    "emb_sga": emb_sga,       # straitified tumor embedding
    "attn_wt": attn_wt,       # attention weight
    "can": dataset["can"],    # cancer type list
    "gene_emb": gene_emb,     # gene embedding
    "tf_gene": model.layer_w_2.weight.data.cpu().numpy(),  # trained weight of tf_gene constrains
    "labels_test": labels_test,      # measured exp on test set
    "preds_test": preds_test,        # predicted exp on test set
    "tmr_test": tmr_test,            # tumor list on test set
    "can_test": dataset_test["can"]  # cancer type list on test set
}


# with open(os.path.join(args.output_dir, "outputx_{}_{}{}.pkl".format(args.dataset_name, args.run_count, args.tag)), "wb") as f:
#   pickle.dump(dataset_out, f, protocol=2)


import pandas as pd


# xf = model.xf


# pths = [hid_tmr]
    
# ## genereate ensemble tf activity matrix
# pths_ensemble = 0
# pths_ensemble += pths[0]
    
# pths_ensemble = pths_ensemble/1

# def generateTFactivity(tf, idx2can, tmr, cans, tf_name):
#     # generate the TF activity matrix
#     df_TF = pd.DataFrame(data = tf, columns = tf_name, index = tmr)
#     can_names = [idx2can[idx] for idx in cans]
#     df_TF["cancer_type"] = can_names
#     return(df_TF)


# data = pickle.load( open("/ihome/hosmanbeyoglu/kor11/tools/CITRUS/data/dataset_CITRUS.pkl", "rb"))
# pathways = xf.columns
# pth_tf = generateTFactivity(pths_ensemble, data["idx2can"],data["tmr"], data["can"], pathways)


# df = pd.DataFrame(np.column_stack([data['tmr'], data['can']]), columns=['tmr', 'cancer'])
# df['cancer'] = df['cancer'].astype(int).replace(data['idx2can'])

from utils import Data

data_csv = Data(
    fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
    fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
    fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
    fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv',
)

# def split_mutants(cancer, gene):    
#     _sm = f'SM_{gene}'
#     _scna = f'SCNA_{gene}'
    
#     dframe = data_csv.sga_sga.loc[df[df.cancer==cancer].tmr]
    
#     wt = dframe[(dframe[_sm] == 0) & (dframe[_scna] == 0)]
#     sm = dframe[(dframe[_sm] == 1) & (dframe[_scna] == 0)]
#     scna = dframe[(dframe[_sm] == 0) & (dframe[_scna] == 1)]
#     sm_scna = dframe[(dframe[_sm] == 1) & (dframe[_scna] == 1)]
    
#     return wt.index.values, sm.index.values, scna.index.values, sm_scna.index.values


# wt, sm, _, _ = split_mutants('BRCA', 'PIK3CA')
# a = pth_tf.loc[wt].values[:, :len(pathways)]
# b = pth_tf.loc[sm].values[:, :len(pathways)]


# from scipy.stats import ttest_ind as ttest

# r = pd.DataFrame([ttest(a[:, j], b[:, j]).pvalue for j in range(len(pathways))], 
#         index=pathways).sort_values(by=0)
# r.columns = ['pvalue']
# print(r.sort_values(by='pvalue')[:25])




import pandas as pd
from scipy.stats import ttest_1samp as ttest
import warnings 
from sklearn import metrics
warnings.filterwarnings("ignore")

d = data_csv.cancerType_sga.loc[dataset['tmr']]
d['index'] = dataset['can'].reshape(-1)


daata = pickle.load( open("/ihome/hosmanbeyoglu/kor11/tools/CITRUS/data/dataset_CITRUS.pkl", "rb") )
cancers = daata['idx2can']

model.eval()


preds, tf, hid_tmr, tf, _, _  = model.forward(torch.tensor(test_set['sga']), torch.from_numpy(test_set['can']))
genes_ = test_set['gep'].shape[1]
test_df = pd.DataFrame(np.concatenate([test_set['gep'], test_set['can'], preds.detach().cpu().numpy()], axis=1))

test_cancers = {}
for ix, canc in cancers.items():
    test_cancers[canc] =  {}
    test_cancers[canc]['test'] = test_df[test_df[genes_]==ix+1].values[:, :genes_]    
    test_cancers[canc]['pred'] = test_df[test_df[genes_]==ix+1].values[:, genes_+1:] 
    
o = ['BLCA', 'BRCA', 'CESC', 'COAD', 
    'ESCA', 'GBM', 'HNSC', 'KIRC', 
    'KIRP', 'LIHC', 'LUAD', 'LUSC', 
    'PCPG', 'PRAD', 'STAD', 'THCA', 
    'UCEC']

_corrs = []
_mses = []
for canc in o:
        corr = checkCorrelations(test_cancers[canc]['test'], test_cancers[canc]['pred'], return_value=True)
        mse = metrics.mean_squared_error(test_cancers[canc]['test'], test_cancers[canc]['pred'])
        
        _corrs.append(corr)
        _mses.append(mse)
        
print('')
print(pd.DataFrame(np.column_stack([_corrs, _mses]), index=o, columns=['CORR', 'MSE']))
