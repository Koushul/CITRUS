import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import argparse
from utils import cfh, logger, Data, bool_ext, checkCorrelations, generate_masks_from_ppi
import os
import argparse
import random
from utils import cfh, logger, Data, bool_ext, checkCorrelations, generate_masks_from_ppi
from biomodels import BioCitrus
import torch
import numpy as np
import sys
from biomodels import weightConstraint
from utils import logger, get_minibatch, evaluate, EarlyStopping, shuffle_data
from tqdm import tqdm
from pathlib import Path
from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore") ##This is bad but temporary

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
    default=300
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
    default=20
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

args = parser.parse_args([])

data = Data(
    fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
    fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
    fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
    fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv',
    cancer_type='BRCA'
)

train_set, test_set = data.get_train_test()
args.gep_size = train_set['gep'].shape[1]
args.tf_gene = data.gene_tf_sga.values.T
args.can_size = len(np.unique(data.cancer_types))


sga_mask, sga_weights, tf_mask, tf_weights = generate_masks_from_ppi(sga = data.sga_sga, tf = data.gene_tf_sga, clust_algo=args.algo, sparse=args.sparse)


sga_mask = sga_mask
sga_weights = sga_weights.t()
tf_mask = tf_mask.t()
tf_weights = tf_weights

from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients

models = []
for i in range(30):
    
    model = BioCitrus(
        args = args, 
        sga_ppi_mask = sga_mask, 
        ppi_tf_mask = tf_mask, 
        sga_ppi_weights = None, 
        ppi_tf_weights = None,
        enable_bias = args.biases
    )

    model.load_state_dict(torch.load(f'/ix/hosmanbeyoglu/kor11/CITRUS_models/BRCA_{i}.pth', 
                                map_location=torch.device('cuda')))
    
    model.eval()
    model.cuda()
    
    models.append(model)
    clear_output(wait=True)
    
    
X = torch.tensor(test_set['sga'])
Y = test_set['gep']

ml_scores = []
iix = list(data.gep_sga.columns).index('PIK3CA')
pbar = tqdm(total=len(models))
for mx, model in enumerate(models):
    pbar.set_description('PIK3CA')
    attr_scores = []
    ig = IntegratedGradients(model)
    idx = iix
    ig_attr_test = ig.attribute(X, target=[idx]*len(X))
    ig_attr_test_sum = ig_attr_test.detach().numpy().sum(0)
    ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)
    attr_scores.append(ig_attr_test_norm_sum)
    ml_scores.append(attr_scores)
    pbar.update()
pbar.close()


sms = []
scnas = []
top = []

for i in range(30):
    ll = list(pd.DataFrame(ml_scores[i][0], 
                        index=data.sga_sga.columns).sort_values(by=0, ascending=False).index)
    sms.append(ll.index('SM_PIK3CA'))
    scnas.append(ll.index('SCNA_PIK3CA'))
    top.append(ll[:10])
    
r = pd.DataFrame(top)
r['rank'] = sms

models = np.array(models)[r.sort_values(by='rank').index[:10]]

all_attr_scores = np.load('all_attr_scores.npy')


def tf_activity(model, target_gene):
    lc = LayerConductance(model, model.gep_output_layer)
    ix = list(data.gep_sga.columns).index(target_gene)
    a = lc.attribute(X, n_steps=7, attribute_to_layer_input=True, target=[ix]*len(X))
    
    ig_attr_test_sum = a.detach().cpu().numpy().sum(0)
    ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)
    
    # g = np.array(data.gene_tf_sga.columns)[np.where(ig_attr_test_norm_sum != 0)[0]]
    # at = ig_attr_test_norm_sum[np.where(ig_attr_test_norm_sum != 0)[0]]
    
    g = np.array(data.gene_tf_sga.columns)
    at = ig_attr_test_norm_sum
    
    df = pd.DataFrame([g, at]).T
    df.columns = ['TF', 'score']
    
    return df.sort_values(by='score', ascending=False)





data = Data(
    fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
    fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
    fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
    fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv',
    cancer_type='BRCA'
)

train_set, test_set = data.get_train_test()
df = data.sga_sga

X1 = df[(df['SM_PIK3CA']==1)]
X0 = df[(df['SM_PIK3CA']==0)]


tf_profiles = np.zeros((X0.shape[0], 320))
model = models[0]
pbar = tqdm(total=len(data.gep_sga.columns))

for ix, g in enumerate(data.gep_sga.columns):
    lc = LayerConductance(model, model.gep_output_layer)
    a = lc.attribute(torch.from_numpy(X0.values), n_steps=5, attribute_to_layer_input=True, target=[ix]*len(X0))
    tf_profiles += a.detach().cpu().numpy()
    pbar.update()
    
pbar.close()
np.save('tf_profiles_1.npy', tf_profiles)

tf_profiles2 = np.zeros((X1.shape[0], 320))
model = models[0]
pbar = tqdm(total=len(data.gep_sga.columns))

for ix, g in enumerate(data.gep_sga.columns):
    lc = LayerConductance(model, model.gep_output_layer)
    a = lc.attribute(torch.from_numpy(X1.values), n_steps=5, attribute_to_layer_input=True, target=[ix]*len(X1))
    tf_profiles2 += a.detach().cpu().numpy()
    pbar.update()
    
pbar.close()

np.save('tf_profiles_2.npy', tf_profiles2)

