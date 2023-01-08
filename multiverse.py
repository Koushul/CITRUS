import os
import numpy as np
from filelock import FileLock

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import ray
import uuid

import os
import argparse
from utils import bool_ext, load_dataset, split_dataset, evaluate, checkCorrelations
from models import CITRUS
import pickle
import torch
import numpy as np
import warnings 
warnings.filterwarnings("ignore")
import yaml

import pandas as pd
from scipy.stats import ttest_1samp as ttest
import warnings 
from sklearn import metrics
warnings.filterwarnings("ignore")

# ray.init()

num_evaluations = 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open('args.yaml', 'r') as f:
    args_dict = yaml.safe_load(f)
    
tf_gene = np.load('tf_gene.npy')
    
def generate_hyperparameters():
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(**args_dict)
    args.tf_gene = tf_gene
    
    return args

  
  
from utils import Data

data_csv = Data(
    fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
    fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
    fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
    fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv',
)
 
daata = pickle.load( open("/ihome/hosmanbeyoglu/kor11/tools/CITRUS/data/dataset_CITRUS.pkl", "rb") )


# dataset, dataset_test = load_dataset(
#     input_dir = args_dict['input_dir'],
#     mask01 = args_dict['mask01'],
#     dataset_name = args_dict['dataset_name'],
#     gep_normalization = args_dict['gep_normalization'],
# )

# d = data_csv.cancerType_sga.loc[dataset['tmr']]
# d['index'] = dataset['can'].reshape(-1)
cancers = daata['idx2can']
        
@ray.remote(num_gpus=float(1/num_evaluations))
def fractured_universe(args, idd):
    
    dataset, dataset_test = load_dataset(
        input_dir = args_dict['input_dir'],
        mask01 = args_dict['mask01'],
        dataset_name = args_dict['dataset_name'],
        gep_normalization = args_dict['gep_normalization'],
    )

    train_set, test_set = split_dataset(dataset, ratio=0.66)



    args.can_size = dataset["can"].max()  # cancer type dimension
    args.sga_size = dataset["sga"].max()  # SGA dimension
    args.gep_size = dataset["gep"].shape[1]  # GEP output dimension
    args.num_max_sga = dataset["sga"].shape[1]  # maximum number of SGAs in a tumor
    args.hidden_size = dataset["tf_gene"].shape[0]
    args.tf_gene = dataset["tf_gene"]

    model = CITRUS(args)  # initialize CITRUS model
    model.uuid = uuid.uuid1()
    model.idd = idd
    model.build(device=device)  # build CITRUS model
    model.to(device)
    model.verbose = False
    print(f'Training {idd} initiated for model uuid: {model.uuid}')
    
    model.fit(
        train_set,
        test_set,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        max_iter=args.max_iter,
        max_fscore=args.max_fscore,
        test_inc_size=args.test_inc_size,
    )

    labels, preds, _, _, _, _, _ = model.test(
        test_set, test_batch_size=args.test_batch_size)
    
    model.eval()
    
    preds, tf, hid_tmr, tf, _, _  = model.forward(
            torch.tensor(test_set['sga']), 
            torch.from_numpy(test_set['can'])
        )
    
    genes_ = test_set['gep'].shape[1]
    test_df = pd.DataFrame(np.concatenate([test_set['gep'], 
                                           test_set['can'], 
                                           preds.detach().cpu().numpy()], axis=1))

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
            
    # # print('')
    # # print(pd.DataFrame(np.column_stack([_corrs, _mses]), index=o, columns=['CORR', 'MSE']))
    
    model.performance = np.column_stack([_corrs, _mses])
    model.cancers = o
    
    if model.pval_corr > 0.3:
        model.save_model(os.path.join(args.output_dir, f'model_random_{model.uuid}.pth'))
    
    
    labels, preds, hid_tmr, emb_tmr, emb_sga, attn_wt, tmr = model.test(
        dataset, test_batch_size=args.test_batch_size)
    
    # labels_test, preds_test, _, _, _, _, tmr_test = model.test(
    #     dataset_test, test_batch_size=args.test_batch_size)
    
    # gene_emb = model.layer_sga_emb.weight.data.cpu().numpy()
    dataset_out = {}
    # dataset_out = {
    #     "labels": labels,         # measured exp 
    #     "preds": preds,           # predicted exp
    #     "hid_tmr": hid_tmr,            # TF activity
    #     "emb_tmr": emb_tmr,       # tumor embedding
    #     "tmr": tmr,               # tumor list
    #     "emb_sga": emb_sga,       # straitified tumor embedding
    #     "attn_wt": attn_wt,       # attention weight
    #     "can": dataset["can"],    # cancer type list
    #     "gene_emb": gene_emb,     # gene embedding
    #     "tf_gene": model.layer_w_2.weight.data.cpu().numpy(),  # trained weight of tf_gene constrains
    #     "labels_test": labels_test,      # measured exp on test set
    #     "preds_test": preds_test,        # predicted exp on test set
    #     "tmr_test": tmr_test,            # tumor list on test set
    #     "can_test": dataset_test["can"]  # cancer type list on test set
    # }


    # with open(os.path.join(args.output_dir, "output_{}.pkl".format(model.idd)), "wb") as f:
    #     pickle.dump(dataset_out, f, protocol=2)
    
    
    return dataset_out, model.idd, model.pval_corr, checkCorrelations(labels, preds, return_value=True)


remaining_ids = []
hyperparameters_mapping = {}

for i in range(num_evaluations):
    hyperparameters = generate_hyperparameters()
    accuracy_id = fractured_universe.remote(hyperparameters, i)
    remaining_ids.append(accuracy_id)
    hyperparameters_mapping[accuracy_id] = hyperparameters
    
    
# Fetch and print the results of the tasks in the order that they complete.
while remaining_ids:
    # Use ray.wait to get the object ref of the first task that completes.
    done_ids, remaining_ids = ray.wait(remaining_ids)
    # There is only one return result by default.
    result_id = done_ids[0]

    hyperparameters = hyperparameters_mapping[result_id]
    dataset_out, idd, pval_corr, accuracy = ray.get(result_id)
    
    print(f'model {idd}: {accuracy:.4f} | {pval_corr:.4f}')

exit()

