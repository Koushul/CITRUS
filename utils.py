#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author : "Yifeng Tao", "Xiaojun Ma"
# Last update: March 2021
# =============================================================================
""" 
Shared utilities for models.py and test_run.py.

"""
from collections import defaultdict
import os
import random
import numpy as np
import pandas as pd
import pickle
from typing import Union

import torch
from torch.autograd import Variable
import torch.nn as nn

from scipy import stats
from sklearn.preprocessing import normalize, scale
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tqdm import tqdm
import networkx as nx
from networkx.algorithms.components import connected_components


# from protclus import COACH, DPCLUS, MCODE

import logging

class CustomFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = '\x1b[38;21m'
    blue = '\x1b[38;5;39m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self):
        super().__init__()
        self.fmt = '(%(filename)s : %(lineno)d) - %(levelname)8s | %(message)s'
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
        
logger = logging.getLogger("__name__")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
cfh = CustomFormatter()
ch.setFormatter(cfh)
logger.addHandler(ch)


def bool_ext(rbool):
    """Solve the problem that raw bool type is always True.

    Parameters
    ----------
    rbool: str
      should be True of False.

    """

    if rbool not in ["True", "False"]:
        raise ValueError("Not a valid boolean string")
    return rbool == "True"


def load_dataset(
    input_dir="data", mask01=False, dataset_name="", gep_normalization="scaleRow"
):

    # load dataset
    data = pickle.load(
        open(os.path.join(input_dir, "{}.pkl".format(dataset_name)), "rb")
    )
    can_r = data["can"]  # cancer type index of tumors: list of int
    sga_r = data["sga"]  # SGA index of tumors: list of list
    gep = data["gep"]  # gep matrix of tumors: continuous data
    tmr = data["tmr"]  # barcodes of tumors: list of str
    tf_gene = np.array(data["tf_gene"])

    #load holdout dataset
    can_r_test = data["can_test"] # cancer type index of tumors: list of int
    sga_r_test = data["sga_test"] # SGA index of tumors: list of list
    gep_test = data["gep_test"]   # GEP matrix of tumors: continuous data
    tmr_test = data["tmr_test"]   # barcodes of tumors: list of str

    if mask01 == True:
        tf_gene[tf_gene != 0] = 1
    else:
        tf_gene = normalize(tf_gene)

    # shift the index of cancer type by +1, 0 is for padding
    can = np.asarray([[x + 1] for x in can_r], dtype=int)

    # shift the index of SGAs by +1, 0 is for padding
    num_max_sga_train = max([len(s) for s in sga_r])
    num_max_sga_test = max([len(s) for s in sga_r_test])
    num_max_sga = max(num_max_sga_train, num_max_sga_test)
    sga = np.zeros( (len(sga_r), num_max_sga), dtype=int )
    for idx, line in enumerate(sga_r):
        line = [s+1 for s in line]
        sga[idx,0:len(line)] = line

    
    if gep_normalization == "scaleRow":
        gep = scale(gep, axis=1)


    # shift the index of cancer type by +1, 0 is for padding
    can_test = np.asarray([[x+1] for x in can_r_test], dtype=int)

    # shift the index of SGAs by +1, 0 is for padding

    sga_test = np.zeros( (len(sga_r_test), num_max_sga), dtype=int )
    for idx, line in enumerate(sga_r_test):
        line = [s+1 for s in line]
        sga_test[idx,0:len(line)] = line  
  
    if gep_normalization == "scaleRow":
        gep_test = scale(gep_test, axis = 1)   
  
    dataset = {"can":can, "sga":sga, "gep":gep, "tmr":tmr, "tf_gene":tf_gene}
    dataset_test = {"can":can_test, "sga":sga_test, "gep":gep_test, "tmr":tmr_test}
    
    return dataset, dataset_test

    
def split_dataset(dataset: dict, ratio: float = 0.66):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.34)  # , random_state=2020)
    X = dataset["sga"]
    y = dataset["can"]

    train_set = {}
    test_set = {}
    for train_index, test_index in sss.split(X, y):  # it only contains one element

        train_set = {
            "sga": dataset["sga"][train_index],
            "can": dataset["can"][train_index],
            "gep": dataset["gep"][train_index],
            "tmr": [dataset["tmr"][idx] for idx in train_index],
        }
        test_set = {
            "sga": dataset["sga"][test_index],
            "can": dataset["can"][test_index],
            "gep": dataset["gep"][test_index],
            "tmr": [dataset["tmr"][idx] for idx in test_index],
        }
    return train_set, test_set


def shuffle_data(dataset):
    rng = list(range(len(dataset["can"])))
    random.Random(2020).shuffle(rng)
    dataset["can"] = dataset["can"][rng]
    dataset["sga"] = dataset["sga"][rng]
    dataset["gep"] = dataset["gep"][rng]
    dataset["tmr"] = [dataset["tmr"][idx] for idx in rng]
    return dataset

def shuffle_tensor(t: torch.Tensor) -> torch.Tensor:
    """Shuffle a tensor in all available dimensions"""
    _idx = torch.randperm(t.nelement())
    t = t.view(-1)[_idx].view(t.size())
    return t

def wrap_dataset(dataset):
    ## FIXME
    """Wrap default numpy or list data into PyTorch variables."""
    dataset["can"] = Variable(torch.LongTensor(dataset["can"]))
    dataset["sga"] = Variable(torch.LongTensor(dataset["sga"]))
    dataset["gep"] = Variable(torch.FloatTensor(dataset["gep"]))



    return dataset


def get_minibatch(dataset, index, batch_size, batch_type="train"):
    """Get a mini-batch dataset for training or test.

    Parameters
    ----------
    dataset: dict
      dict of lists, including SGAs, cancer types, GEPs, patient barcodes
    index: int
      starting index of current mini-batch
    batch_size: int
    batch_type: str
      batch strategy is slightly different for training and test
      "train": will return to beginning of the queue when `index` out of range
      "test": will not return to beginning of the queue when `index` out of range

    Returns
    -------
    batch_dataset: dict
      a mini-batch of the input `dataset`.

    """

    sga = dataset["sga"]
    can = dataset["can"]
    gep = dataset["gep"]
    tmr = dataset["tmr"]

    if batch_type == "train":
        batch_sga = [sga[idx % len(sga)] for idx in range(index, index + batch_size)]
        batch_can = [can[idx % len(can)] for idx in range(index, index + batch_size)]
        batch_gep = [gep[idx % len(gep)] for idx in range(index, index + batch_size)]
        batch_tmr = [tmr[idx % len(tmr)] for idx in range(index, index + batch_size)]
    elif batch_type == "test":
        batch_sga = sga[index : index + batch_size]
        batch_can = can[index : index + batch_size]
        batch_gep = gep[index : index + batch_size]
        batch_tmr = tmr[index : index + batch_size]
    batch_dataset_in = {
        "sga": batch_sga,
        "can": batch_can,
        "gep": batch_gep,
        "tmr": batch_tmr,
    }

    batch_dataset = wrap_dataset(batch_dataset_in)
    return batch_dataset


def evaluate(labels, preds, epsilon=1e-4):
    """Calculate performance metrics given ground truths and prediction results.

    Parameters
    ----------
    labels: matrix of 0/1
      ground truth labels
    preds: matrix of float in [0,1]
      predicted labels
    epsilon: float
      a small Laplacian smoothing term to avoid zero denominator

    Returns
    -------
    precision: float
    recall: float
    f1score: float
    accuracy: float

    """

    flat_labels = np.reshape(labels, -1)
    flat_preds = np.reshape(preds, -1)

    corr_spearman = stats.spearmanr(flat_preds, flat_labels)[0]
    corr_pearson = stats.pearsonr(flat_preds, flat_labels)[0]
    return (corr_spearman, corr_pearson)


def checkCorrelations(labels, preds):
    corr_row_pearson = 0
    corr_row_spearman = 0
    corr_col_pearson = 0
    corr_col_spearman = 0
    nsample = labels.shape[0]
    ngene = labels.shape[1]

    for i in range(nsample):
        corr_row_pearson += stats.pearsonr(preds[i, :], labels[i, :])[0]
        corr_row_spearman += stats.spearmanr(preds[i, :], labels[i, :])[0]
    corr_row_pearson = corr_row_pearson / nsample
    corr_row_spearman = corr_row_spearman / nsample

    print(
        "spearman sample mean: %.3f, pearson sample mean: %.3f"
        % (corr_row_spearman, corr_row_pearson)
    )

    for j in range(ngene):
        corr_col_pearson += stats.pearsonr(preds[:, j], labels[:, j])[0]
        corr_col_spearman += stats.spearmanr(preds[:, j], labels[:, j])[0]
    corr_col_pearson = corr_col_pearson / ngene
    corr_col_spearman = corr_col_spearman / ngene

    print(
        "spearman gene mean: %.3f, pearson gene mean: %.3f"
        % (corr_col_spearman, corr_col_pearson)
    )


class EarlyStopping(object):
    def __init__(self, mode="max", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False
        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            return True
        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)

def get_ppi_edge_list(signor_file: str = './data/SIGNOR_PPI.tsv') -> list:
    """
    Generate a list of all edges from the SIGNOR Protein-Protein interaction graph
    SIGNOR tsv download: https://signor.uniroma2.it/downloads.php
    """

    ## TODO: Add filtering options for score thresholds and interaction types

    ppi = pd.read_csv(signor_file, sep='\t')
    edges = ppi.query('TYPEA == "protein" or TYPEB == "protein"')[['ENTITYA', 'ENTITYB', 'SCORE']].dropna()
    edges = edges.loc[(~edges.ENTITYA.str.contains('"'))] #filter out non gene entries 
    logger.info('Loaded %i edges from the ppi' % len(edges))
    

    return edges.values


def get_overlap_network(gene_symbols: list, weighted: bool = True) -> nx.Graph:
    """
    Return a subgraph of relevant genes induced from the Protein-Protein Interaction Network
    Code adapted from https://github.com/paulmorio/gincco
    """

    logger.info('Using induced overlap network')

    ppi_edge_list = get_ppi_edge_list()
    sga_ppi_edge_list = []

    for edge in ppi_edge_list:
        if edge[0] in gene_symbols and edge[1] in gene_symbols:
            sga_ppi_edge_list.append(edge)

    G = nx.Graph()

    if weighted:
        G.add_weighted_edges_from(sga_ppi_edge_list)
    else:
        G.add_edges_from(sga_ppi_edge_list)

    return G


def get_connected_sga_ppi_network(gene_symbols: list, weighted: bool=True) -> nx.Graph:
    """Returns a NetworkX graph of the largest connected PPI network at the 
    intersection of the give gene set and the SIGNOR (human) PPI network.

    Code adapted from https://github.com/paulmorio/gincco
    """

    logger.info('Using largest connected PPI network')

    ppi_edge_list = get_ppi_edge_list()
    sga_ppi_edge_list = []

    ## find overlapping nodes in SGA and PPI
    for edge in ppi_edge_list:
        if edge[0] in gene_symbols and edge[1] in gene_symbols:
            sga_ppi_edge_list.append(edge)

    G = nx.Graph()
    
    if weighted:        
        G.add_weighted_edges_from(sga_ppi_edge_list)
        largest_cc = max(connected_components(G), key=len)
        largest_cc = G.subgraph(largest_cc).copy()
    
    else:
        G.add_edges_from(sga_ppi_edge_list)
        largest_cc = max(connected_components(G), key=len)
        largest_cc = G.subgraph(largest_cc).copy()

    return largest_cc



def generate_mask_from_ppi(sga: pd.DataFrame, clust_algo:str='DPCLUS') -> Union[torch.Tensor, torch.Tensor]:

    """
    Generates a bio-informed mask for connecting 
    somatic alteration input node to the clustered ppi network
    
    """
    genes = set([name.split('_')[1] for name in sga.columns])

    ## Induce a subgraph from the shared genes between the 
    ## alterations and the ppi network

    # G = get_connected_sga_ppi_network(gene_symbols=genes)
    G = get_overlap_network(gene_symbols=genes)
    # nx.write_edgelist(G, "graph.txt", data=False)

    ## Apply the clustering algorithm, creating unique protein clusters
    ## TODO: refactor to avoid writing to disk
    # c = DPCLUS('graph.txt')
    # c.cluster()
    # c.save_clusters('MCODE_clusters.txt')

    logger.info('Using %s clustering algorithm' % clust_algo)

    if clust_algo == 'DPCLUS':
        precomputed_clusters = 'DPCLUS_clusters.txt'
    elif clust_algo == 'COACH':
        precomputed_clusters = 'COACH_clusters.txt'
    elif clust_algo == 'MCODE':
        precomputed_clusters = 'MCODE_clusters.txt'
    else:
        logging.warning('Unknown Clustering Algorithm')
        precomputed_clusters = 'DPCLUS_clusters.txt'

    with open(precomputed_clusters, "r") as fh:
        lines = fh.readlines()
        clusterindex_to_genes = {}
        for i, c in enumerate(lines):
            clustlist = c.strip().split(" ")
            if len(c) == 0:
                continue
            clusterindex_to_genes[i] = clustlist

    ## Create mapping from sga genes to the protein clusters
    gene_to_clusterindices = defaultdict(list)
    for c in clusterindex_to_genes.keys():
        for g in clusterindex_to_genes[c]:
            gene_to_clusterindices[g].append(c)

    index_to_sga = dict(enumerate(sga.columns))
    index_to_genesymbol = dict(enumerate(genes))
    genesymbol_to_index = {}
    for key in index_to_genesymbol.keys():
        if index_to_genesymbol[key] in genesymbol_to_index:
            pass
        else:
            genesymbol_to_index[index_to_genesymbol[key]] = key  

    alteration_to_gene = dict(zip(sga.columns, [i.split('_')[1] for i in sga.columns])) ## 'SM_TUSC3':'TUSC3'


    ## Create mask and weights
    adj = np.zeros((len(sga.columns), len(list(clusterindex_to_genes.keys()))))
    weights = np.zeros_like(adj)

    unmapped_genes = []
    weight_tracker = defaultdict(float) #accumulate edge weights from sga nodes to clusters

    for index, alt in index_to_sga.items():
        gs = alteration_to_gene[index_to_sga[index]] # sga gene
        if gs in gene_to_clusterindices.keys():
            for cluster_index in gene_to_clusterindices[gs]:
                adj[index, cluster_index] = 1
                for g in clusterindex_to_genes[cluster_index]:    
                    weight_tracker[cluster_index] += G.get_edge_data(gs, g, {'weight': 0.0})['weight']
        else:
            unmapped_genes.append(gs)


    logger.info('Generated mask with %i clusters and %i edges' % 
        (len(list(clusterindex_to_genes.keys())), len(sga.columns)-len(unmapped_genes), ))

    biomask = torch.from_numpy(adj).int()
    weights = torch.from_numpy(weights).float()
    

    return biomask, weights


def load_sga(sga_file: str = './data/CITRUS_SGA_SGAseparated.csv') -> pd.DataFrame:
    """
    Returns a DataFrame representing the somatic gene alterations as a binary table
    """
    sga_df = pd.read_csv(sga_file)
    sga_df = sga_df.set_index(sga_df.columns[0])
    sga_df.index.name = None

    return sga_df


class Data:
    """
    Repositiory for all the data required for training and evaluation
    Holds and interconnects all the different data modalities 

    Required files(csv):
        cancerType_SGA
        gene_tf_SGA
        GEP_SGA
        SGA_GEP_TF_SGA
        SGA_SGA
    """

    
    def _read_csv(self, csv:str, use_cache:bool=True):
        pqt_file = csv[:-3] + 'parquet'
        if os.path.isfile(pqt_file) and use_cache:
            df = pd.read_parquet(pqt_file)
        else:
            df = pd.read_csv(csv)
            df = df.set_index(df.columns[0])
            df.index.name = None

            df.to_parquet(pqt_file)

        # df = df.sample(frac=1).reset_index(drop=True)

        return df

    def __init__(self, fcancerType_SGA:str, fgene_tf_SGA:str, fGEP_SGA:str, fSGA_SGA:str):
        logger.info("Loading data files")
            
        self.cancerType_sga = self._read_csv(fcancerType_SGA)
        self.gene_tf_sga = self._read_csv(fgene_tf_SGA)
        self.gep_sga = self._read_csv(fGEP_SGA)
        self.sga_sga = self._read_csv(fSGA_SGA).replace(2, 1)

        assert all(self.gep_sga.index == self.sga_sga.index)
        assert all(self.cancerType_sga.index == self.sga_sga.index)
        
        self.alterations = np.array(self.sga_sga.columns, dtype=str)
        self.tumor_ids = np.array(self.sga_sga.index, dtype=str)
        self.sga_genes = np.unique(np.array([name.split('_')[1] for name in self.alterations], dtype=str))
        self.cancer_types = np.array(self.cancerType_sga['type'].values, dtype=str) #non-unique list


    def get_train_test(self) -> Union[dict, dict]:
        _encoder = {value:key for key, value in dict(enumerate(np.sort(np.unique(self.cancer_types)))).items()}
        encoded_cancer_types = self.cancerType_sga['type'].apply(lambda x: _encoder[x]).values + 1

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.34)
        X = self.sga_sga.values
        y = encoded_cancer_types

        train_set = {}
        test_set = {}
        for train_index, test_index in sss.split(X, y):  # it only contains one element
    
            train_set = {
                "sga": self.sga_sga.values[train_index],
                "can": encoded_cancer_types[train_index],
                "gep": self.gep_sga.values[train_index],
                "tmr": np.array([self.tumor_ids[idx] for idx in train_index]),
            }
            
            test_set = {
                "sga": self.sga_sga.values[test_index],
                "can": encoded_cancer_types[test_index],
                "gep": self.gep_sga.values[test_index],
                "tmr": np.array([self.tumor_ids[idx] for idx in test_index]),
            }

        return train_set, test_set



if __name__ == '__main__':


    # print("Loading dataset...")
    # dataset, dataset_test = load_dataset(
    # input_dir='./data',
    # dataset_name="dataset_CITRUS")

    # print(dataset['sga'].shape)
    # print(dataset['sga'])

    data = Data(
        fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
        fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
        fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
        fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv'
    )

    generate_mask_from_ppi(data.sga_sga)



    # train_test_split()

    # print(dataset['tf_gene'])
    # print(dataset['tf_gene'].shape)


    # print(sga_df.values.shape)
    # print(biomask.shape)
