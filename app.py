from pathlib import Path
import pandas as pd
import numpy as sns
import streamlit as st

hallmark = pd.read_csv('hallmark.csv')


st.set_page_config(
    page_title='CITRUS+',
    page_icon="🍋",
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)


st.markdown('### CITRUS+ 🍋')


# from utils import Data, get_ppi_edge_list

# data_csv = Data(
#     fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
#     fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
#     fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
#     fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv',
# )

# ppi = pd.DataFrame(get_ppi_edge_list(sparse=False)[:, :2], columns=['A', 'B'])
# tf_ppi = ppi[ppi.A.isin(data_csv.tf) | ppi.B.isin(data_csv.tf)]

# tfs = pd.DataFrame(data_csv.tf)
# tfs.columns = ['tf']
# tfs['interacts_with'] = tfs.tf.apply(lambda x: set(tf_ppi[(tf_ppi==x).any(axis=1)].values.reshape(-1)))

import os
import argparse
from utils import bool_ext, load_dataset, split_dataset, evaluate, checkCorrelations
from models import CITRUS
import pickle
import torch
import numpy as np
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
    "--embedding_size",
    help="embedding dimension of genes and tumors",
    type=int,
    default=256,
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
    default=16
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

args = parser.parse_args([])

# @st.cache
# def load_data():
#     dataset, dataset_test = load_dataset(
#         input_dir=args.input_dir,
#         mask01=args.mask01,
#         dataset_name=args.dataset_name,
#         gep_normalization=args.gep_normalization,
#     )

#     train_set, test_set = split_dataset(dataset, ratio=0.66)
    
#     return dataset, train_set, test_set

# with st.spinner('Loading dataset'):
    
#     dataset, train_set, test_set = load_data()

#     args.can_size = dataset["can"].max()  # cancer type dimension
#     args.sga_size = dataset["sga"].max()  # SGA dimension
#     args.gep_size = dataset["gep"].shape[1]  # GEP output dimension
#     args.num_max_sga = dataset["sga"].shape[1]  # maximum number of SGAs in a tumor

#     args.hidden_size = dataset["tf_gene"].shape[0]
#     args.tf_gene = dataset["tf_gene"]


args.can_size = 17
args.sga_size = 11998
args.gep_size = 5541
args.num_max_sga = 1396
args.hidden_size = 320
args.tf_gene = np.load('tf_gene.npy')



saved_models = [i.name for i in Path('./output').glob('*.pth')]


st.image('./CITRUS.png')

model_choice = st.selectbox(f'Choose model', saved_models)

with st.spinner('Loading PyTorch model'):
    model = CITRUS(args)  # initialize CITRUS model
    model.build(device=device)  # build CITRUS model
    model.to(device);

    model.load_state_dict(torch.load(f'./output/{model_choice}', 
                        map_location=torch.device('cpu')))
    model.eval()

st.markdown('#### Model Summary')
st.code(model)


# from utils import Data
from scipy.stats import ttest_ind

# @st.cache
# def load_data_csv():
#     return Data(
#         fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
#         fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
#         fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
#         fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv',
#     )

# data = load_data_csv()
# d = data.cancerType_sga.loc[dataset['tmr']]
# d['index'] = dataset['can'].reshape(-1)

# xdf = pd.DataFrame(enumerate(dataset['tmr']))
# xdf.columns = ['idx', 'id']

# brca = pd.DataFrame(data.sga_sga.loc[data.cancerType_sga[data.cancerType_sga['type']=='BRCA'].index])
# # brca = pd.DataFrame(data.sga_sga.loc[data.cancerType_sga.index])

# is_mutated_gene = 'PIK3CA'

# wt = brca[(brca[f'SM_{is_mutated_gene}'] == 0) & (brca[f'SCNA_{is_mutated_gene}'] == 0)]
# sm_mut = brca[(brca[f'SM_{is_mutated_gene}'] == 1) & (brca[f'SCNA_{is_mutated_gene}'] == 0)]
# scna_mut = brca[(brca[f'SM_{is_mutated_gene}'] == 0) & (brca[f'SCNA_{is_mutated_gene}'] == 1)]
# sm_scna_mut = brca[(brca[f'SM_{is_mutated_gene}'] == 1) & (brca[f'SCNA_{is_mutated_gene}'] == 1)]
# # wt.shape, sm_mut.shape, scna_mut.shape, sm_scna_mut.shape

# wt.to_parquet('wt.parquet', index=None)
# sm_mut.to_parquet('sm_mut.parquet', index=None)

wt = pd.read_parquet('wt.parquet')
sm_mut = pd.read_parquet('sm_mut.parquet')

# xdf.to_parquet('xdf.parquet', index=None)
# np.save('sga.npy', dataset['sga'])
# np.save('can.npy', dataset['can'])

import gzip
f = gzip.GzipFile('sga.npy.gz', 'r')
sga = np.load(f)
f.close()

g = gzip.GzipFile('can.npy.gz', 'r')
can = np.load(g)
g.close()

xdf = pd.read_parquet('xdf.parquet')

idx = xdf[xdf.id.isin(wt.index)].idx.values
X = torch.from_numpy(sga)[idx]
C = torch.from_numpy(can)[idx]
r = model(X, C, pathways=True).data.numpy()

idx = xdf[xdf.id.isin(sm_mut.index)].idx.values
X = torch.from_numpy(sga)[idx]
C = torch.from_numpy(can[idx])
s = model(X, C, pathways=True).data.numpy()


p_predicted = pd.DataFrame(ttest_ind(r, s).pvalue, 
        index=hallmark.Description, 
        columns=['pvalue']).sort_values(by='pvalue', ascending=True).loc[hallmark.Description].pvalue.values


p_exp = hallmark['pvalue'].values

p_predicted = -1*np.log10(p_predicted)
p_exp = -1*np.log10(p_exp)


results = pd.DataFrame([p_exp, p_predicted]).T
results.columns = ['-log10 (MCF10A pvalue)', '-log10 (CITRUS+ pvalue)']
results.index = hallmark.Description

st.markdown('#### MCF10A Data')
st.caption('Sorted by pvalue')
st.dataframe(hallmark[['Description', 'pvalue', 'qvalues', 'p.adjust']])


results['desc'] = results.index

import plotly.express as px
from scipy.stats import spearmanr

st.markdown('#### CITRUS+ versus MCF10A data')
st.caption('Hover on data to see pathway names')
with st.spinner('Plotting pvalues...'):

    fig = px.scatter(results, title=f'spearmanr: {spearmanr(p_predicted, p_exp)}',
        x='-log10 (MCF10A pvalue)', 
        y='-log10 (CITRUS+ pvalue)', hover_data=['desc'])
    
    st.plotly_chart(fig)
    
with st.expander('Raw values'):
    st.markdown('#### CITRUS+ Results')
    st.dataframe(results.drop('desc', axis=1))
    
pathway = st.selectbox('View genes in pathway', hallmark.Description.str[9:], len(hallmark)-3)

st.write(f'Genes in {pathway}')
st.write(hallmark[hallmark.Description==f'HALLMARK_{pathway}']['core_enrichment'].values[0].split('/'))