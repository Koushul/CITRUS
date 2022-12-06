from pathlib import Path
import pandas as pd
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
import yaml

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    device_name = torch.cuda.get_device_name(0)
else:
    device_name = 'cpu'


with open('args.yaml', 'r') as f:
    args_dict = yaml.safe_load(f)

parser = argparse.ArgumentParser()
args = argparse.Namespace(**args_dict)
args.tf_gene = np.load('tf_gene.npy')
  
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



saved_models = [i.name for i in Path('./output').glob('*.pth')]


st.image('./CITRUS.png')

model_choice = st.selectbox(f'Choose model', saved_models)

# with st.spinner('Loading PyTorch model'):
#     model = CITRUS(args)  # initialize CITRUS model
#     model.build(device=device)  # build CITRUS model
#     model.to(device);

#     model.load_state_dict(torch.load(f'./output/{model_choice}', 
#                         map_location=torch.device('cpu')))
#     model.eval()

# st.markdown('#### Model Summary')
# st.code(model)


# from utils import Data
from scipy.stats import ttest_ind


st.markdown('#### MCF10A Data')
st.caption('Sorted by pvalue')
st.dataframe(hallmark[['Description', 'pvalue', 'qvalues', 'p.adjust']])

