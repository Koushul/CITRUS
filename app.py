from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import spearmanr

hallmark = pd.read_csv('hallmark.csv')


st.set_page_config(
    page_title='CITRUS+',
    page_icon="🍋",
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)


st.markdown('## 🍋 CITRUS+ ')


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
# from models import CITRUS
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


saved_models = [i.name for i in Path('./output').glob('*.pth')]

st.image('./CITRUS.png')
args.tf_gene = []
with st.expander('View all hyperparameters'):
    st.write(vars(args))
    
st.title(f'Models in ensemble: {10}')    

st.code("""CITRUS(
  (layer_sga_emb): Embedding(11999, 256, padding_idx=0)
  (layer_can_emb): Embedding(18, 256, padding_idx=0)
  (layer_w_0): Linear(in_features=256, out_features=256, bias=True)
  (layer_beta): Linear(in_features=256, out_features=16, bias=True)
  (layer_dropout_1): Dropout(p=0.2, inplace=False)
  (layer_dropout_2): Dropout(p=0.2, inplace=False)
  (layer_dropout_3): Dropout(p=0.5, inplace=False)
  (layer_w_1): Linear(in_features=256, out_features=50, bias=True)
  (bnorm_pathways): BatchNorm1d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (pathways): Linear(in_features=50, out_features=320, bias=True)
  (bnorm_tf): BatchNorm1d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer_w_2): Linear(in_features=320, out_features=5541, bias=True)
  (loss): MSELoss()
)""")


# model_choice = st.selectbox(f'Choose model', saved_models)

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
# from scipy.stats import ttest_ind





# st.markdown('#### MCF10A Data')
# st.caption('Sorted by pvalue')
# st.dataframe(hallmark[['Description', 'pvalue', 'qvalues', 'p.adjust']])

a, b, c = st.columns(3)

with a:
    st.markdown(f'#### Mean Predicted Expression')
    st.table(pd.read_csv('perf.csv').set_index('Unnamed: 0'))

with b:
    st.markdown(f'#### Random Sample Within Cancer Type')
    st.table(pd.read_csv('perf_shuffled_within.csv').set_index('Unnamed: 0'))

with c:
    st.markdown(f'#### Random Sample Across All Samples')
    st.table(pd.read_csv('perf.csv').set_index('Unnamed: 0'))

p_predicted = np.load('p_predicted.npy')
p_exp = np.load('p_exp.npy')

p_predicted = -np.log(p_predicted) 
p_exp = -np.log(p_exp) 

results = pd.DataFrame([p_exp, p_predicted]).T

results.columns = ['-log10 (MCF10A pvalue)', '-log10 (CITRUS+ pvalue)']
results.index = hallmark.Description
results['desc'] = results.index


with st.spinner('Plotting pvalues...'):
    st.markdown('#### CITRUS+ versus MCF10A_hallmark_PI3K_Activation')
    st.caption('Hover on data to see pathway names')
    fig = px.scatter(results, title=f'spearmanr: {spearmanr(p_predicted, p_exp).correlation:.5f}',
        x='-log10 (MCF10A pvalue)', 
        y='-log10 (CITRUS+ pvalue)', hover_data=['desc'], height = 800, width=800)
    
    st.plotly_chart(fig)
    
    
    
p_predicted = np.load('p_predicted2.npy')
p_exp = np.load('p_exp2.npy')

p_predicted = -np.log(p_predicted) 
p_exp = -np.log(p_exp) 

results = pd.DataFrame([p_exp, p_predicted]).T

results.columns = ['-log10 (MCF10A pvalue)', '-log10 (CITRUS+ pvalue)']
results.index = hallmark.Description
results['desc'] = results.index
    
with st.spinner('Plotting pvalues...'):
    st.markdown('#### CITRUS+ versus MCF10A_hallmark_PI3K_Inhibition')
    st.caption('Hover on data to see pathway names')
    fig = px.scatter(results, title=f'spearmanr: {spearmanr(p_predicted, p_exp).correlation:.5f}',
        x='-log10 (MCF10A pvalue)', 
        y='-log10 (CITRUS+ pvalue)', hover_data=['desc'], color_discrete_sequence=['red'], height = 800, width=800)
    
    st.plotly_chart(fig)