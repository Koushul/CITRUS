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
st.code(vars(args))


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


st.markdown(f'#### Performance')
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
        y='-log10 (CITRUS+ pvalue)', hover_data=['desc'])
    
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
        y='-log10 (CITRUS+ pvalue)', hover_data=['desc'])
    
    st.plotly_chart(fig)