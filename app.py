from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.stats import spearmanr
from scipy.stats import kruskal

hallmark = pd.read_csv('hallmark.csv')


st.set_page_config(
    page_title='CITRUS+',
    page_icon="🍋",
    # layout="wide",
    # initial_sidebar_state="expanded",
    menu_items={}
)



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


import datetime

st.markdown(f'## Research Progress: 🍋 CITRUS+ ')
st.caption(str(datetime.datetime.now()))

with st.sidebar:
    page = st.radio('Page', 
        ['Intro', 'Pathways', 'Training', 'Performance', 'PCA Analysis', 'Mutant vs Wildtype', 'PIK3CA Analysis', 'HPV+/HPV- Analysis', 'NFE2L2 Analysis']
    
    )

st.markdown('#### '+ page)
if page == 'Intro':
    st.write('This work builds on top of our previous paper (CITRUS)')
    st.image('paper.png')

    st.markdown('---')
    st.image('ci.jpeg')

elif page == 'Pathways':
    st.write('CITRUS+ incorporates biological pathway information')
    st.image('./CITRUS.png')


    with st.expander('Show PyTorch model implementation'):
        st.code("""
        def forward(self, sga_index, can_index):                    
            # cancer type embedding
            emb_can = self.layer_can_emb(can_index)
            emb_can = emb_can.view(-1, self.embedding_size)
            
            E_t = self.layer_sga_emb(sga_index)

            # squeeze and tanh-curve the gene embeddings
            E_t_flatten = E_t.view(-1, self.embedding_size)
            E_t1_flatten = torch.tanh(self.layer_w_0(E_t_flatten))
            
            # multiplied by attention heads
            E_t2_flatten = self.layer_beta(E_t1_flatten)
            E_t2 = E_t2_flatten.view(-1, self.num_max_sga, self.attention_head)

            # normalize by softmax
            E_t2 = E_t2.permute(1, 0, 2)
            A = F.softmax(E_t2, dim=0)
            A = A.permute(1, 0, 2)
            
            # multi-head attention weighted sga embedding:
            emb_sga = torch.sum(torch.bmm(A.permute(0, 2, 1), E_t), dim=1)
            emb_sga = emb_sga.view(-1, self.embedding_size)

            # add cancer type embedding
            emb_tmr = emb_can + emb_sga
   
        """)

        if st.button('Load Model'):
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

    st.markdown('---')

    st.markdown(
        """
        Issues during training:
        - Gene expression values are highly correlated
        - Training is unstable
        - High sparsity of connections between Pathway and TF layer
        - Interpretation of trained models
        """
    )

    with st.expander('Show DropConnect implementation'):
        st.code("""
mask = torch.from_numpy(self.mask).float().to(device)
self.pathways.weight.data = self.pathways.weight.data * mask.T
self.pathways.weight.register_hook(lambda grad: grad.mul_(mask.T))
        """)


if page == 'Training':
    st.image('https://github.com/ray-project/ray/raw/master/doc/source/images/ray_header_logo.png')

    st.code("""
import ray

@ray.remote(num_gpus=0.25)
def train_model(args, idd):
    dataset, dataset_test = load_dataset()
    train_set, test_set = split_dataset(dataset, ratio=0.66)
    model = CITRUS(args)
    model.uuid = uuid.uuid1()
    model.to(device)
    model.fit(
        train_set,
        test_set,
    )

    return model.performance



remaining_ids = []
hyperparameters_mapping = {}

for i in range(num_evaluations):
    hyperparameters = generate_hyperparameters()
    model_id = train_model.remote(hyperparameters, i)
    remaining_ids.append(model_id)
    hyperparameters_mapping[model_id] = hyperparameters
    
    
# Fetch and print the results of the tasks in the order that they complete.
while remaining_ids:
    done_ids, remaining_ids = ray.wait(remaining_ids)
    result_id = done_ids[0]
    hyperparameters = hyperparameters_mapping[result_id]
    performance = ray.get(result_id)
    """)
    

    with st.expander('View all hyperparameters'):
        st.write(vars(args))


if page == 'Performance':

    args.tf_gene = []

        
    st.title(f'Models in ensemble: {10}')    
    # st.write(saved_models)


    del args

    a, b, c = st.columns(3)



    with a:
        st.markdown(f'#### Mean Predicted Expression')
        st.table(pd.read_csv('perf.csv').set_index('Unnamed: 0'))

    with b:
        st.markdown(f'#### Shuffle Within Cancer Type')
        st.table(pd.read_csv('perf_shuffled_within.csv').set_index('Unnamed: 0'))

    with c:
        st.markdown(f'#### Shuffle Across All Samples')
        st.table(pd.read_csv('perf_shuffled_across.csv').set_index('Unnamed: 0'))

    st.markdown('----')

    with st.expander('Compare with CITRUS'):
        st.image('citrus_validation.png')


if page == 'PCA Analysis':
    # f, g = st.columns(2)

    st.markdown('### PCA - Transcription Factors')
    st.image('./tf.JPG')

    st.markdown('----')

    st.markdown('### PCA - Pathways')
    st.image('./pathway.JPG')


if page == 'PIK3CA Analysis':
    p_predicted = np.load('p_predicted.npy')
    p_exp = np.load('p_exp.npy')

    p_predicted = -np.log(p_predicted) 
    p_exp = -np.log(p_exp) 

    results = pd.DataFrame([p_exp, p_predicted]).T

    results.columns = ['-log10 (MCF10A pvalue)', '-log10 (CITRUS+ pvalue)']
    results.index = hallmark.Description
    results['desc'] = results.index

    w, y = st.columns(2)

    with w:
        with st.spinner('Plotting pvalues...'):
            st.markdown('#### PI3K_Activation')
            st.caption('Hover on data to see pathway names')
            fig = px.scatter(results, title=f'spearmanr: {spearmanr(p_predicted, p_exp).correlation:.5f}',
                x='-log10 (MCF10A pvalue)', 
                y='-log10 (CITRUS+ pvalue)', hover_data=['desc'], height = 600, width=600)
            
            st.plotly_chart(fig)
        
        
        
    p_predicted = np.load('p_predicted2.npy')
    p_exp = np.load('p_exp2.npy')

    p_predicted = -np.log(p_predicted) 
    p_exp = -np.log(p_exp) 

    results = pd.DataFrame([p_exp, p_predicted]).T

    results.columns = ['-log10 (MCF10A pvalue)', '-log10 (CITRUS+ pvalue)']
    results.index = hallmark.Description
    results['desc'] = results.index

    with y:
        with st.spinner('Plotting pvalues...'):
            st.markdown('#### PI3K_Inhibition')
            st.caption('Hover on data to see pathway names')
            fig = px.scatter(results, title=f'spearmanr: {spearmanr(p_predicted, p_exp).correlation:.5f}',
                x='-log10 (MCF10A pvalue)', 
                y='-log10 (CITRUS+ pvalue)', hover_data=['desc'], color_discrete_sequence=['red'], height = 600, width=600)
            
            st.plotly_chart(fig)

    st.markdown('----')

if page == 'Mutant vs Wildtype':

    st.markdown('### Inferred activities as a function of mutational status')        
    st.image('./NFE2L2.png')
    st.image('./TP53.png')
    st.image('./E2F1.png')  
        
    st.markdown('----')


if page == 'HPV+/HPV- Analysis':
    st.markdown(f'#### HPV cell line data')
    hpv = pd.read_csv('HPV_analysis.csv')

    normalization = st.selectbox('Nomalization Method', ['Log2RPKM', 'Log2TPM', 'LogCPM'])

    df = pd.read_csv(f'{normalization}_38562g8s.txt', sep='\t', index_col=0)[:-5].dropna()
    st.dataframe(df)

    st.caption('Select two columns to compare')
    aa, bb = st.columns(2)
    col_a = aa.selectbox('Column A', df.columns, 0)
    col_b = bb.selectbox('Column B', df.columns, 4)

    st.caption('Pathways with p-value < 0.05 are shown below')
    st.caption('P-values were computed using Kruskal-Wallis H-test')
    gg = []
    for desc, genes in hallmark[['Description', 'core_enrichment']].values:
        geneset = genes.split('/')
        pval = kruskal(df.reindex(geneset)[col_a].dropna().values, df.reindex(geneset)[col_b].dropna().values).pvalue
        if pval < 0.05:
            st.write(desc + ' ', pval)
            gg.append(desc)


    st.markdown('##### Common pathways between Cell Line Data & CITRUS+: ')
    st.write(set(hpv[hpv.pvalue<0.05].astype(str).Description).intersection(set(gg)))


    st.markdown('----')

    st.markdown(f'#### HPV+ (n=60) vs HPV- (n=314)')
    st.table(hpv[hpv.pvalue<0.05].astype(str))
    st.table(hpv[hpv.pvalue>=0.05].astype(str))



    st.markdown('----')

if page == 'NFE2L2 Analysis':

    taba, tabb, tabc = st.tabs(["Volcano Plots", "Heatmaps", 'Tables'])

    with taba:

        import dash_bio
        cancers = ['LUAD', 'UCEC', 'LIHC', 'KIRC', 'PRAD', 'COAD', 'THCA', 'BRCA',
            'KIRP', 'BLCA', 'LUSC', 'STAD', 'HNSC', 'CESC', 'GBM', 'ESCA',
            'PCPG']

        v = pd.read_csv(f'NFE2L2_BRCA.csv')
        c_choice = st.selectbox('Cancer Type', cancers)
        v = pd.read_csv(f'NFE2L2_{c_choice}.csv')
        st.caption('wilcoxon rank sum test')
        try:
            fig = dash_bio.VolcanoPlot(dataframe=v, 
                        point_size=7,
                        p='P',
                        xlabel=f'Mean Difference',
                        # ylabel=f'-log10(})',
                        title=f'Volcano plot for {c_choice}'
                    )
            st.plotly_chart(fig)
        except:
            st.error('Not enough mutant samples')

    with tabb:
        st.title('Fraction of significant pathways | CITRUS vs Null model')
        st.image('heatmap.png')

        st.title('Fraction of significant TFs | CITRUS vs Null model')

        st.image('heatmap_tf.png')

    with tabc: 
        st.markdown(f'#### NFR2L2 Mutants vs Wildtypes')
        st.table(pd.read_csv('./NFE2L2.csv').astype(str))


