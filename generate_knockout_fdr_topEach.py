import pickle
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

df_type = pd.read_csv("data/CITRUS_canType_SGAseparated.csv", index_col=0)
typeSet = sorted(set(df_type['type']))

# kncSGAs = ['PIK3CA', 'PTEN', 'MAP2K4','GATA3','TP53', 'CDH1' ]
kncSGAs = ["SM_TP53","SM_PIK3CA","SM_PTEN", 'SM_MAP2K4', "SM_GATA3", "SM_CDH1"]
# knockoutSGAs_cans['UCEC'] =  ['SM_PIK3CA', 'SM_PTEN', 'SM_KRAS', 'SM_TP53',  'SM_CTNNB1' ]

# dtype = "TFs"
dtype = "Inters"

Nrun = 10

sub = "leakyRelu_NonNegWt_Nrun" + str(Nrun)

path = './outputs/{0}/{1}/ave/'.format(sub, dtype)

df_ori = pd.read_csv(path + 'df_ori_ave.csv', index_col=0)
    
genes = df_ori.columns

for can in ["BRCA"]:   

    df_ori_can = df_ori.loc[df_type['type']==can,:]
    
    values_diff={}
    for kncSGA in kncSGAs:
        df_knc = pd.read_csv(path+'/df_knc_ave_{0}.csv'.format(kncSGA), index_col=0)

        df_knc_can = df_knc.loc[df_type['type']==can,:]
        
       
        diffs = np.array(np.mean((df_knc_can - df_ori_can), axis=0))
        
        _, pvalues = np.array(stats.mstats.ttest_ind(df_ori_can, df_knc_can, axis=0))
        _, pvalues_fdr = fdrcorrection(np.array(pvalues))


        values_diff[kncSGA] = [pvalues, diffs]  #0:pvalues, 1: diffs
     
    
    topN = 12
    commidxs = set()
    for kncSGA in kncSGAs:
        pvalues = values_diff[kncSGA][0] #pvalues
        topidxs = np.argsort(pvalues)[:topN]
        # diffs = values_SGAs[SGA][1] #diff
        # topidxs = np.argsort(np.abs(diffs))[::-1][:topN]        
        
        commidxs = set(commidxs) | set(topidxs)
    
    commidxs = list(commidxs)
 
    topgenes = np.array(genes)[commidxs]
    
    
    
    df_values = pd.DataFrame()
    for kncSGA in kncSGAs:
         df_value = pd.DataFrame()
         df_value['pvalue'] = values_diff[kncSGA][0][commidxs]
         df_value['log10pval'] = -np.log10(df_value['pvalue'])
         df_value['diff'] = values_diff[kncSGA][1][commidxs]
         df_value['protein'] = topgenes
         df_value['kncSGA'] = kncSGA
         
         # df_values = df_values.append(df_value, ignore_index=True)
         df_values = pd.concat([df_values,df_value],axis=0)
   
   
    adj=[]                  
    for v in df_values['log10pval'].values:
        if v > -np.log10(0.0001):
            adj.append(6)
        elif v > -np.log10(0.001):
            adj.append(4)
        elif v > -np.log10(0.01):   
            adj.append(2)
        elif v > -np.log10(0.5): 
            adj.append(0)
        else:
            adj.append(0)
             
    df_values['adj'] = adj      
    
     
    df_values.to_csv("outputs/Fig5/{}_{}_pvalue_top12_eachKnc.csv".format(can, dtype), index=False)
        
    