# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 17:24:27 2022

@author: XIM33
"""
import pickle
import pandas as pd
from scipy.stats import ttest_ind
import statsmodels.stats.multitest as multi
import sys
import numpy as np

from statsmodels.stats.multitest import fdrcorrection
from scipy import stats
import os

df_SGA = pd.read_csv("data/CITRUS_bfs_fastRun_SGAseparated_TFprefixed_SGA_forSparseNet.csv", index_col=0)
df_type = pd.read_csv("data/CITRUS_canType_SGAseparated.csv", index_col=0)

df_SGA = df_SGA.reindex(df_type.index)
typeSet = sorted(set(df_type['type']))


def get_mut_samples(mutSGA, tp):

    if mutSGA.startswith("SM_") or mutSGA.startswith("SCNA_"):
        mut_samp = df_SGA.index[(df_SGA[mutSGA]!=0) & (df_type['type']==tp)]
    else:
        if mutSGA != "MAP2K4":
            mut_samp = df_SGA.index[((df_SGA['SCNA_'+ mutSGA] != 0) | (df_SGA['SM_' + mutSGA] != 0)) & (df_type['type']==tp)]
        else:
            mut_samp = df_SGA.index[(df_SGA['SM_' + mutSGA] != 0) & (df_type['type']==tp)]
            
    return(mut_samp)


def ttest(df_ori, df_knc):
    t_test = pd.DataFrame(data=0, index = df_ori.columns, columns=["pvalue","log10pval", "pvalue_fdr","diff", 'mean_knc','mean_ori'])
    
    _, pvalues = np.array(stats.mstats.ttest_ind(df_ori, df_knc, axis=0))
    _, pvalues_fdr = fdrcorrection(np.array(pvalues))
    # pvalue_types[tp] = pvalues_fdr
    
    mean_ori = np.mean(df_ori, axis=0)
    mean_knc = np.mean(df_knc, axis=0)
    
    diff = mean_knc - mean_ori
    
    # diff =np.mean( (df_knc_t - df_ori_t), axis=0)
    # epsilon = sys.float_info.epsilon
    t_test['pvalue'] = pvalues
    t_test['log10pval'] = -np.log10(pvalues )
    t_test['pvalue_fdr'] = pvalues_fdr
    t_test['diff'] = diff
    t_test['mean_ori'] = mean_ori
    t_test['mean_knc'] = mean_knc
   
    return(t_test)



######################################################################################
# settting

pure = True

path = "E:\\Xiaojun\\Sparse_Network\\SparseNet\\output_server\\CITRUS_SGAseparated_TFprefixed_new\\iterations_1000_newCode\\"

jobName = "lr0.01_l20.0_100_NonNegWt_leakyRelu/"
# jobName = "lr0.01_l20.0_100_NonNegWt_tanh_SM/"
# jobName = "lr0.01_l20.0_100_NonNegWt_tanh/"

path += jobName

dtype = "Inters"
# dtype = "TFs"

# kncSGAs = ["TP53","PIK3CA","PTEN", 'MAP2K4', "GATA3", "CDH1"]
kncSGAs = ["SM_TP53","SM_PIK3CA","SM_PTEN", 'SM_MAP2K4', "SM_GATA3", "SM_CDH1"]
# kncSGAs = ["PIK3CA"]
# kncSGAs = ["SM_PIK3CA"]

Nrun = 10

# sub = "lr0.01"
# sub = "tanh_NonNegWt_knockup"+str(Nrun)
# sub = "relu_NonNegWtBi_5000"
# sub = "noActivation" + str(Nrun)
# sub = "NonNegWtBi" + str(Nrun)
# sub = "NonNegWtBi_knockup_" + str(Nrun)

sub = "leakyRelu_NonNegWt_Nrun{}".format(str(Nrun))
# sub = "tanh_NonNegWt_Nrun{}_SM".format(str(Nrun))
# sub = "tanh_NonNegWt_Nrun{}".format(str(Nrun))

interPath = './outputs/{0}/{1}/ave/'.format(sub, dtype)
if not os.path.exists(interPath):
   os.makedirs(interPath)
   
outPath = './outputs/{0}/{1}/'.format(sub, dtype)

################################################################################
# # read in data of all runs

df_ori_ave=0
for i in range(1, Nrun+1):
    print (i)
    if dtype == "TFs":
        df_ori = pd.read_csv(path+"run_{}/ori/TF_activity.csv".format(i), index_col=0)
    else:
        df_ori = pd.read_csv(path+"run_{}/ori/interNodes_activity.csv".format(i), index_col=0)
    
    df_ori_ave += df_ori
 
df_ori_ave = df_ori_ave/Nrun
df_ori_ave.to_csv(interPath + 'df_ori_ave.csv', index=True)



df_knc_aves = []
for kncSGA in kncSGAs:
    print ("kncSGA=",kncSGA)
    df_knc_ave=0
    for i in range(1, Nrun+1):
        print("i=",i)
        if dtype == "TFs":
            df_knc = pd.read_csv(path+"run_{}/knockout_{}/TF_activity.csv".format(i, kncSGA), index_col=0)
        else:
            df_knc = pd.read_csv(path+"run_{}/knockout_{}/interNodes_activity.csv".format(i, kncSGA), index_col=0)
        
        df_knc_ave += df_knc
     
    df_knc_ave = df_knc_ave/Nrun
    df_knc_ave.to_csv(interPath + 'df_knc_ave_{}.csv'.format(kncSGA), index=True)
    


## load average

df_ori_ave = pd.read_csv(interPath + 'df_ori_ave.csv', index_col=0)

df_knc_aves = []
for i in range(len(kncSGAs)):
    kncSGA = kncSGAs[i]
    df_knc_aves.append(pd.read_csv(interPath + '/df_knc_ave_{0}.csv'.format(kncSGA), index_col=0))

# typeSet = list(set(df_type['type']))



###############################################

# for tp in typeSet:

for i in range(len(kncSGAs)):
    kncSGA = kncSGAs[i]
   
    for tp in ["BRCA"]:#typeSet:
        
        df_ori_t = df_ori_ave[df_type['type']==tp]
        df_knc_t = df_knc_aves[i][df_type['type']==tp] 
        
        if pure == True:
            
            mut_samp = get_mut_samples(kncSGA, tp)
            
            df_ori_t = df_ori_t.loc[mut_samp]
            df_knc_t = df_knc_t.loc[mut_samp]
    
    
        t_test = ttest(df_ori_t, df_knc_t)
        
        if  pure == True:
            t_test.to_csv(outPath + "/{0}_t_test_{1}_ave_pure.csv".format(tp, kncSGA), index=True)
        else:
            t_test.to_csv(outPath + "/{0}_t_test_{1}_ave.csv".format(tp, kncSGA), index=True)

#internode: 1092
#edges: 8740/231635
#SGA 1867
#TF: 208
#mRAN: 5541

# t_test1 = pd.read_csv(outPath + "/{0}_t_test_{1}_ave.csv".format(tp, "TP53"), index_col=0)
# t_test2 = pd.read_csv(outPath + "/{0}_t_test_{1}_ave.csv".format(tp, "PIK3CA"), index_col=0)
