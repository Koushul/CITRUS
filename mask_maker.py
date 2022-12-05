import pandas as pd
import numpy as np
from tqdm import tqdm

hallmark = pd.read_excel('/ihome/hosmanbeyoglu/kor11/tools/CITRUS/FW__MCF10A_wild_type_and_PIK3CA_H1047R_knock-in_cell_lines/Supplementary Table S4.xlsx', 
    sheet_name='MCF10A_hallmark_PI3K_Inhibition')

hallmark = hallmark[['Description', 'core_enrichment']]

from utils import Data, get_ppi_edge_list

data_csv = Data(
    fGEP_SGA = 'data/CITRUS_GEP_SGAseparated.csv',
    fgene_tf_SGA = 'data/CITRUS_gene_tf_SGAseparated.csv',
    fcancerType_SGA = 'data/CITRUS_canType_SGAseparated.csv',
    fSGA_SGA = 'data/CITRUS_SGA_SGAseparated.csv',
)

ppi = pd.DataFrame(get_ppi_edge_list(sparse=False)[:, :2], columns=['A', 'B'])



hallmark_mask = np.zeros((hallmark.shape[0], len(data_csv.tf)), dtype=int)

def does_interact(tf, geneset):
    if tf in geneset:
        return True
    else:
        for gene in geneset:
            if len(ppi[(ppi.A == tf) & (ppi.B == gene)]) > 0 or len(ppi[(ppi.A == gene) & (ppi.B == tf)]) > 0:
                return True
            
    return False


pbar = tqdm(total=len(hallmark.values))
for idx, (pathway, genes) in enumerate(hallmark.values):
    for idy, tf in enumerate(data_csv.tf):
        pbar.set_description(f'{pathway[9:]} | ({idy}/{len(data_csv.tf)}) - {tf}')
        if does_interact(tf, genes.split('/')):
            hallmark_mask[idx, idy] = 1
    
    pbar.update(1)
pbar.close()

np.save('hallmark_mask.npy', hallmark_mask)