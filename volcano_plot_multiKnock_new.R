# import HNSC HPV pvalue data generated from python code A_Check_HNSC_HPV_pvalue.py
library(basicPlotteR)
library(tidyverse)

dir = dirname(rstudioapi::getSourceEditorContext()$path)
setwd(dir)
dev.off()
rm(list=ls())


volcano_plot <- function(t_test, N, title = "" )
{
  topN <- t_test  %>% 
    dplyr::filter(pvalue < 0.05) %>%
    dplyr::select(log10pval)  %>%
    dplyr:: arrange( desc(log10pval)) %>%
    head(N)
  
  topN$log10pval <- formatC(topN$log10pval,format = "e", digits = 2 )
  
  sigTFs <- rownames(topN)
  
  minx <- min(t_test[,'diff']) - 0.01
  maxx <- max(t_test[,'diff']) + 0.01
  
  with(t_test, plot(diff,log10pval, pch=20, xlim=c(minx,maxx), ylab="-log10 pvalue",
                    xlab="mean difference", col="gray",
                    main=paste(tp, title)))  
  
  mysubset <- t_test[sigTFs,]

  with(mysubset, points(diff, log10pval, pch=20, col="red"))
  
  with(mysubset, addTextLabels(xCoords=diff,yCoords=log10pval,labels=sigTFs, cex.label=0.7, col.label = "black"))
 
  return(topN)
}

typeSet = c('COAD',	 'BRCA',	 'CESC',	 'PRAD',	 'LIHC',	 'LUAD',	 'UCEC',	 'BLCA',	 'KIRC',	 'GBM',	 'PCPG',	 'HNSC',	 'THCA',	 'ESCA',	 'STAD',	 'KIRP',	 'LUSC')

# sub = "tanh_NonNegWt_Nrun20_SM"
# sub = "tanh_NonNegWt_Nrun20"
sub = "tanh_NonNegWt_Nrun10"
# sub = "tanh_NonNegWt_Nrun10_SM"
# sub = "leakyRelu_NonNegWt_Nrun10"
# 

pure = TRUE

# knSGA = "TP53"
# knSGA = "KEAP1"
knSGA = "PIK3CA"

# knSGA = "SM_PIK3CA"
# knSGA = "SM_TP53"
# knSGA = "SM_PTEN"
# knSGA = "SM_MAP2K4"
# knSGA = "SM_CDH1"
# knSGA = "SM_GATA3"
# knSGA = "PIK3CA"

# dtype = "TFas"
dtype = "Inters"

tp="BRCA"


if (pure){
  t_test <- read.table(paste0('./outputs/',sub,'/',dtype,'/',tp,'_t_test_',knSGA,'_ave_pure.csv'),sep=",",check.names = FALSE, header = TRUE, row.names = 1, stringsAsFactors = FALSE)  
}else{
  t_test <- read.table(paste0('./outputs/',sub,'/',dtype,'/',tp,'_t_test_',knSGA,'_ave.csv'),sep=",",check.names = FALSE, header = TRUE, row.names = 1, stringsAsFactors = FALSE)  
}



if (dtype == "TFs"){
  topN <- volcano_plot(t_test,20, title=paste0("knock",knSGA,"_ave"))
}else{
  topN <- volcano_plot(t_test, 20, title=paste0("knock",knSGA,"_ave"))
}
 
topN

# # 
# plots.dir.path <- list.files(tempdir(), pattern="rs-graphics", full.names = TRUE);
# plots.png.paths <- list.files(plots.dir.path, pattern=".png", full.names = TRUE)
# file.copy(from=plots.png.paths, to=paste0("./Figures/",knSGA,"/",dtype))
