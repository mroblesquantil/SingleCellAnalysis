#library("devtools")
#devtools::install_github("YosefLab/SymSim")

library(SymSim)

# Simular una población
ngenes <- 500
ncells_total <- 250
min_popsize <- 30
#true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total, ngenes=ngenes, evf_type="one.population", Sigma=0.4, randseed=0)

# Este tipo de dato tiene 
# - conteos: true_counts_res$counts (matriz genes x celulas)
# - gene_effects: 
# - cell_meta: Particularmente tiene la población
# - kinetic_params:

#tsne_true_counts <- PlotTsne(meta=true_counts_res[[3]], data=log2(true_counts_res[[1]]+1), evf_type="one.population", n_pc=20, label='pop', saving = F, plotname="one.population")
#tsne_true_counts[[2]]

data(gene_len_pool)
gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
#observed_counts <- True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[3]], protocol="nonUMI", alpha_mean=0.1, alpha_sd=0.05, gene_len=gene_len, depth_mean=1e5, depth_sd=3e3)

######### Five populations

phyla <- Phyla3()


true_counts_res <- SimulateTrueCounts(ncells_total=ncells_total, min_popsize=min_popsize, i_minpop=2, ngenes=ngenes, nevf=10, evf_type="discrete", n_de_evf=9, vary="s", Sigma=0.4, phyla=phyla, randseed=0)
true_counts_res_dis <- true_counts_res
tsne_true_counts <- PlotTsne(meta=true_counts_res[[3]], data=log2(true_counts_res[[1]]+1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="discrete populations (true counts)")
tsne_true_counts[[2]]

#observed_counts <- True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[3]], protocol="nonUMI", alpha_mean=0.1, alpha_sd=0.05, gene_len=gene_len, depth_mean=1e5, depth_sd=3e3)
#tsne_nonUMI_counts <- PlotTsne(meta=observed_counts[[2]], data=log2(observed_counts[[1]]+1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="observed counts nonUMI")
#tsne_nonUMI_counts[[2]]

observed_counts <- True2ObservedCounts(true_counts=true_counts_res[[1]], meta_cell=true_counts_res[[3]], protocol="UMI", alpha_mean=0.05, alpha_sd=0.02, gene_len=gene_len, depth_mean=5e4, depth_sd=3e3)
tsne_UMI_counts <- PlotTsne(meta=observed_counts[[2]], data=log2(observed_counts[[1]]+1), evf_type="discrete", n_pc=20, label='pop', saving = F, plotname="observed counts UMI")
tsne_UMI_counts[[2]]

########## Save data
X <- data.frame(observed_counts$counts)
y <- observed_counts$cell_meta["pop"]

write.matrix(X,file="conteos_symsim_500_250_3.csv")
write.matrix(y,file="clusters_symsim_500_250_3.csv")
