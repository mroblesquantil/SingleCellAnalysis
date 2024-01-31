library(Seurat)
library(SeuratDisk)
library(SeuratData)
library(patchwork)
library(dplyr)
library("rhdf5")
library(tidyverse)

path_folder <- "generados/zi_negative_binomial_dim10000/hg19/"
path_output <- "generados/zi_negative_binomial_dim10000/resultados/clusters.csv"

print_genes_cells <- function(seurat_obj){
  # Obtener el número de características (genes)
  num_genes <- nrow(seurat_obj)
  
  # Obtener el número de células (muestras)
  num_cells <- ncol(seurat_obj)
  
  # Imprimir los resultados
  cat("Número de características (genes):", num_genes, "\n")
  cat("Número de células (muestras):", num_cells, "\n")
}

pbmc.data <-  Read10X(data.dir = path_folder) 
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbmc3k")

print_genes_cells(pbmc)

# Examinamos algunos genes 
#pbmc.data[c("CD3D", "TCL1A", "MS4A1"), 1:30]

# Seurat permite hacer un análisis de las observaciones
# Podemos saber: Número de genes únicos detectados en cada célula
#                Número de células activadas en cada gen
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

#plot1 <- FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "percent.mt")
#plot2 <- FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
#plot1 + plot2

#pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

print_genes_cells(pbmc)

###############################################################################################
############################################################## NORMALIZACIÓN
###############################################################################################

# La normalización por default normaliza por célula, multiplica por un factor de 10000 y aplica logaritmo.
pbmc <- NormalizeData(pbmc)

###############################################################################################
############################################################## HIGHLY VARIABLE FEATURES
###############################################################################################

# Se encuentran los genes altamente variables (que se expresen mucho en algunas células y poco en otras)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(pbmc), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(pbmc)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
#plot1 + plot2

###############################################################################################
############################################################## SCALING
###############################################################################################

# Vuelve la media a 0 y la varianza a 1
all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

###############################################################################################
############################################################## DIMENSIONALITY REDUCTION
###############################################################################################

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
print_genes_cells(pbmc)

print(pbmc[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")
DimPlot(pbmc, reduction = "pca")

DimHeatmap(pbmc, dims = 1, cells = 500, balanced = TRUE)

ElbowPlot(pbmc)

###############################################################################################
############################################################## DIMENSIONALITY REDUCTION
###############################################################################################


pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 2.672283915)
pbmc <- RunUMAP(pbmc, dims = 1:10)

DimPlot(pbmc, reduction = "umap")


cluster_results <- pbmc@meta.data
write.csv(cluster_results, path_output, row.names=FALSE)
