library(dplyr)
library(stringr)
library(mixOmics)
library(ggplot2)
source("../hub_provinces/scripts/utils.R")

# Loading data ------------------------------------------------------------
print("Loading data")
smd <- read.csv("./provinces_final/data/metadata_1454_cluster_labels.csv", row.names=1)
gmd <- read.csv("./provinces_final/data/genome_metadata.tsv", sep="\t", row.names=1)
genomes <- read.csv("./provinces_final/data/counts/genomes_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
genus <- read.csv("./provinces_final/data/counts/genus_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
family <- read.csv("./provinces_final/data/counts/family_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
klass <- read.csv("./provinces_final/data/counts/class_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
order <- read.csv("./provinces_final/data/counts/order_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
phylum <- read.csv("./provinces_final/data/counts/phylum_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
KEGG_ko <- read.csv("./provinces_final/data/counts/KEGG_ko_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
KEGG_Pathway <- read.csv("./provinces_final/data/counts/KEGG_Pathway_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
KEGG_Reaction <- read.csv("./provinces_final/data/counts/KEGG_Reaction_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
KEGG_rclass <- read.csv("./provinces_final/data/counts/KEGG_rclass_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
COG_category <- read.csv("./provinces_final/data/counts/COG_category_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=F)
phylum <- subset(phylum, select=-c(p__unclassified))
env <- smd[,env_cols]

# Parameters --------------------------------------------------------------
default_keepX <- 5
ncomp <- 2
data_dir <- "./provinces_final/data/R/drafts"
plot_dir <- "./provinces_final/figures/img/drafts/R"
color.blocks <- c("#A56768", "#76ADAC", "#5F5FA0", "#7F7F7F")

# sPLS-DA (polar) -------------------------------------------------------------
province_metadata <- list(
  `0` = list(description = "Antarctic Polar", label = "APLR", category = "POLR", counts = 30, color = '#c7c7c7'),
  `14` = list(description = "Arctic Polar", label = "BPLR", category = "POLR", counts = 42, color = '#7f7f7f')
)
smd_polar <- smd %>% subset(sourmash_k_10_1487_25m %in% c(0, 14))
smd_polar$cluster_label <- smd_polar$sourmash_k_10_1487_25m
smd_polar$label_column <- sapply(smd_polar$cluster_label, function(x) province_metadata[[as.character(x)]]$label)
smd_polar$colors <- sapply(smd_polar$cluster_label, function(x) province_metadata[[as.character(x)]]$color)
colors <- c("#c7c7c7","#7f7f7f")
pch_styles <- c(15, 17)
Y <- smd_polar$label_column
province_category <- "polar"
print(sprintf("Running sPLS-DA models for %s provinces.", province_category))

X <- list(Taxonomy=genomes[smd_polar %>% row.names,],
          Function_Pathway=KEGG_Pathway[smd_polar %>% row.names,],
          Function_ko=KEGG_ko[smd_polar %>% row.names,],
          Environment=env[smd_polar %>% row.names,])
table <- "genomes_Pathway_KO_Environment"
model <- block.splsda(X, Y, ncomp=ncomp,
                      keepX=list(Taxonomy=rep(default_keepX, ncomp),
                                 Function_Pathway=rep(default_keepX, ncomp),
                                 Function_ko=rep(default_keepX, ncomp)))
p <- plotIndiv(model, col=colors, ind.names=F, pch = pch_styles)
p$df %>% write.csv(sprintf("./provinces_final/data/R/block_splsda_polar_%s_keepX%s_ncomp_%s_plotIndiv.csv", table, default_keepX, ncomp))
p <- plotVar(model, col=color.blocks, cutoff=0.0)
plotVarDf <- data.frame(p$x, p$y, p$Block, p$names, p$col)
plotVarDf %>% write.csv(sprintf("./provinces_final/data/R/block_splsda_polar_%s_keepX%s_ncomp_%s_plotVar.csv", table, default_keepX, ncomp))
p <-cimDiablo(model,
          margins=c(10,25),
          color.Y=colors,
          color.blocks=color.blocks,
          clust.method=c("average", "average"), transpose=T)
p$mat %>% cbind(p$row.sideColors) %>% write.csv(sprintf("./provinces_final/data/R/block_splsda_polar_%s_keepX%s_ncomp_%s_cim.csv", table, default_keepX, ncomp))
network(model)


# biplot
model <- spca(X$Taxonomy, ncomp=2, keepX=c(50, 50))

# rCCA
# model <- rcc(X$Taxonomy, X$Environment)



# sPLS-DA (temperate) -------------------------------------------------------------
province_metadata <- list(
  `10` = list(description = "Upwelling Areas", label = "UPWL", category = "TEMP", counts = 139, color = '#2ca02c'),
  `2` = list(description = "S. Subtropical Convergence", label = "SSTC", category = "TEMP", counts = 43, color = '#d62728'),
  `3` = list(description = "North Atlantic Drift/Agulhas", label = "NADR", category = "TEMP", counts = 34, color = '#FF9896'),
  `7` = list(description = "Mediterranean", label = "MEDI", category = "TEMP", counts = 82, color = '#DBDC8D')
)
smd_temp <- smd %>% subset(sourmash_k_10_1487_25m %in% c(2, 3, 7, 10))
smd_temp$cluster_label <- smd_temp$sourmash_k_10_1487_25m
smd_temp$label_column <- sapply(smd_temp$cluster_label, function(x) province_metadata[[as.character(x)]]$label)
smd_temp$colors <- sapply(smd_temp$cluster_label, function(x) province_metadata[[as.character(x)]]$color)
colors <- c("#DBDC8D", "#FF9896","#d62728", "#2ca02c")
pch_styles <- c(15, 16, 17, 18)
Y <- smd_temp$label_column
province_category <- "temperate"
print(sprintf("Running sPLS-DA models for %s provinces.", province_category))

X <- list(Taxonomy=genomes[smd_temp %>% row.names,],
          Function_Pathway=KEGG_Pathway[smd_temp %>% row.names,],
          Function_ko=KEGG_ko[smd_temp %>% row.names,],
          Environment=env[smd_temp %>% row.names,])
table <- "genomes_Pathway_KO_Environment"
model <- block.splsda(X, Y, ncomp=ncomp,
                      keepX=list(Taxonomy=rep(default_keepX, ncomp),
                                 Function_Pathway=rep(default_keepX, ncomp),
                                 Function_ko=rep(default_keepX, ncomp)),
                      near.zero.var=TRUE)
p <- plotIndiv(model, col=colors, ind.names=F, pch = pch_styles)
p$df %>% write.csv(sprintf("./provinces_final/data/R/block_splsda_temp_%s_keepX%s_ncomp_%s_plotIndiv.csv", table, default_keepX, ncomp))
p <- plotVar(model, col=color.blocks, cutoff=0.0)
plotVarDf <- data.frame(p$x, p$y, p$Block, p$names, p$col)
plotVarDf %>% write.csv(sprintf("./provinces_final/data/R/block_splsda_temp_%s_keepX%s_ncomp_%s_plotVar.csv", table, default_keepX, ncomp))
p <-cimDiablo(model,
              margins=c(10,25),
              color.Y=colors,
              color.blocks=color.blocks,
              clust.method=c("average", "average"))
p$mat %>% cbind(p$row.sideColors) %>% write.csv(sprintf("./provinces_final/data/R/block_splsda_temp_%s_keepX%s_ncomp_%s_cim.csv", table, default_keepX, ncomp))
network(model)

# sPLS-DA (tropical) -------------------------------------------------------------
province_metadata <- list(
  `11` = list(description = "Pacific Equatorial Divergence/Countercurrent", label = "PEQD", category = "TROP", counts = 54, color = '#9467bd'),
  `5` = list(description = "Subtropical Gyres", label = "TGYR", category = "TROP", counts = 161, color = '#aec7e8'),
  `9` = list(description = "Broad Tropical", label = "TROP", category = "TROP", counts = 818, color = '#1f77b4')
)
smd_trop <- smd %>% subset(sourmash_k_10_1487_25m %in% c(11, 5, 9))
smd_trop$cluster_label <- smd_trop$sourmash_k_10_1487_25m
smd_trop$label_column <- sapply(smd_trop$cluster_label, function(x) province_metadata[[as.character(x)]]$label)
smd_trop$colors <- sapply(smd_trop$cluster_label, function(x) province_metadata[[as.character(x)]]$color)
colors <- c("#9467bd","#aec7e8", "#1f77b4")
pch_styles <- c(5, 8, 9)
Y <- smd_trop$label_column
province_category <- "tropical"
print(sprintf("Running sPLS-DA models for %s provinces.", province_category))

X <- list(Taxonomy=genus[smd_trop %>% row.names,],
          Function_Pathway=KEGG_Pathway[smd_trop %>% row.names,],
          Function_ko=KEGG_ko[smd_trop %>% row.names,],
          Environment=env[smd_trop %>% row.names,])
table <- "genus_Pathway_KO_Environment"
model <- block.splsda(X, Y, ncomp=ncomp,
                      keepX=list(Taxonomy=rep(default_keepX, ncomp),
                                 Function_Pathway=rep(default_keepX, ncomp),
                                 Function_ko=rep(default_keepX, ncomp)),
                      near.zero.var=TRUE)
p <- plotIndiv(model, col=colors, ind.names=F, pch = pch_styles)
p$df %>% write.csv(sprintf("./provinces_final/data/R/block_splsda_trop_%s_keepX%s_ncomp_%s_plotIndiv.csv", table, default_keepX, ncomp))
p <- plotVar(model, col=color.blocks, cutoff=0.2)
plotVarDf <- data.frame(p$x, p$y, p$Block, p$names, p$col)
plotVarDf %>% write.csv(sprintf("./provinces_final/data/R/block_splsda_trop_%s_keepX%s_ncomp_%s_plotVar.csv", table, default_keepX, ncomp))
p <-cimDiablo(model,
              margins=c(10,25),
              color.Y=colors,
              color.blocks=color.blocks,
              clust.method=c("average", "average"))
p$mat %>% cbind(p$row.sideColors) %>% write.csv(sprintf("./provinces_final/data/R/block_splsda_trop_%s_keepX%s_ncomp_%s_cim.csv", table, default_keepX, ncomp))
network(model)

p$mat
