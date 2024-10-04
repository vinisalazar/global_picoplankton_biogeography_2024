library(dplyr)
library(stringr)
library(mixOmics)
library(ggplot2)



# Loading data ------------------------------------------------------------
smd <- read.csv("./provinces_final/data/metadata_1454_cluster_labels.csv", row.names=1)
gmd <- read.csv("./provinces_final/data/genome_metadata.tsv", sep="\t", row.names=1)
genomes <- read.csv("./provinces_final/data/counts/genomes_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
genus <- read.csv("./provinces_final/data/counts/genus_trimmed_mean_formatted_clean_normalised.csv", row.names=1, check.names=FALSE)
genus <- genus[, !(names(genus) %in% c("g__unclassified"))]
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

# Parameters --------------------------------------------------------------
default_keepX <- 5
ncomp <- 10
data_dir <- "./provinces_final/data/R"
plot_dir <- "./provinces_final/figures/img/drafts/R"

# sPLS-DA (k=10) -------------------------------------------------------------
province_metadata <- list(
  `0` = list(description = "Antarctic Polar", label = "APLR", category = "POLR", counts = 30, color = '#c7c7c7'),
  `10` = list(description = "Upwelling Areas", label = "UPWL", category = "TEMP", counts = 139, color = '#2ca02c'),
  `11` = list(description = "Pacific Equatorial Divergence/Countercurrent", label = "PEQD", category = "TROP", counts = 54, color = '#9467bd'),
  `14` = list(description = "Arctic Polar", label = "BPLR", category = "POLR", counts = 42, color = '#7f7f7f'),
  `16` = list(description = "Baltic Sea", label = "BALT", category = "BALT", counts = 51, color = '#8c564b'),
  `2` = list(description = "S. Subtropical Convergence", label = "SSTC", category = "TEMP", counts = 43, color = '#d62728'),
  `3` = list(description = "North Atlantic Drift/Agulhas", label = "NADR", category = "TEMP", counts = 34, color = '#FF9896'),
  `5` = list(description = "Subtropical Gyres", label = "TCON", category = "TROP", counts = 161, color = '#aec7e8'),
  `7` = list(description = "Mediterranean", label = "MEDI", category = "TEMP", counts = 82, color = '#DBDC8D'),
  `9` = list(description = "Broad Tropical", label = "TROP", category = "TROP", counts = 818, color = '#1f77b4')
)
smd$cluster_label <- smd$sourmash_k_10_1487_25m
smd$label_column <- sapply(smd$cluster_label, function(x) province_metadata[[as.character(x)]]$label)
smd$colors <- sapply(smd$cluster_label, function(x) province_metadata[[as.character(x)]]$color)
colors <- c("#c7c7c7",  "#8c564b", "#7f7f7f", "#dbdc8d", "#ff9896","#9467bd", "#d62728" ,"#aec7e8", "#1f77b4", "#2ca02c")
pch_styles <- c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
Y <- smd$label_column
province_category <- "all"
data_dir <- "./provinces_final/data/R/drafts"


## genome -------------------------------------------------------------------
X <- genomes[smd %>% row.names,]
table <- "genomes"
ncomp <- 8
default_keepX <- 4
model <- splsda(X, Y, keepX=rep(default_keepX, ncomp), ncomp=ncomp)
p <- cim(model, margins=c(10,10), row.sideColors = smd$colors, transpose=T, clust.method=c("average", "average"))
p$mat %>% write.csv(sprintf("%s/splsda_%s_%s_ncomp%s_keepX%s_cim.csv", data_dir, province_category, table, model$ncomp, model$keepX[1]))


## genus -------------------------------------------------------------------
X <- genus[smd %>% row.names,]
table <- "genus"
ncomp <- 10
default_keepX <- 2
model <- splsda(X, Y, keepX=rep(default_keepX, ncomp), ncomp=ncomp)
p <- cim(model, margins=c(10,10), row.sideColors = smd$colors, transpose=T, clust.method=c("average", "average"))
p$mat %>% write.csv(sprintf("%s/splsda_%s_%s_ncomp%s_keepX%s_cim.csv", data_dir, province_category, table, model$ncomp, model$keepX[1]))

## kegg_pathway / tune -------------------------------------------------------------------
X <- KEGG_Pathway[smd %>% row.names,]
table <- "KEGG_Pathway"
tune_results_all[[table]] <- tune.splsda(X, Y, ncomp=4, test.keepX=c(5,6,7,8,9,10,11,12,13,14,15), auc=T, progressBar=T, cpus=4, nrepeat=5)
model <- splsda(X, Y, keepX=tune_results_all[[table]]$choice.keepX, ncomp=tune_results_all[[table]]$choice.ncomp$ncomp)
p <- cim(model, margins=c(10,15), row.sideColors = smd$colors, transpose=T, clust.method=c("average", "average"), scale=TRUE, center=TRUE)
p <- plotIndiv(model, ind.names=F, comp = c(1,2), col=colors, legend=F, pch=pch_styles, title="sPLS-DA of KEGG Pathway, comps 1 and 2, keepX=(5, 10)")
p$df %>%write.csv(sprintf("%s/splsda_%s_%s_ncomp%s_keepX%s_indiv.csv", data_dir, province_category, table, "c1_2",paste(model$keepX,collapse="_")))
p <- plotIndiv(model, ind.names=F, comp = c(3,4), col=colors, legend=F, pch=pch_styles, title="sPLS-DA of KEGG Pathway, comps 3 and 4, keepX=(20, 20)")
p$df %>%write.csv(sprintf("%s/splsda_%s_%s_ncomp%s_keepX%s_indiv.csv", data_dir, province_category, table, "c3_4",paste(model$keepX,collapse="_")))
for (contrib in c("max", "min")) {
    for (comp in seq(model$ncomp)) {
      p <- plotLoadings(model, comp=comp, method="median", contrib=contrib, title = sprintf("Component %s, Method = 'median', contrib = '%s'", comp, contrib), legend.color = colors, legend=F)
      p$X %>% as.data.frame() %>% filter_all(any_vars(. != 0)) %>% as.matrix() %>% write.csv(sprintf("%s/splsda_%s_%s_comp%s_keepX%s_loadings_median_contrib_%s.csv", data_dir, province_category, table, comp, paste(model$keepX,collapse="_"), contrib))
    }
  }


## kegg_pathway / keepX15, ncomp4 -------------------------------------------------------------------
X <- KEGG_Pathway[smd %>% row.names,]
table <- "KEGG_Pathway"
model <- splsda(X, Y, keepX=rep(15, 4), ncomp=4)
p <- cim(model, margins=c(10,15), row.sideColors = smd$colors, transpose=T, clust.method=c("average", "average"), scale=TRUE, center=TRUE)
p <- plotIndiv(model, ind.names=F, comp = c(1,2), col=colors, legend=F, pch=pch_styles, title="sPLS-DA of KEGG Pathway, comps 1 and 2, keepX=(15, 15)")
p$df %>%write.csv(sprintf("%s/splsda_%s_%s_ncomp%s_keepX%s_indiv.csv", data_dir, province_category, table, "c1_2",paste(model$keepX,collapse="_")))
p <- plotIndiv(model, ind.names=F, comp = c(3,4), col=colors, legend=F, pch=pch_styles, title="sPLS-DA of KEGG Pathway, comps 3 and 4, keepX=(15, 15)")
p$df %>%write.csv(sprintf("%s/splsda_%s_%s_ncomp%s_keepX%s_indiv.csv", data_dir, province_category, table, "c3_4",paste(model$keepX,collapse="_")))
for (contrib in c("max")) {
    for (comp in seq(model$ncomp)) {
      p <- plotLoadings(model, comp=comp, method="median", contrib=contrib, title = sprintf("Component %s, Method = 'median', contrib = '%s'", comp, contrib), legend.color = colors, legend=F)
      p$X %>% as.data.frame() %>% filter_all(any_vars(. != 0)) %>% as.matrix() %>% write.csv(sprintf("%s/splsda_%s_%s_comp%s_keepX%s_loadings_median_contrib_%s.csv", data_dir, province_category, table, comp, paste(model$keepX,collapse="_"), contrib))
    }
  }

