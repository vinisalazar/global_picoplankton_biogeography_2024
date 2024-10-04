# Load required libraries
source("/local/path/to/scripts/utils.R")
print_with_timestamp("Starting script 'preprocess_data.R'. Loading required libraries.")
suppressWarnings(suppressMessages(library(mixOmics)))
suppressWarnings(suppressMessages(library(dplyr)))
suppressWarnings(suppressMessages(library(zCompositions)))

## Loading and formatting count data ---------------------------------------
print_with_timestamp("Loading genomes data.")
genomes = read.csv("/local/path/to/data/counts/genomes_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
genus = read.csv("/local/path/to/data/counts/genus_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
family = read.csv("/local/path/to/data/counts/family_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
class = read.csv("/local/path/to/data/counts/class_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
order = read.csv("/local/path/to/data/counts/order_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
phylum = read.csv("/local/path/to/data/counts/phylum_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
print_with_timestamp("Loading BRITE data.")
BRITE = read.csv("/local/path/to/data/counts/BRITE_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
print_with_timestamp("Loading KEGG_ko data.")
KEGG_ko = read.csv("/local/path/to/data/counts/KEGG_ko_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
print_with_timestamp("Loading KEGG_Pathway data.")
KEGG_Pathway = read.csv("/local/path/to/data/counts/KEGG_Pathway_trimmed_mean_formatted_filtered_clean.csv", row.names=1, check.names=FALSE)
print_with_timestamp("Loading KEGG_rclass data.")
KEGG_rclass = read.csv("/local/path/to/data/counts/KEGG_rclass_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
print_with_timestamp("Loading KEGG_Reaction data.")
KEGG_Reaction = read.csv("/local/path/to/data/counts/KEGG_Reaction_trimmed_mean_formatted_clean.csv", row.names=1, check.names=FALSE)
print_with_timestamp("Loading COG category data.")
COG_category = read.csv("/local/path/to/data/counts/COG_category_trimmed_mean_formatted_filtered_clean.csv", row.names=1, check.names=FALSE)
print_with_timestamp("Loading env data.")
env = read.csv("/data/gpfs/projects/punim1989/biogo-hub/provinces_final/data/R/env_data_clean.csv", row.names=1, check.names=FALSE)

## Preprocessing and normalization -----------------------------------------
print_with_timestamp("Rounding to integers.")
genomes = round(genomes, 0)
genus = round(genus, 0)
family = round(family, 0)
class = round(class, 0)
order = round(order, 0)
phylum = round(phylum, 0)
BRITE = round(BRITE, 0)
KEGG_ko = round(KEGG_ko, 0)
KEGG_Pathway = round(KEGG_Pathway, 0)
KEGG_rclass = round(KEGG_rclass, 0)
KEGG_Reaction = round(KEGG_Reaction, 0)
COG_category = round(COG_category, 0)
# Not done for env

print_with_timestamp("Removing features that appear in less than two samples.")
has_min_positive_values <- function(column) {
  sum(column > 0) >= 2
}
genomes <- genomes[, apply(genomes, 2, has_min_positive_values)]
genus <- genus[, apply(genus, 2, has_min_positive_values)]
family <- family[, apply(family, 2, has_min_positive_values)]
class <- class[, apply(class, 2, has_min_positive_values)]
order <- order[, apply(order, 2, has_min_positive_values)]
phylum <- phylum[, apply(phylum, 2, has_min_positive_values)]
BRITE <- BRITE[, apply(BRITE, 2, has_min_positive_values)]
KEGG_ko <- KEGG_ko[, apply(KEGG_ko, 2, has_min_positive_values)]
KEGG_Pathway <- KEGG_Pathway[, apply(KEGG_Pathway, 2, has_min_positive_values)]
KEGG_rclass <- KEGG_rclass[, apply(KEGG_rclass, 2, has_min_positive_values)]
KEGG_Reaction <- KEGG_Reaction[, apply(KEGG_Reaction, 2, has_min_positive_values)]
COG_category <- COG_category[, apply(COG_category, 2, has_min_positive_values)]
env <- env[, apply(env[,env_cols], 2, has_min_positive_values)]

print_with_timestamp("Performing multiplicative replacement of zeros for taxonomy profiles.")
genomes_ = cmultRepl(genomes, output="p-counts", z.warning=0.999)
genus_ = cmultRepl(genus, output="p-counts", z.warning=0.999)
family_ = cmultRepl(family, output="p-counts", z.warning=0.999)
class_ = cmultRepl(class, output="p-counts", z.warning=0.999)
order_ = cmultRepl(order, output="p-counts", z.warning=0.999)
phylum_ = cmultRepl(phylum, output="p-counts", z.warning=0.999)
print_with_timestamp("Performing multiplicative replacement of zeros for functional profiles.")
print_with_timestamp("For BRITE data.")
BRITE_ = cmultRepl(BRITE, output="p-counts")
print_with_timestamp("For KEGG_ko data.")
KEGG_ko_ = cmultRepl(KEGG_ko, output="p-counts")
print_with_timestamp("For KEGG_Pathway data.")
KEGG_Pathway_ = cmultRepl(KEGG_Pathway, output="p-counts")
print_with_timestamp("For KEGG_rclass data.")
KEGG_rclass_ = cmultRepl(KEGG_rclass, output="p-counts")
print_with_timestamp("For KEGG_Reaction data.")
KEGG_Reaction_ = cmultRepl(KEGG_Reaction, output="p-counts")
# Not done for COG category (no zeroes)

print_with_timestamp("Performing CLR transform for taxonomy profiles.")
genomes_ = logratio.transfo(genomes_, logratio = "CLR")
genus_ = logratio.transfo(genus_, logratio = "CLR")
family_ = logratio.transfo(family_, logratio = "CLR")
class_ = logratio.transfo(class_, logratio = "CLR")
order_ = logratio.transfo(order_, logratio = "CLR")
phylum_ = logratio.transfo(phylum_, logratio = "CLR")
print_with_timestamp("Performing CLR transform for functional profiles.")
BRITE_ = logratio.transfo(BRITE_, logratio = "CLR")
KEGG_ko_ = logratio.transfo(KEGG_ko_, logratio = "CLR")
KEGG_Pathway_ = logratio.transfo(KEGG_Pathway_, logratio = "CLR")
KEGG_rclass_ = logratio.transfo(KEGG_rclass_, logratio = "CLR")
KEGG_Reaction_ = logratio.transfo(KEGG_Reaction_, logratio = "CLR")
COG_category_ = logratio.transfo(COG_category, logratio = "CLR")
env = scale(env)
genomes_ %>% write.csv("/local/path/to/data/counts/genomes_trimmed_mean_formatted_clean_normalised.csv")
genus_ %>% write.csv("/local/path/to/data/counts/genus_trimmed_mean_formatted_clean_normalised.csv")
family_ %>% write.csv("/local/path/to/data/counts/family_trimmed_mean_formatted_clean_normalised.csv")
class_ %>% write.csv("/local/path/to/data/counts/class_trimmed_mean_formatted_clean_normalised.csv")
order_ %>% write.csv("/local/path/to/data/counts/order_trimmed_mean_formatted_clean_normalised.csv")
phylum_ %>% write.csv("/local/path/to/data/counts/phylum_trimmed_mean_formatted_clean_normalised.csv")
BRITE_ %>% write.csv("/local/path/to/data/counts/BRITE_trimmed_mean_formatted_clean_normalised.csv")
KEGG_ko_ %>% write.csv("/local/path/to/data/counts/KEGG_ko_trimmed_mean_formatted_clean_normalised.csv")
KEGG_Pathway_ %>% write.csv("/local/path/to/data/counts/KEGG_Pathway_trimmed_mean_formatted_clean_normalised.csv")
KEGG_rclass_ %>% write.csv("/local/path/to/data/counts/KEGG_rclass_trimmed_mean_formatted_clean_normalised.csv")
KEGG_Reaction_ %>% write.csv("/local/path/to/data/counts/KEGG_Reaction_trimmed_mean_formatted_clean_normalised.csv")
COG_category_ %>% write.csv("/local/path/to/data/counts/COG_category_trimmed_mean_formatted_clean_normalised.csv")

# genomes_ <- read.csv("/local/path/to/data/counts/genomes_trimmed_mean_formatted_clean_normalised.csv", row.names=1)
# BRITE_ <- read.csv("/local/path/to/data/counts/BRITE_trimmed_mean_formatted_clean_normalised.csv", row.names=1)
# KEGG_ko_ <- read.csv("/local/path/to/data/counts/KEGG_ko_trimmed_mean_formatted_clean_normalised.csv", row.names=1)
# KEGG_Pathway_ <- read.csv("/local/path/to/data/counts/KEGG_Pathway_trimmed_mean_formatted_clean_normalised.csv", row.names=1)
# KEGG_rclass_ <- read.csv("/local/path/to/data/counts/KEGG_rclass_trimmed_mean_formatted_clean_normalised.csv", row.names=1)
# KEGG_Reaction_ <- read.csv("/local/path/to/data/counts/KEGG_Reaction_trimmed_mean_formatted_clean_normalised.csv", row.names=1)
# COG_category_ <- read.csv("/local/path/to/data/counts/COG_category_trimmed_mean_formatted_clean_normalised.csv", row.names=1)

## Loading and formatting metadata -----------------------------------------
print_with_timestamp("Loading metadata.")
md_all = read.csv("/local/path/to/data/metadata_1454_cluster_labels.csv", row.names=1)

# print_with_timestamp("Filtering by metadata.")
md = md_all %>%  subset(depth <= 200)

## Conciliate row names ----------------------------------------------------
# tax_counts = genomes[md %>% row.names,]
print_with_timestamp("Exporting data.")
env = env[md %>% row.names, ]
genomes = genomes_[md %>% row.names, ]
genus = genus_[md %>% row.names, ]
family = family_[md %>% row.names, ]
klass = class_[md %>% row.names, ]
order = order_[md %>% row.names, ]
phylum = phylum_[md %>% row.names, ]
BRITE = BRITE_[md %>% row.names, ]
KEGG_ko = KEGG_ko_[md %>% row.names, ]
KEGG_Pathway = KEGG_Pathway_[md %>% row.names, ]
KEGG_rclass = KEGG_rclass_[md %>% row.names, ]
KEGG_Reaction = KEGG_Reaction_[md %>% row.names, ]
COG_category = COG_category_[md %>% row.names, ]
env %>% write.csv("/local/path/to/data/R/env_data_clean_normalised.csv")


# Add func here later
save(
  genomes,
  genus,
  family,
  klass,
  order,
  phylum,
  env,
  BRITE,
  KEGG_ko,
  KEGG_Pathway,
  KEGG_rclass,
  KEGG_Reaction,
  COG_category,
  file="/local/path/to/data/R/normalised_counts.RData")

print_with_timestamp("All done!")
