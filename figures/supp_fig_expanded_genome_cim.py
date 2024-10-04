#!/usr/bin/env python

import os
import sys

import pandas as pd
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
 
sys.path.insert(0, "/local/path/to/scripts/")
from plotting_utils import palettes

os.environ["PATH"] = (
    os.environ["PATH"]
    + ":/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/texlive/20230313/bin/x86_64-linux/"
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.usetex"] = True
mpl.rc('text.latex', preamble=r'\usepackage{cmbright}')

smd = pd.read_csv("/local/path/to/data/metadata_1454_cluster_labels.csv", index_col=0, dtype={"sourmash_k_10_1487_25m": "object"})
gmd = pd.read_csv("/local/path/to/genome_metadata.tsv", sep="\t", index_col=0)

tables_normalised = {
    "genomes": pd.read_csv("/local/path/to/data/counts/genomes_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
    "genus": pd.read_csv("/local/path/to/data/counts/genus_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
    "family": pd.read_csv("/local/path/to/data/counts/family_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
    "class": pd.read_csv("/local/path/to/data/counts/class_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
    "order": pd.read_csv("/local/path/to/data/counts/order_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
    "phylum": pd.read_csv("/local/path/to/data/counts/phylum_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
    "BRITE": pd.read_csv("/local/path/to/data/counts/BRITE_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
}
tables_renamed = {
    "KEGG_ko": pd.read_csv("/local/path/to/data/counts/KEGG_ko_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
    "KEGG_Pathway": pd.read_csv("/local/path/to/data/counts/KEGG_Pathway_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
    "KEGG_rclass": pd.read_csv("/local/path/to/data/counts/KEGG_rclass_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
    "KEGG_Reaction": pd.read_csv("/local/path/to/data/counts/KEGG_Reaction_trimmed_mean_formatted_clean_normalised.csv", index_col=0),
}
tables_normalised["genomes"] = tables_normalised["genomes"].join(smd["sourmash_k_10_1487_25m"].astype(int), how="inner")
tables_renamed["KEGG_ko"] = tables_renamed["KEGG_ko"].join(smd["sourmash_k_10_1487_25m"].astype(int), how="inner")

ncomp = 20
keepX = 5
rank = "genomes"
ranks = "domain phylum class order family genus sci_names".split()

df = pd.read_csv(f"/local/path/to/data/R/final/splsda_all_{rank}_ncomp{ncomp}_keepX{keepX}_cim.csv", index_col=0)

new_cols = df.columns.tolist()

for i, ix in enumerate(new_cols):
    if rank == "genomes":
        if ix in gmd.index:
            new_cols[i] = gmd.loc[ix, "phylum class order family sci_names".split()]
            new_cols[i]["index"] = new_cols[i].name
            new_cols[i] = new_cols[i].to_list()
            # new_cols[i][-2] = "\\textit{" + new_cols[i][-2] + "}"
            new_cols[i] = "; ".join([s[3:] if s[1] == "_" else s for s in new_cols[i]])
    else:
        if ix in gmd[rank].to_list():
            new_cols[i] = gmd[gmd[rank] == ix][ranks[:ranks.index(rank) + 1]].iloc[0]
            new_cols[i] = new_cols[i].to_list()
            # new_cols[i][-2] = "\\textit{" + new_cols[i][-2] + "}"
            new_cols[i] = "; ".join([s[3:] if s[1] == "_" else s for s in new_cols[i]])[:-2]

df.columns = new_cols

col_colors = smd.loc[df.index, "sourmash_k_10_1487_25m"].astype(int).map({k: v["color"] for k, v in palettes["k_10"].items()})
col_colors.name = "Province"


def main():
    p = sns.clustermap(df.T,
                    cmap="Spectral_r", figsize=(20, 30),
                    xticklabels=False,
                    col_cluster=True,
                    col_colors=col_colors,
                    dendrogram_ratio=(0.1, 0.1),
                    cbar_pos=(0.75, 0.9, 0.18, 0.05),
                    vmax=5,
                    vmin=-5,
                    cbar_kws={"orientation": "horizontal"})
    # Increase the font size of yticks
    _ = p.ax_heatmap.set_yticklabels(p.ax_heatmap.get_yticklabels(), fontsize=12)
    _ = p.ax_col_colors.set_yticklabels(p.ax_col_colors.get_yticklabels(), fontsize=12)

    # Increase the font size of colorbar ticks
    _ = p.cax.set_xticklabels(p.cax.get_xticklabels(), fontsize=12)
    _ = p.cax.xaxis.set_label_position("top")
    _ = p.cax.set_xlabel("Correlation", fontsize=18, labelpad=10)

    # Increase the font size of col_color ticks
    _ = p.ax_col_colors.set_xticklabels(p.ax_col_colors.get_xticklabels(), fontsize=20)

    plt.savefig(f"/local/path/to/figures/img/final_vectors/supp_fig_taxonomic_profile.pdf", dpi=400, bbox_inches="tight")


if __name__ == "__main__":
    main()
