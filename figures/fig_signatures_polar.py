#!/usr/bin/env python
# coding: utf-8


"""
Plot selected variables from sPLS-DA for polar provinces.
"""


import os
import sys
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text

sys.path.insert(0, "/local/path/to/scripts/")

os.environ["PATH"] = (
    os.environ["PATH"]
    + ":/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/texlive/20230313/bin/x86_64-linux/"
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.usetex"] = True
mpl.rc("text.latex", preamble=r"\usepackage{cmbright}")


machine = "Alfred"
wd = (
    "/local/path/to/"
    if machine == "Alfred"
    else "/data/gpfs/projects/punim1989/biogo-hub/"
)
keepX = 5

gmd = pd.read_csv(
    f"{wd}/provinces_final/data/genome_metadata.tsv", sep="\t", index_col=0
)
smd = pd.read_csv(
    f"{wd}/provinces_final/data/metadata_1454_cluster_labels.csv", index_col=0
)
cim = pd.read_csv(
    f"{wd}/provinces_final/data/R/block_splsda_polar_genomes_Pathway_KO_Environment_keepX{keepX}_ncomp_2_cim.csv",
    index_col=0,
)
plotIndiv = pd.read_csv(
    f"{wd}/provinces_final/data/R/block_splsda_polar_genomes_Pathway_KO_Environment_keepX{keepX}_ncomp_2_plotIndiv.csv",
    index_col=0,
)
plotVar = pd.read_csv(
    f"{wd}/provinces_final/data/R/block_splsda_polar_genomes_Pathway_KO_Environment_keepX{keepX}_ncomp_2_plotVar.csv",
    index_col=0,
)
kegg_ko = pd.read_csv(
    f"{wd}/provinces_final/data/counts/KEGG_ko_trimmed_mean_formatted_clean_normalised.csv",
    index_col=0,
    nrows=2,
)
kegg_Pathway = pd.read_csv(
    f"{wd}/provinces_final/data/counts/KEGG_Pathway_trimmed_mean_formatted_filtered_clean.csv",
    index_col=0,
    nrows=2,
)

col_colors = []

for col in cim.columns[:-1]:
    if col in gmd.index:
        col_colors.append("#A56768")
    elif col in kegg_ko.columns:
        col_colors.append("#5F5FA0")
    elif col in kegg_Pathway.columns:
        col_colors.append("#76ADAC")
    else:
        col_colors.append("#7F7F7F")

plotIndiv = plotIndiv.reset_index()
to_be_renamed_ix = plotIndiv[
    plotIndiv.merge(smd["sample_name"], left_on="index", right_index=True, how="left")[
        "sample_name"
    ].isnull()
].index
plotIndiv.loc[to_be_renamed_ix, "index"] = plotIndiv.loc[to_be_renamed_ix, "index"].str[
    :-1
]
median_scatter = (
    plotIndiv["index x y group Block".split()]
    .groupby("index")
    .median(numeric_only=True)
    .join(
        plotIndiv.drop_duplicates("index")["index group col".split()].set_index("index")
    )
)
env_cols = "Salinity Nitrate OceanTemperature DissolvedMolecularOxygen Silicate pH Phosphate SeaIceCover Chlorophyll".split()
polar = smd[smd["ProvCategory"] == "Polar"]
polar = polar.drop(["ARCT_P_1177_SRX8973636", "ARCT_P_1187_SRX8973635"])
ranks = "domain phylum class order family genus species"


def format_names(ix, ranks="domain phylum class order family genus species"):
    ranks = ranks.split()
    return "; ".join(gmd.loc[ix, ranks].str[3:].tolist())


def split_on_capitals(s, length=3):
    return " ".join(re.findall(r"[A-Z][a-z]*", s)) if len(s) > length else s


plot_data = cim.T.rename(columns=lambda x: "_".join(x.split("_")[3:]))
plot_data = plot_data.rename(
    index=lambda x: (
        x + ": " + format_names(x, ranks="class order family genus")
        if x in gmd.index
        else x
    )
)
plot_data = plot_data.rename(
    index=lambda x: (
        split_on_capitals(x)
        if x in env_cols + ["DissolvedIron", "SeaWaterSpeed"]
        else x
    )
)

g = sns.clustermap(
    plot_data.iloc[:-1, :].astype(float),
    cmap="Spectral_r",
    col_colors=plot_data.iloc[-1].rename("Province"),
    row_colors=pd.Series(col_colors, index=plot_data.iloc[:-1].index, name="Block"),
    figsize=(15, 15),
    # cbar_pos=(0.05, 0.8, 0.05, 0.15),
    cbar_pos=None,
    dendrogram_ratio=(0.1, 0.1),
    method="average",
)
g.gs.update(top=0.6, bottom=0.05)


gs = mpl.gridspec.GridSpec(
    1, 2, top=0.95, bottom=0.65, left=0.05, right=0.95, wspace=0.15
)

axes = g.figure.add_subplot(gs[0]), g.figure.add_subplot(gs[1])
sample_markers = ["v", "^"]
var_markers = {
    "Taxonomy": "D",
    "Environment": "X",
    "Function_Pathway": "o",
    "Function_ko": "s",
}
palette = plotIndiv["col"].unique().tolist()
var_palette = plotVar["p.col"].unique().tolist()

plotIndiv_ = sns.scatterplot(
    data=median_scatter.rename(columns={"group": "Province"}),
    x="x",
    y="y",
    style="Province",
    hue="Province",
    palette=palette,
    markers=sample_markers,
    s=100,
    edgecolor="k",
    ax=axes[0],
)

plotIndiv_.set_xlabel("X-Variate 1 median")
plotIndiv_.set_ylabel("X-Variate 2 median")
plotIndiv_.set_title("Median of ordination scores")

plotVar_ = sns.scatterplot(
    data=plotVar.rename(columns={"p.Block": "Block"}),
    x="p.x",
    y="p.y",
    hue="Block",
    style="Block",
    palette=var_palette,
    markers=var_markers,
    ax=axes[1],
    s=150,
)

plotVar_.set_xlabel("")
plotVar_.set_ylabel("")
ticks = [-1, -0.5, 0, 0.5, 1]
plotVar_.set_xticks(ticks)
plotVar_.set_yticks(ticks)
circle = mpl.patches.Circle(
    (0, 0), 0.5, color="black", fill=False, lw=0.5, ls="--", zorder=0
)
plotVar_.add_artist(circle)
circle = mpl.patches.Circle(
    (0, 0), 1, color="black", fill=False, lw=0.5, ls="--", zorder=0
)
plotVar_.add_artist(circle)


labels_to_add = {
    # Top center
    1: {"x": -0.5, "y": 0.15, "arrow": True, "ranks": "class order family"},
    28: {"x": 0.05, "y": 0.1, "arrow": True, "remove_str": " substrate-binding_4253"},
    20: {"x": -0.2, "y": -0.2, "arrow": True},
    15: {
        "x": -0.5,
        "y": 0.12,
        "arrow": True,
        "remove_str": "and other terpenoid-quinone ",
    },
    16: {"x": -0.2, "y": -0.25, "arrow": True},
    # Center right
    10: {"x": -0.65, "y": 0.25, "arrow": True, "ranks": "class order family"},
    17: {"x": 0.05, "y": -0.1, "arrow": True},
    25: {"x": -0.5, "y": -0.1, "arrow": True},
    # Center left
    7: {"x": -0.0, "y": -0.15, "arrow": True, "ranks": "class order family"},
    11: {"x": -0.5, "y": 0.2, "arrow": True},
    14: {"x": -0.5, "y": 0.2, "arrow": True},
    18: {"x": -0.0, "y": -0.15, "arrow": True},
    19: {"x": -0.5, "y": -0.2, "arrow": True},
    # Environment
    31: {"x": 0.02, "y": -0.085, "arrow": False},
    32: {"x": -0.075, "y": -0.0, "arrow": False},
    33: {"x": 0.02, "y": -0.085, "arrow": False},
    34: {"x": 0.2, "y": -0.1, "arrow": False},
    35: {"x": 0.075, "y": -0.06, "arrow": False},
    36: {"x": 0.02, "y": -0.085, "arrow": False},
    37: {"x": 0.02, "y": -0.085, "arrow": False},
    38: {"x": 0.0, "y": -0.1, "arrow": False},
    39: {"x": -0.06, "y": 0.05, "arrow": False},
    40: {"x": 0.02, "y": -0.07, "arrow": False},
    41: {"x": 0.02, "y": -0.085, "arrow": False},
}

# Adding labels from the "p.names" column
for ix, row in plotVar.iterrows():
    x, y, name, block = row["p.x"], row["p.y"], row["p.names"], row["p.Block"]
    if ix in labels_to_add.keys():
        if block not in ("Taxonomy", "Environment"):
            name = name.capitalize()
        if block == "Environment":
            name = split_on_capitals(name)
        x_offset, y_offset = labels_to_add[ix]["x"], labels_to_add[ix]["y"]
        if block == "Taxonomy":
            name = format_names(name, labels_to_add[ix]["ranks"])
        if "remove_str" in labels_to_add[ix]:
            name = name.replace(labels_to_add[ix]["remove_str"], "")
        if labels_to_add[ix]["arrow"]:
            plotVar_.annotate(
                name,
                xy=(x, y),
                xytext=(x + x_offset, y + y_offset),
                fontsize=10,
                arrowprops=dict(arrowstyle="-", lw=1.5, color="black"),
            )
        else:
            text = plotVar_.text(
                x + x_offset,
                y + y_offset,
                name,
                fontsize=10,
                color="black",
                horizontalalignment="center",
                weight="semibold",
            )

del g.ax_cbar

g.fig.text(
    0.03, 0.96, "\\textbf{A}", fontsize=20, weight="bold", ha="left", va="center"
)
g.fig.text(
    0.51, 0.96, "\\textbf{B}", fontsize=20, weight="bold", ha="left", va="center"
)
g.fig.text(
    0.03, 0.575, "\\textbf{C}", fontsize=20, weight="bold", ha="left", va="center"
)


plt.savefig("../final_draft_imgs/fig5_polar_signatures.svg", bbox_inches="tight")
gmd.loc[plotVar.sort_values(["p.y", "p.x"], ascending=False)["p.names"].iloc[:5]]
sns.clustermap(
    cim.iloc[:, :-1].T,
    cmap="Spectral_r",
    col_colors=cim.iloc[:, -1],
    row_colors=col_colors,
    figsize=(15, 15),
    method="average",
)
