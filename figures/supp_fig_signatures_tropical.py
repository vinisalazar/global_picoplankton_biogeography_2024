#!/usr/bin/env python
# coding: utf-8


"""
Plot selected variables from sPLS-DA for tropical provinces.
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


wd = "/local/path/to/"
keepX = 5

gmd = pd.read_csv(
    f"{wd}/provinces_final/data/genome_metadata.tsv", sep="\t", index_col=0
)
smd = pd.read_csv(
    f"{wd}/provinces_final/data/metadata_1454_cluster_labels.csv", index_col=0
)
cim = pd.read_csv(
    f"{wd}/provinces_final/data/R/block_splsda_trop_genus_Pathway_KO_Environment_keepX{keepX}_ncomp_2_cim.csv",
    index_col=0,
)
plotIndiv = pd.read_csv(
    f"{wd}/provinces_final/data/R/block_splsda_trop_genus_Pathway_KO_Environment_keepX{keepX}_ncomp_2_plotIndiv.csv",
    index_col=0,
)
plotVar = pd.read_csv(
    f"{wd}/provinces_final/data/R/block_splsda_trop_genus_Pathway_KO_Environment_keepX{keepX}_ncomp_2_plotVar.csv",
    index_col=0,
)
kegg_ko = pd.read_csv(
    f"{wd}/provinces_final/data/counts/KEGG_ko_trimmed_mean_formatted_clean_normalised.csv",
    index_col=0,
)
kegg_Pathway = pd.read_csv(
    f"{wd}/provinces_final/data/counts/KEGG_Pathway_trimmed_mean_formatted_filtered_clean.csv",
    index_col=0,
)

col_colors = []

for col in cim.columns[:-1]:
    if col in gmd.index.to_list() or col in gmd["genus"].unique():
        col_colors.append("#A56768")
    elif col in kegg_ko.columns:
        col_colors.append("#5F5FA0")
    elif col in kegg_Pathway.columns:
        col_colors.append("#76ADAC")
    else:
        col_colors.append("#7F7F7F")


plotIndiv["group"] = plotIndiv["group"].str.replace("TCON", "TGYR")
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


# sns.scatterplot(data=median_scatter, x="x", y="y", style="group", hue="group", palette=median_scatter["col"].unique().tolist(), markers=["v", "^"])


env_cols = "Salinity Nitrate OceanTemperature DissolvedMolecularOxygen Silicate pH Phosphate SeaIceCover Chlorophyll DissolvedIron SeaWaterSpeed".split()


# polar = smd[smd["ProvCategory"] == "Polar"]
# polar = polar.drop(['ARCT_P_1177_SRX8973636', 'ARCT_P_1187_SRX8973635'])


# ## Composite figure


ranks = "domain phylum class order family genus species"


def format_names(
    ix, ranks="domain phylum class order family genus species", taxa_type="genome"
):
    ranks = ranks.split()
    if taxa_type == "genome":
        fmt_name = "; ".join(gmd.loc[ix, ranks].str[3:].tolist())
    else:
        gmd_ = gmd.copy()
        gmd_.index = gmd_[taxa_type]
        gmd_ = gmd_.drop_duplicates(taxa_type)
        fmt_name = "; ".join(gmd_.loc[ix, ranks].str[3:].tolist())
    return fmt_name


def split_on_capitals(s, length=3):
    return " ".join(re.findall(r"[A-Z][a-z]*", s)) if len(s) > length else s


plotIndiv["col"].unique().tolist()


plot_data = cim.T.rename(columns=lambda x: "_".join(x.split("_")[3:]))
plot_data = plot_data.rename(
    index=lambda x: (
        format_names(x, ranks="class order family genus", taxa_type="genus")
        if x in gmd["genus"].to_list()
        else x
    )
)
plot_data = plot_data.rename(
    index=lambda x: split_on_capitals(x) if x in env_cols else x
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
sample_markers = {"TROP": "o", "TGYR": "D", "PEQD": "s"}
var_markers = {
    "Taxonomy": "D",
    "Environment": "X",
    "Function_Pathway": "o",
    "Function_ko": "s",
}
palette = (
    plotIndiv[["group", "col"]].drop_duplicates().set_index("group")["col"].to_dict()
)
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
plotVar_.legend(loc="lower left")

labels_to_add = {
    9: {"x": 0, "y": -0.1, "arrow": True, "ranks": "class order family"},
    11: {
        "x": -0.2,
        "y": -0.15,
        "arrow": True,
    },  # Ether lipid
    15: {"x": 0.15, "y": 0.025, "arrow": True},
    16: {
        "x": -0.5,
        "y": -0.1,
        "arrow": True,
    },
    25: {"x": 0.05, "y": -0.0, "arrow": True, "remove_str": " HypA/HypF_4578"},
    19: {"x": -0.3, "y": -0.1, "arrow": True, "remove_str": " HypB_4579"},
    13: {"x": -0.8, "y": -0.05, "arrow": True, "remove_str": " HypB_4579"},
    17: {"x": -0.2, "y": 0.2, "arrow": True},
    12: {"x": -0.6, "y": -0.2, "arrow": True},
    1: {"x": -0.2, "y": 0.1, "arrow": True, "ranks": "class order family"},
    14: {"x": 0.0, "y": -0.1, "arrow": True},
    18: {"x": -0.3, "y": 0.1, "arrow": True},
    30: {"x": 0.05, "y": 0.05, "arrow": True},
    8: {"x": 0.05, "y": -0.1, "arrow": True, "ranks": "class order family"},
    10: {"x": -0.5, "y": 0.05, "arrow": True, "ranks": "class order family"},
    20: {"x": -0.1, "y": -0.1, "arrow": True, "ranks": "class order family"},
    2: {"x": -0.4, "y": 0.1, "arrow": True, "ranks": "class order"},
    3: {"x": 0.0, "y": 0.05, "arrow": True, "ranks": "class order family"},
    7: {"x": -0.3, "y": -0.1, "arrow": True, "ranks": "class order family"},
    27: {"x": 0.0, "y": -0.085, "arrow": True},
    # Environment
    31: {"x": 0.02, "y": -0.085, "arrow": False},
    32: {"x": -0.1, "y": -0.0, "arrow": False},
    33: {"x": 0.02, "y": -0.085, "arrow": False},
    34: {"x": 0.3, "y": -0.05, "arrow": False},
    35: {"x": 0.02, "y": -0.085, "arrow": False},
    36: {"x": 0.02, "y": -0.085, "arrow": False},
    37: {"x": 0.02, "y": -0.085, "arrow": False},
    38: {"x": 0.02, "y": -0.085, "arrow": False},
    39: {"x": 0.02, "y": -0.085, "arrow": False},
    40: {"x": 0.02, "y": -0.085, "arrow": False},
}

# Adding labels from the "p.names" column
for ix, row in plotVar.iterrows():
    x, y, name, block = row["p.x"], row["p.y"], row["p.names"], row["p.Block"]
    if ix in labels_to_add.keys():
        if block not in ("Taxonomy", "Environment"):
            name = name[0].upper() + name[1:]
        if block == "Environment":
            name = split_on_capitals(name)
        x_offset, y_offset = labels_to_add[ix]["x"], labels_to_add[ix]["y"]
        if block == "Taxonomy":
            name = format_names(name, labels_to_add[ix]["ranks"], taxa_type="genus")
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

_ = g.fig.text(
    0.03, 0.96, "\\textbf{A}", fontsize=20, weight="bold", ha="left", va="center"
)
_ = g.fig.text(
    0.51, 0.96, "\\textbf{B}", fontsize=20, weight="bold", ha="left", va="center"
)
_ = g.fig.text(
    0.03, 0.575, "\\textbf{C}", fontsize=20, weight="bold", ha="left", va="center"
)

plt.savefig(
    "../final_draft_imgs/supp_fig_trop_signatures_genus.pdf", bbox_inches="tight"
)
