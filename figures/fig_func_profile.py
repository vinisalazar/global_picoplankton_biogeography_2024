#!/usr/bin/env python

import os
import sys
from collections import OrderedDict


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

smd = pd.read_csv("/local/path/to/data/metadata_1454_cluster_labels.csv", index_col=0, dtype={"sourmash_final": "object"})
gmd = pd.read_csv("/local/path/to/genome_metadata.tsv", sep="\t", index_col=0)
data_dir = "/local/path/to/data/R/final"
keepX = "15_15_15_15"

rank = "KEGG_Pathway"
counts = dict()
loadings = {rank: {"max": dict()}}#, "min": dict()}}
plotIndiv_c1_2 = dict()
plotIndiv_c3_4 = dict()

counts[rank] = pd.read_csv(f"/local/path/to/data/counts/{rank}_trimmed_mean_formatted_clean_normalised.csv", index_col=0)
plotIndiv_c1_2[rank] = pd.read_csv(f"{data_dir}/splsda_all_{rank}_ncompc1_2_keepX{keepX}_indiv.csv", index_col=0)
plotIndiv_c3_4[rank] = pd.read_csv(f"{data_dir}/splsda_all_{rank}_ncompc3_4_keepX{keepX}_indiv.csv", index_col=0)
plotIndiv_c1_2[rank]["group"] = plotIndiv_c1_2[rank]["group"].str.replace("SSTC", "OTEM").str.replace("TCON", "TGYR").str.replace("UPWL", "CTEM").str.replace("MEDI", "MTEM").str.replace("NADR", "STEM")
plotIndiv_c3_4[rank]["group"] = plotIndiv_c3_4[rank]["group"].str.replace("SSTC", "OTEM").str.replace("TCON", "TGYR").str.replace("UPWL", "CTEM").str.replace("MEDI", "MTEM").str.replace("NADR", "STEM")

for k, v in counts.items():
    counts[k] = v.join(smd["sourmash_k_10_1487_25m"].astype(int), how="inner")

for contrib in ("max",):
    for comp in (1, 2, 3, 4):
        df = pd.read_csv(f"{data_dir}/splsda_all_KEGG_Pathway_comp{comp}_keepX{keepX}_loadings_median_contrib_{contrib}.csv", index_col=0)
        df["abs_importance"] = df["importance"].apply(abs)
        df = df.sort_values("abs_importance").reset_index()
        df.columns = [i.replace("SSTC", "SANT").replace("TCON", "TGYR").replace("UPWL", "CTEM") for i in df.columns]
        df["GroupContrib"] = df["GroupContrib"].str.replace("SSTC", "SANT").str.replace("TCON", "TGYR").str.replace("UPWL", "CTEM")
        loadings[rank][contrib][f"comp{comp}"] = df

palette = palettes["k_10"]
palette = {v["label"]: v["color"] for k, v in palette.items()}

colors = plotIndiv_c1_2[rank].groupby("group")["col"].unique().apply(lambda l: l[0]).to_dict()
markers = {v["label"]: v["marker"] for k, v in palettes["k_10"].items()}
order = [palettes["k_10"][i]["label"] for i in [14, 16, 10, 2, 3, 7, 5, 11, 9, 0]]


def main():
    # Create GridSpec and subplots
    nrows, ncols = 4, 2
    fig = plt.figure(figsize=(15, 20))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.2, top=1.0, bottom=0.0, wspace=0.1, height_ratios=[1, 0.5, 1, 0.5], width_ratios=[1, 0.5])
    plotIndiv_c1_2_ax = fig.add_subplot(gs[0, 0])
    plotIndiv_c3_4_ax = fig.add_subplot(gs[2, 0])

    # PlotIndiv data
    sns.scatterplot(data=plotIndiv_c1_2[rank],
                    x="x",
                    y="y",
                    s=50,
                    hue="group",
                    palette=colors,
                    hue_order = order,
                    style="group",
                    markers=markers,
                    ax=plotIndiv_c1_2_ax,
                    alpha=0.7,
                    edgecolors="k",
                    linewidth=0.5,
                    
                )

    sns.scatterplot(data=plotIndiv_c3_4[rank],
                    x="x",
                    y="y",
                    s=50,
                    hue="group",
                    palette=colors,
                    hue_order=order,
                    style="group",
                    markers=markers,
                    ax=plotIndiv_c3_4_ax,
                    legend=False,
                    alpha=0.7,
                    edgecolors="k",
                    linewidth=0.5,
                )


    # PlotIndiv customisation
    plotIndiv_c1_2_ax.set_title("Components 1 and 2", fontsize=16)
    plotIndiv_c1_2_ax.set_xlabel("X-Variate 1: 22\% expl. var")
    plotIndiv_c1_2_ax.set_ylabel("X-Variate 2: 16\% expl. var", rotation=0, labelpad=50)
    plotIndiv_c1_2_ax.yaxis.set_label_coords(-.15, .4)
    plotIndiv_c3_4_ax.set_title("Components 3 and 4", fontsize=16)
    plotIndiv_c3_4_ax.set_xlabel("X-Variate 1: 19\% expl. var")
    plotIndiv_c3_4_ax.set_ylabel("X-Variate 2: 5\% expl. var", rotation=0, labelpad=50)
    plotIndiv_c3_4_ax.yaxis.set_label_coords(-.15, .5)

    handles, labels = plotIndiv_c1_2_ax.get_legend_handles_labels()
    for h in handles:
        h.set_markeredgewidth(1)
        h.set_markeredgecolor("black")
    legend_handles = [(palettes["k_10"][i]["marker"], palettes["k_10"][i]["label"], palettes["k_10"][i]["color"]) for i in [14, 16, 10, 2, 3, 7, 5, 11, 9, 0]]
    legend_handles = [mpl.lines.Line2D([], [], color=c, linestyle="None", marker=m, markeredgecolor="black", markeredgewidth=1, label=p) for (m, p, c) in legend_handles]
    legend = plotIndiv_c1_2_ax.legend(handles=legend_handles, title="Province", bbox_to_anchor=(-0.1, 1), prop={"size": "x-large"}, markerscale=2)#fontsize="x-large")
    legend.get_title().set_fontsize(18)

    # PlotLoadings data
    for comp in (1, 2, 3, 4):
        col = 0 if comp in (1, 3) else 1
        row = comp if comp in (1, 3) else (comp - 2)
        ax = fig.add_subplot(gs[row, col])
        if comp % 2:
            sns.barplot(loadings[rank]["max"][f"comp{comp}"], x="importance", y="index", hue="GroupContrib", palette=colors, legend=False, ax=ax, linewidth=0.5, edgecolor="k")
            ax.set_xlim(-.6, .6)
        else:
            sns.barplot(loadings[rank]["max"][f"comp{comp}"], y="importance", x="index", hue="GroupContrib", palette=colors, legend=False, ax=ax, linewidth=0.5, edgecolor="k")
            ax.tick_params(axis='x', rotation=270)
            ax.set_ylim(-.6, .6)
        variate_offset = 0 if comp <= 2 else 2
        ax.set_title(f"X-variate {comp - variate_offset} loading importance. Comp. {comp}", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize='large')

    outfile = f"/local/path/to/figures/img/final_vectors/fig4_func_profile_keepX{keepX}.svg"
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    print(f"Done. Wrote to '{outfile}'.")

if __name__ == "__main__":
    main()
