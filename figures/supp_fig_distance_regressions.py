#!/usr/bin/env python
# coding: utf-8

#### ADD LATEX RENDERING
import os

os.environ["PATH"] = (
    os.environ["PATH"]
    + ":/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/texlive/20230313/bin/x86_64-linux/"
)
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.usetex"] = True

#### OTHER IMPORTS
import sys
sys.path.insert(0, "/local/path/to/scripts/")
from utils import print_with_timestamp
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from functools import reduce
from itertools import combinations
from pathlib import Path
from math import ceil
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D


import seaborn as sns
from scipy.stats import spearmanr

# # Run this if necessary
# !python ./hub_provinces/scripts/create_distance_matrices.py

def distribution_plot(df, columns, plot_type=sns.boxenplot, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_data = df[columns.keys()]
    plot_data = (plot_data * 100).melt()
    plot_data["variable_fmt"] = plot_data["variable"].map(columns)
    data_type_color_mappings_ = {
        v: data_type_color_mappings[k] for k, v in columns.items()
    }
    plot_data["color"] = plot_data["variable"].apply(lambda v: colors[data_types[v]])
    plot_type(
        data=plot_data,
        x="variable_fmt",
        y="value",
        ax=ax,
        palette=data_type_color_mappings_,
        **kwargs,
    )
    ax.set_ylabel("Dissimilarity (\%)")
    ax.set_xlabel("Data blocks")


def create_pairplot(
    df,
    save_path=None,
    plot_type=sns.scatterplot,
    nrows=5,
    add_diagonals=False,
    color_axes=True,
    omit=None,
    square_fig=False,
    combinations_=None,
    skip_func_func=True,
    print_combination=True,
    **kwargs,
):
    # Create a figure with three panels arranged horizontally
    if omit:
        if isinstance(omit, str):
            omit = omit.split()
        df = df.drop(columns=omit)
    if not combinations_:
        combinations_ = list(combinations(df.columns, 2))
    if skip_func_func:
        combinations_ = [
            i
            for i in combinations_
            if not (("kegg" in i[0].lower()) and ("kegg" in i[1].lower()))
        ]
    num_plots = len(combinations_)
    ncols = ceil(num_plots / nrows)
    if square_fig:
        nrows = ceil(num_plots**0.5)
        nrows = ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flat  # Flatten the ndarray to iterate over individual subplots

    for i, ax in enumerate(axes):
        if i < num_plots:  # In case there are more subplots than needed
            column1, column2 = combinations_[i]
            if print_combination:
                print_with_timestamp(f"Plotting '{column1}' vs. '{column2}'.")
            if ("sourmash" in (columns := [column1, column2])) and (
                "genomes" not in columns
            ):
                placeholder = column1
                column1 = column2
                column2 = placeholder
            x = df[column1]
            y = df[column2]
            column1_fmt = (
                column1.capitalize()
                .replace("_", " ")
                .replace("Kegg", "KEGG")
                .replace("ko", "KO")
                .replace("Brite", "BRITE")
                .replace("Genomes", "Taxonomy")
                .replace("Env", "Env. parameters")
            )
            column2_fmt = (
                column2.capitalize()
                .replace("_", " ")
                .replace("Kegg", "KEGG")
                .replace("ko", "KO")
                .replace("Brite", "BRITE")
                .replace("Genomes", "Taxonomy")
                .replace("Env", "Env. parameters")
            )
            column1_color = colors[data_types[column1]]
            column2_color = colors[data_types[column2]]
            column1_color = data_type_color_mappings[column1]
            column2_color = data_type_color_mappings[column2]

            # Create a scatter plot
            if plot_type == sns.scatterplot:
                plot_type(
                    x=x,
                    y=y,
                    ax=ax,
                    color="darkgray",
                    alpha=0.25,
                    zorder=2,
                    edgecolor="k",
                    linewidth=1.25,
                    **kwargs,
                )
            else:
                plot_type(x=x, y=y, ax=ax, **kwargs)

            # Calculate the Spearman correlation coefficient and p-value
            rho, p_value = spearmanr(x, y)

            # Add correlation info to the legend
            legend_text = f"$\\rho = {rho:.2f}$"
            ax.legend(
                [legend_text],
                loc="lower right",
                markerscale=0,
                frameon=False,
                fontsize=14,
                handletextpad=-2,
                handlelength=0,
            )

            # Set titles and labels
            ax.set_title(f"{column1_fmt} vs. {column2_fmt}")
            ax.set_xlabel(column1_fmt)
            ax.set_ylabel(column2_fmt)

            if color_axes:
                # Set colors
                ax.spines["bottom"].set_color(column1_color)
                ax.spines["right"].set_color(column1_color)
                ax.tick_params(axis="x", colors=column1_color)
                ax.xaxis.label.set_color(column1_color)
                ax.spines["top"].set_color(column2_color)
                ax.spines["left"].set_color(column2_color)
                ax.tick_params(axis="y", colors=column2_color)
                ax.yaxis.label.set_color(column2_color)

            if add_diagonals:
                # Create patches for diagonals
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()
                lower_diag = Polygon(
                    [[x_min, y_min], [x_max, y_min], [x_max, y_max]],
                    closed=True,
                    facecolor=column1_color,
                    alpha=0.075,
                    zorder=0,
                )
                upper_diag = Polygon(
                    [[x_min, y_min], [x_min, y_max], [x_max, y_max]],
                    closed=True,
                    facecolor=column2_color,
                    alpha=0.075,
                    zorder=0,
                )

                # Create a dotted line across the diagonal
                diagonal_line = Line2D(
                    [x_min, x_max],
                    [y_min, y_max],
                    linestyle="--",
                    color="gray",
                    linewidth=1,
                    zorder=1,
                    alpha=0.7,
                )

                # Add objects
                ax.add_line(diagonal_line)
                ax.add_patch(upper_diag)
                ax.add_patch(lower_diag)
        else:
            # Clear axes that are not used
            plt.delaxes(ax)

    fig.suptitle(
        "Pairwise distance correlations between blocks of data.", fontsize=16, y=1.01
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

df = pd.read_csv(
    "/data/gpfs/projects/punim1989/biogo-hub/provinces_final/data/distances/merged_distances_2132_filtered_zeros.csv",
)
smd = pd.read_csv("/local/path/to/data/metadata_1454_cluster_labels.csv", index_col=0)

breakpoint()
df = df[df["query"].isin(smd.index)]
df = df[df["subject"].isin(smd.index)]
df = df.set_index(["query", "subject"])

df = df[
    "genomes sourmash searoute env KEGG_ko KEGG_Pathway KEGG_rclass KEGG_Reaction BRITE".split()
]


data_types = {
    "genomes": "Taxonomy",
    "KEGG_ko": "Functional",
    "KEGG_Pathway": "Functional",
    "KEGG_rclass": "Functional",
    "KEGG_Reaction": "Functional",
    "BRITE": "Functional",
    "sourmash": "Sequence",
    "searoute": "Environmental",
    "env": "Environmental",
}

colors = {
    "Taxonomy": "darkblue",
    "Functional": "darkcyan",
    "Sequence": "darkred",
    "Environmental": "darkgreen",
}

data_colors = {
    "genomes": "#00108B",
    "KEGG_ko": "#e1e105",
    "KEGG_Pathway": "#afaf04",
    "KEGG_Reaction": "#8B8B03",
    "KEGG_rclass": "#969603",
    "sourmash": "#8B0002",
}
data_colors = {
    "genomes": "#00108B",
    "KEGG_ko": "#8B8B03",
    "KEGG_Pathway": "#8B8B03",
    "KEGG_Reaction": "#8B8B03",
    "KEGG_rclass": "#8B8B03",
    "sourmash": "#8B0002",
}

data_type_color_mappings = {k: colors[v] for k, v in data_types.items()}
omit = "BRITE KEGG_rclass".split()

combinations_ = [
    ("genomes", "env"),
    ("genomes", "searoute"),
    ("genomes", "KEGG_ko"),
    ("genomes", "KEGG_Pathway"),
    ("genomes", "KEGG_Reaction"),
    ("genomes", "sourmash"),
    ("sourmash", "env"),
    ("sourmash", "searoute"),
    ("sourmash", "KEGG_ko"),
    ("sourmash", "KEGG_Pathway"),
    ("sourmash", "KEGG_Reaction"),
    ("searoute", "KEGG_ko"),
    ("searoute", "KEGG_Pathway"),
    ("searoute", "KEGG_Reaction"),
    ("env", "KEGG_ko"),
    ("env", "KEGG_Pathway"),
    ("env", "KEGG_Reaction"),
    ("KEGG_ko", "KEGG_Pathway"),
    ("KEGG_ko", "KEGG_Reaction"),
    ("KEGG_Pathway", "KEGG_Reaction"),
    ("searoute", "env"),
]
def main():
    scatter_kws = {
        "color": "darkgray",
        "alpha": 0.75,
        "zorder": 2,
        "edgecolor": "k",
        "linewidth": 1.25,
    }
    print_with_timestamp("Creating pairplot with binned data.")
    create_pairplot(
        df,
        plot_type=sns.regplot,
        fit_reg=False,
        omit="BRITE KEGG_rclass",
        nrows=3,
        add_diagonals=True,
        scatter_kws=scatter_kws,
        color="k",
        x_bins=1000,
        save_path="/data/gpfs/projects/punim1989/biogo-hub/provinces_final/figures/img/drafts/fig_supp_distance_regressions_binned_finaldraft.png",
    )

    scatter_kws = {
        "color": "darkgray",
        "alpha": 0.75,
        "zorder": 2,
        "edgecolor": "k",
        "linewidth": 1.25,
    }

    print_with_timestamp("Creating pairplot with all data.")
    scatter_kws = {
        "color": "darkgray",
        "alpha": 0.75,
        "zorder": 2,
        "edgecolor": "k",
        "linewidth": 1.25,
    }
    create_pairplot(
        df,
        omit="BRITE KEGG_rclass",
        nrows=3,
        add_diagonals=True,
        save_path="/data/gpfs/projects/punim1989/biogo-hub/provinces_final/figures/img/drafts/fig_supp_distance_regressions_not_binned.png",
    )
    create_pairplot(
        df,
        omit=[],
        nrows=6,
        add_diagonals=True,
        save_path="/data/gpfs/projects/punim1989/biogo-hub/provinces_final/figures/img/drafts/fig_supp_distance_regressions_all_data_blocks.png",
        skip_func_func=False
    )
    columns = {
        "genomes": "Taxonomy",
        "sourmash": "$k$-mer signature",
        "env": "Environmental parameters",
        "KEGG_ko": "KEGG KO",
        "KEGG_Pathway": "KEGG Pathway",
        "KEGG_Reaction": "KEGG Reaction",
    }
    # distribution_plot(df, columns, box_kws={"alpha": 0.75})


if __name__ == "__main__":
    main()