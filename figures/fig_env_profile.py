#!/usr/bin/env python

"""
Plot environmental profile (color-coded PCA).
"""
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

sys.path.insert(0, "/local/path/to/scripts/")
from plotting_utils import palettes, split_on_capitals

sys.path.insert(0, "/local/path/to/scripts/")
from utils import print_with_timestamp

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

os.environ["PATH"] = (
    os.environ["PATH"]
    + ":/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/texlive/20230313/bin/x86_64-linux/"
)
plt.rcParams["font.sans-serif"] = "CMU Sans Serif"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["text.usetex"] = True
mpl.rc("text.latex", preamble=r"\usepackage{cmbright}")

from cartopy import crs as ccrs
from cartopy import feature as cfeature

from scipy.stats import yeojohnson
from sklearn.preprocessing import StandardScaler


from pca import pca


def map_to_color_channel(component):
    max_abs_value = np.max(np.abs(component))
    color_channel = 128 * (1 + component / max_abs_value)
    return np.clip(color_channel, 0, 255)  # Clip values to the range [0, 255]


def main():
    print_with_timestamp("Processing data and performing PCA.")
    df = pd.read_csv("/local/path/to/data/distances/sourmash.csv", index_col=0)
    md = pd.read_csv(
        "/local/path/to/data/metadata_1454_cluster_labels.csv", index_col=0
    )

    df = df.loc[md.index, md.index]

    env_cols = pd.Index(
        [
            "Salinity",
            "pH",
            "Nitrate",
            "OceanTemperature",
            "DissolvedMolecularOxygen",
            "Silicate",
            "DissolvedIron",
            "Phosphate",
            "SeaIceCover",
            "Chlorophyll",
        ],
    )

    md = md.drop_duplicates(subset=env_cols)

    data = md[env_cols].values
    # Apply Box-Cox transformation to each column
    transformed_data = np.empty_like(data, dtype=float)
    lambda_values = []

    for i in range(data.shape[1]):
        transformed_column, lambda_value = yeojohnson(
            data[:, i] + 1
        )  # Adding 1 to handle zero values
        transformed_data[:, i] = transformed_column
        lambda_values.append(lambda_value)

    X = StandardScaler().fit_transform(transformed_data)

    labels = env_cols
    model = pca(n_components=3, normalize=True)
    results = model.fit_transform(X, col_labels=labels)

    df_comp = results["PC"]
    df_comp.index = md.index

    # Assuming df_comp is your DataFrame with columns PC1, PC2, and PC3

    # Step 1: Standardize the principal components (rescale to have unit variance)
    scaler = StandardScaler()
    df_comp_standardized = pd.DataFrame(
        scaler.fit_transform(df_comp), columns=df_comp.columns
    )

    # Step 2: Decorrelate the components using the Mahalanobis transformation
    cov_matrix = np.cov(df_comp_standardized, rowvar=False)
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    df_comp_decorrelated = pd.DataFrame(
        np.dot(df_comp_standardized, inv_cov_matrix), columns=df_comp.columns
    )

    # Apply the mapping to each principal component
    df_color_channels = df_comp_decorrelated.apply(map_to_color_channel)

    # Step 4: Combine the color channels to attribute a single composite color to each station
    composite_colors = pd.DataFrame(
        {
            "Blue": df_color_channels["PC1"],
            "Red": df_color_channels["PC2"],
            "Green": df_color_channels["PC3"],
        }
    )

    # Optional: Convert the composite color values to integers
    composite_colors = composite_colors.astype(int)

    # Display the resulting DataFrame with composite colors
    def rgb_to_hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(
            int(rgb["Red"]), int(rgb["Green"]), int(rgb["Blue"])
        )

    # Apply the rgb_to_hex function to each row (axis=1)
    composite_colors["HexCode"] = composite_colors.apply(rgb_to_hex, axis=1)
    composite_colors["HexCode"].index = md.index
    md["color"] = composite_colors["HexCode"]

    df_all = pd.read_csv(
        "/local/path/to/data/models/model_data/sourmash_k_10_1487_25m_not_scaled_16270117_points_X_test.csv",
        index_col=0,
    )

    palette_composite_fig = {
        v["label"]: v["color"] for k, v in palettes["k_10"].items()
    }

    # Create a figure instance
    fig = plt.figure(figsize=(20, 18))

    # Define the grid specification
    gs = mpl.gridspec.GridSpec(
        4, 4, figure=fig, width_ratios=[1.2, 0.3, 0.3, 0.3], wspace=0.1, hspace=0.3
    )  # , hspace=0.5, wspace=0.5)

    # Define projection
    projection = ccrs.PlateCarree()

    # Add subplot for the map
    map_ax = fig.add_subplot(gs[2:, :2], projection=projection)
    map_ax.set_extent([-180, 180, -90, 90], crs=projection)

    # Add Natural Earth data
    map_ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m"),
        ec="black",
        lw=0.5,
        alpha=1,
        color="#EFEFDB",
        zorder=1,
    )
    map_ax.add_feature(cfeature.LAKES, ec="black", lw=0.5, zorder=1)
    map_ax.add_feature(cfeature.RIVERS, zorder=1)
    map_ax.coastlines(resolution="10m", lw=0.5, zorder=1)
    gl = map_ax.gridlines(
        draw_labels=True,
        linewidth=0.25,
        color="black",
        alpha=0.7,
        linestyle="--",
        ylabel_style={"size": 8, "color": "black"},
        xlabel_style={"size": 8, "color": "black"},
        auto_inline=True,
        ylocs=[-60, -40, -23.5, 0, 23.5, 40, 60],
        zorder=0,
    )
    gl.xlines = False
    gl.right_labels = False
    gl.top_labels = False
    jitter = 0.5
    for ix, row in md.iterrows():
        x, y, c, p = (
            row["longitude"],
            row["latitude"],
            row["color"],
            row["sourmash_k_10_1487_25m"],
        )
        m = palettes["k_10"][p]["marker"]

        # Add jitter to x and y coordinates
        if jitter:
            x += np.random.uniform(-jitter, jitter)
            y += np.random.uniform(-jitter, jitter)
        (line,) = map_ax.plot(
            x,
            y,
            marker=m,
            markersize=10,
            color=c,
            zorder=3,
            markeredgecolor="k",
            alpha=0.85,
        )
        handle = mpl.lines.Line2D(
            [], [], color="black", marker=m, markersize=10, label=p
        )
    legend_handles = [
        (
            palettes["k_10"][i]["marker"],
            palettes["k_10"][i]["label"],
            palettes["k_10"][i]["color"],
        )
        for i in [14, 16, 10, 2, 3, 7, 5, 11, 9, 0]
    ]
    legend_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color=c,
            linestyle="None",
            marker=m,
            markersize=12,
            markeredgecolor="black",
            markeredgewidth=1,
            label=p,
        )
        for (m, p, c) in legend_handles
    ]
    map_ax.legend(handles=legend_handles, ncols=10, bbox_to_anchor=(0.985, 1.1))

    # Add subplot for the biplot
    biplot_ax = fig.add_subplot(gs[:2, 0])
    _, _ = model.biplot(
        c=md["color"],
        marker=md["sourmash_k_10_1487_25m"]
        .map({k: v["marker"] for k, v in palettes["k_10"].items()})
        .values,
        ax=biplot_ax,
        fontweight="normal",
        legend=None,
        title=None,
        arrowdict={
            "weight": "bold",
        },
        visible=True,
        fig=fig,
    )
    biplot_ax.tick_params(axis="x", labelsize=12)
    biplot_ax.tick_params(axis="y", labelsize=12)
    biplot_ax.set_xlabel(biplot_ax.get_xlabel().replace("%", r"\%"), fontsize=14)
    biplot_ax.set_ylabel(biplot_ax.get_ylabel().replace("%", r"\%"), fontsize=14)
    biplot_ax.set_title(biplot_ax.get_title().replace("%", r"\%"), fontsize=16)

    # Env cols:
    axes = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
    ]
    env_cols = [
        "OceanTemperature",
        "Salinity",
        "Nitrate",
        "Phosphate",
        "DissolvedMolecularOxygen",
        "SeaIceCover",
        "pH",
        "Silicate",
        "Chlorophyll",
        "DissolvedIron",
    ]
    units = {
        "OceanTemperature": r"$^{\circ}$C",
        "Salinity": "PSU",
        "Nitrate": r"mmol/m$^3$",
        "Phosphate": r"mmol/m$^3$",
        "DissolvedMolecularOxygen": r"mmol/m$^3$",
        "SeaIceCover": r"\%",
        "pH": "",
        "Silicate": r"mmol/m$^3$",
        "Chlorophyll": r"mg/m$^3$",
        "DissolvedIron": r"mmol/m$^3$",
    }
    df_all["SeaIceCover"] = df_all["SeaIceCover"] * 100
    plot_data = df_all.copy()
    # plot_data = df_all.sample(int(1e5), random_state=42)
    del df_all
    plot_data["province"] = (
        plot_data["province"]
        .astype(int)
        .map({k: v["label"] for k, v in palettes["k_10"].items()})
    )
    for col, (i, j) in zip(env_cols, axes):
        print_with_timestamp(f"Plotting '{col}' boxenplot.")
        ax = fig.add_subplot(gs[i, j])
        # sns.stripplot(plot_data, x=col, y="province", ax=ax, hue="province", palette=palette_composite_fig, alpha=0.25, size=2, order = [palettes["k_10"][i]["label"] for i in [14, 16, 10, 2, 3, 7, 5, 11, 9, 0]], zorder=1, linewidth=0.25, edgecolor="k")
        sns.boxenplot(
            plot_data,
            x=col,
            y="province",
            ax=ax,
            hue="province",
            palette=palette_composite_fig,
            order=[
                palettes["k_10"][i]["label"] for i in [14, 16, 10, 2, 3, 7, 5, 11, 9, 0]
            ],
            showfliers=False,
            zorder=2,
        )
        ax.set_ylabel("")
        ytick_cols = "Nitrate SeaIceCover Silicate DissolvedIron".split()
        ax.yaxis.tick_right()
        if col not in ytick_cols:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
        ax.xaxis.set_major_locator(
            mpl.ticker.MaxNLocator(nbins=5, min_n_ticks=4, integer=True)
        )
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
        unit = f" ({units[col]})" if len(units[col]) > 0 else ""
        ax.set_xlabel(f"{split_on_capitals(col)}{unit}", fontsize=14)
        ax.xaxis.label.set_size(12)
        ax.xaxis.set_label_position("top")
    fig.text(
        0.105, 0.89, "\\textbf{A}", fontsize=20, fontweight="bold", ha="left", va="top"
    )
    fig.text(
        0.105, 0.47, "\\textbf{B}", fontsize=20, fontweight="bold", ha="left", va="top"
    )
    fig.text(
        0.54, 0.89, "\\textbf{C}", fontsize=20, fontweight="bold", ha="left", va="top"
    )

    outfile = "/local/path/to/figures/img/final_vectors/fig2_env.pdf"
    plt.savefig(
        outfile, dpi=600 if outfile[-3:] == "png" else "figure", bbox_inches="tight"
    )
    print_with_timestamp(f"Done. Wrote figure to '{outfile}'.")


if __name__ == "__main__":
    main()
