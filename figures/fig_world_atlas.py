#!/usr/bin/env python

"""
Plot province area projections.
"""

import os
import re
import sys
from glob import glob
import argparse
import xarray as xr

from collections import OrderedDict

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    cut_tree,
    set_link_color_palette,
)
import geopandas as gpd

from pathlib import Path
import PIL
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from palettable import cartocolors
import xarray as xr

sys.path.insert(0, "/local/path/to/scripts/")
from utils import print_with_timestamp

sys.path.insert(0, "/local/path/to/scripts/")
from plotting_utils import palettes

os.environ["PATH"] = (
    os.environ["PATH"]
    + ":/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/texlive/20230313/bin/x86_64-linux/"
)
os.environ["CARTOPY_USER_BACKGROUNDS"] = (
    "/data/gpfs/projects/punim1989/biogo-hub/data/misc/NaturalEarth"
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.usetex"] = True
mpl.rc("text.latex", preamble=r"\usepackage{cmbright}")
PIL.Image.MAX_IMAGE_PIXELS = 233280001

get_k = lambda s: s.split("_k_")[-1].split("_")[0]


def list_to_dict_with_indices(input_list):
    item_to_index = {}
    for index, item in enumerate(input_list):
        item_to_index[item] = index
    return item_to_index


def find_ssp(string):
    pattern = re.compile(r"ssp(\d{3})")
    match = pattern.search(string)

    if match:
        return match.group()
    else:
        return None


palette = palettes["k_10"]


def leaf_label_func(id):
    leaves = {
        2877: 16,
        2862: 11,
        2896: 5,
        2897: 9,
        2880: 0,
        2881: 14,
        2891: 10,
        2893: 2,
        2888: 3,
        2894: 7,
    }
    label = f"{palette[leaves[id]]['label']}"
    return label


def link_color_func(id):
    link_color_dict = {
        2898: palette[9]["color"],
        2902: palette[11]["color"],
        2901: palette[14]["color"],
        2899: palette[3]["color"],
        2900: palette[2]["color"],
        2903: palette[10]["color"],
        2904: "black",
        2905: "black",
        2906: palette[16]["color"],
    }
    return link_color_dict[id]


def plot_models(
    model_name,
    model_data,
    sample_metadata,
    col_name,
    markersize,
    alpha,
    central_longitude,
    save,
    sample_plot=0,
    figsize=(40, 40),
    color_category="province",
    accuracy="Unknown",
    jitter=0.25,
    add_gridlines=False,
    markersize_sample=10,
    color_dict={v["label"]: v for k, v in palettes["k_10"].items()},
    currents=False,
):
    # Set projection
    projection_plate_carree = ccrs.PlateCarree()
    projection = ccrs.PlateCarree(central_longitude=central_longitude)

    # Create a figure and assign the GridSpec to it
    fig = plt.figure(figsize=figsize)
    gs = mpl.gridspec.GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.00,
        wspace=0.00,
        height_ratios=[0.3, 1],
        width_ratios=[1, 0.3],
        bottom=0.2,
        top=0.8,
    )

    # Define subplots
    ax1 = fig.add_subplot(gs[:-1, :-1])  # Top-left subplot
    ax2 = fig.add_subplot(gs[:-1, -1])  # Top-right subplot
    ax3 = fig.add_subplot(gs[-1:, :], projection=projection)  # Bottom subplot

    # ax1 - Dendrogram
    Z_prime = np.fromfile("/local/path/to/data/clustering/z_collapsed_dend_k10.csv")
    Z_prime = Z_prime.reshape(1453, 4)

    pmetadata_sorted = {str(k): v for k, v in color_dict.items()}
    pmetadata_sorted = {k: v for k, v in sorted(pmetadata_sorted.items())}
    try:
        print_with_timestamp("Plotting dendrogram.")
        with plt.rc_context({"lines.linewidth": 5}):
            D = dendrogram(
                Z_prime,
                color_threshold=0.99,
                orientation="top",
                leaf_font_size=30,
                ax=ax1,
                truncate_mode="lastp",
                p=10,
                get_leaves=True,
                leaf_label_func=leaf_label_func,
                link_color_func=link_color_func,
            )
    except KeyError:
        breakpoint()
    ax1.set_ylim(0.97, 1)
    ax1.tick_params(
        axis="y",
        labelsize=30,
    )

    # ------------------------------
    # ax3 - Atlas
    # ------------------------------
    # ax3.set_title(model_name + f" - {round(accuracy, 3)} accuracy", fontsize=20, fontweight="bold")
    ax3.set_global()
    # ax3.add_feature(cfeature.LAND, color="lightgray")
    ax3.add_feature(
        cfeature.NaturalEarthFeature("physical", "land", "10m"),
        ec="black",
        lw=1.2,
        alpha=1,
        color="#EFEFDB",
        zorder=2,
    )
    ax3.add_feature(cfeature.LAKES, ec="black", lw=1.2, zorder=2)
    ax3.add_feature(cfeature.RIVERS, zorder=2)
    ax3.coastlines(resolution="10m", lw=1.2, zorder=2)
    # ax3.background_img(name='NaturalEarthRelief', resolution='high')

    if sample_plot:
        print_with_timestamp(f"Downsampling plot to {sample_plot} data points.")
        model_data = model_data.sample(sample_plot)

    if "latitude" not in model_data.columns:
        model_data = model_data.reset_index()

    print_with_timestamp(f"Plotting '{model_name}'.")
    ax3.scatter(
        model_data["longitude"],
        model_data["latitude"],
        marker="o",
        s=markersize,
        color=model_data["color"],
        alpha=model_data["alpha"],
        rasterized=True,
        zorder=0,
    )

    # Add sample dots
    sample_metadata[color_category] = sample_metadata[col_name]
    color_dict_leg, marker_dict_leg = {}, {}
    # # Change this to avoid duplicate stations
    # for ix, row in sample_metadata.iterrows():
    for ix, row in sample_metadata.drop_duplicates("coords").iterrows():
        x, y = row["longitude"], row["latitude"]

        # Add jitter to x and y coordinates
        if jitter:
            x += np.random.uniform(-jitter, jitter)
            y += np.random.uniform(-jitter, jitter)
        try:
            color = color_dict[row[color_category]]["color"]
            marker = color_dict[row[color_category]]["marker"]
            (line,) = ax3.plot(
                x,
                y,
                marker=marker,
                # marker="o",
                markersize=markersize_sample,
                color=color,
                # color="darkgray",
                zorder=2,
                markeredgecolor="k",
                markeredgewidth=1,
                alpha=0.85,
                transform=projection_plate_carree,
            )
            color_dict_leg[row[color_category]] = color
            marker_dict_leg[row[color_category]] = marker
        except:
            breakpoint()
            raise

    patches = {
        str(label): Line2D(
            [0],
            [0],
            marker=marker_dict_leg[label],
            markersize=30,
            markerfacecolor=color,
            color=color,
            markeredgecolor="k",
            markeredgewidth=1.5,
            linestyle="None",
        )
        for label, color in color_dict_leg.items()
    }
    try:
        patches = {
            f"{color_dict[k]['label']} ({color_dict[k]['counts']})": v
            for k, v in sorted(patches.items())
        }
    except (KeyError, ValueError):
        patches = {
            f"{color_dict[int(k)]['label']} ({color_dict[int(k)]['counts']})": v
            for k, v in sorted(patches.items())
        }

    # Add gridlines
    if add_gridlines:
        gl = ax3.gridlines(
            draw_labels=True,
            linewidth=1.5,
            color="black",
            alpha=0.7,
            linestyle="--",
            ylabel_style={"size": 30, "color": "black", "weight": "bold"},
            xlabel_style={"size": 30, "color": "black", "weight": "bold"},
            auto_inline=True,
            ylocs=[-60, -40, -23.5, 0, 23.5, 40, 60],
            zorder=1,
        )
        gl.xlines = False
        gl.top_labels = False
        gl.bottom_labels = True
        gl.left_labels = True
        gl.right_labels = False
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()

    if currents:
        print_with_timestamp("Loading and plotting currents.")
        currents = gpd.read_file("/local/path/to/data/misc/ocean_currents.geojson")
        currents.plot(
            ax=ax3,
            transform=projection_plate_carree,
            alpha=1,
            color="lightgray",
            edgecolor="k",
            linewidth=0.5,
            zorder=1,
        )

    # ------------------------------
    # ax2 - Legend
    # ------------------------------
    print_with_timestamp("Plotting legend.")
    ax2.axis("off")
    # Specified order list
    order_list = ["B14", "B16", "B10", "B2", "B3", "B7", "B5", "B11", "B9", "B0"]
    order_list = [
        "BPLR",
        "BALT",
        "CTEM",
        "OTEM",
        "STEM",
        "MTEM",
        "TGYR",
        "TRLO",
        "TRHI",
        "APLR",
    ]
    sorted_keys = sorted(
        patches.keys(), key=lambda x: order_list.index(x.split(" ")[0])
    )

    # Create an OrderedDict with the sorted keys and corresponding values
    reordered_patches = OrderedDict((key, patches[key]) for key in sorted_keys)

    legend = ax2.legend(
        reordered_patches.values(),
        reordered_patches.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=1,
        fontsize=25,
    )
    legend.set_title("Province", prop={"weight": "bold", "size": 40})
    frame = legend.get_frame()
    fig.text(
        0.105,
        0.785,
        "\\textbf{A}",
        fontsize=40,
        ha="left",
        va="center",
    )
    fig.text(
        0.105,
        0.625,
        "\\textbf{B}",
        fontsize=40,
        ha="left",
        va="center",
    )

    # ------------------------------
    # PLOT FIGURE
    # ------------------------------
    N = len(model_data) if not sample_plot else sample_plot
    if save:
        model_dir = Path("/local/path/to/data/models/plots/")
        if not Path(model_dir).exists():
            Path(model_dir).mkdir()
        outfile = model_dir.joinpath(f"{model_name}")
        if isinstance(save, str):
            outfile = save
        elif isinstance(save, bool):
            outfile = model_dir.joinpath(f"{model_name}.png")
        print_with_timestamp(f"Saving figure to '{outfile}'.")
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
    else:
        plt.tight_layout()
        plt.show()


def get_model_accuracy(
    X, y, model_name, scaler=StandardScaler(), scale=True, random_state=4, test_size=0.2
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    (
        clf.fit(scaler.fit_transform(X_train), y_train)
        if scale
        else clf.fit(X_train, y_train)
    )
    y_pred = clf.predict(scaler.fit_transform(X_test)) if scale else clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = pd.DataFrame(
        confusion_matrix(y_test, y_pred), index=clf.classes_, columns=clf.classes_
    )
    report = classification_report(y_test, y_pred)
    print_with_timestamp(f"Model: {model_name}")
    print_with_timestamp(f"Accuracy: {accuracy}")
    print_with_timestamp("Confusion Matrix:")
    print(confusion)
    print_with_timestamp("Classification Report:")
    print(report)
    output_dict = {
        "model_name": model_name,
        "accuracy": accuracy,
        "confusion": confusion,
        "report": report,
    }
    return output_dict


def main(args):
    smd = pd.read_csv(args.metadata_file, index_col=0)
    env_cols = args.env_cols.split()
    (
        scaled,
        force_create_model,
        random_state,
        env_data_glob_string,
        test_size,
        markersize_sample,
        currents,
    ) = (
        args.scale,
        args.force,
        args.random_state,
        args.env_data_glob_string,
        args.test_size,
        args.markersize_sample,
        args.currents,
    )
    scaled_str = "scaled" if scaled else "not_scaled"
    scaler = StandardScaler()

    print_with_timestamp("Loading environmental data.")
    env_datasets = glob(env_data_glob_string)
    env_datasets = {k: xr.open_dataset(k, engine="netcdf4") for k in env_datasets}
    env_datasets = {v.title.split()[1]: v for k, v in env_datasets.items()}
    for k, v in env_datasets.items():
        vars = list(v.keys())
        var = [i for i in vars if i.endswith("_mean")][0]
        env_datasets[k] = v[var].rename(k)
    merged_env_data = xr.merge(list(env_datasets.values()))
    del env_datasets
    merged_env_data = merged_env_data.to_dataframe()[env_cols]
    merged_env_data = merged_env_data.dropna(how="any")
    N = len(merged_env_data) if not args.sample_model else int(args.sample_model)

    column = "sourmash_k_10_1487_25m"
    smd[column] = smd[column].map({k: v["label"] for k, v in palettes["k_10"].items()})

    print_with_timestamp(f"Fitting model for the following column: '{column}'.")
    X = smd.drop_duplicates("coords")[env_cols]
    y = smd.drop_duplicates("coords")[column]
    try:
        model_report = get_model_accuracy(
            X,
            y,
            column,
            scale=scaled,
            scaler=scaler,
            test_size=test_size,
            random_state=random_state,
        )
    except:
        print_with_timestamp(f"Failed to fit model '{column}'.")
        raise
    model_time = (
        "" if "baseline" in env_data_glob_string else find_ssp(env_data_glob_string)
    )
    model_name = f"{model_time}{column}_{scaled_str}_{N}_points"
    model_file = f"/local/path/to/data/models/model_data/{model_name}.joblib"
    X_test_file = f"/local/path/to/data/models/model_data/{model_name}_X_test.csv"
    y_hat_file = f"/local/path/to/data/models/model_data/{model_name}_y_hat.csv"
    if (
        not all([Path(file).exists() for file in [model_file, X_test_file, y_hat_file]])
    ) or force_create_model:
        # Initialize and train the RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)

        # Make predictions on the test set
        X_test = merged_env_data.sample(int(N)).dropna(how="any")
        print_with_timestamp(f"Fitting model '{model_name}'.")
        X_test = (
            pd.DataFrame(
                scaler.fit_transform(X_test), index=X_test.index, columns=X_test.columns
            )
            if scaled
            else X_test
        )
        (
            clf.fit(
                pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns),
                y,
            )
            if scaled
            else clf.fit(X, y)
        )
        y_hat = clf.predict(X_test)
        y_hat = pd.DataFrame(y_hat, index=X_test.index, columns=["province"])
        X_test[clf.classes_] = clf.predict_proba(X_test)
        X_test["province"] = y_hat
        X_test["color"] = X_test["province"].map(
            {v["label"]: v["color"] for k, v in palettes["k_10"].items()}
        )
        X_test.columns = [str(i) for i in X_test.columns]
        X_test["alpha"] = X_test.apply(
            lambda row: row[str(row["province"])] ** 2, axis=1
        )

        # Save the model to disk
        X_test.to_csv(X_test_file)
        y_hat.to_csv(y_hat_file)
        joblib.dump(clf, model_file)
    else:
        print_with_timestamp(f"Loading model '{model_name}' from disk.")
        clf = joblib.load(model_file)
        X_test = pd.read_csv(X_test_file)
        if "province" not in X_test.columns:
            y_hat = pd.read_csv(
                y_hat_file,
                usecols=[
                    "province",
                ],
            )
            X_test["province"] = y_hat

    sample_plot = int(args.sample_plot) if args.sample_plot else 0
    if isinstance(args.outfile, (str, bool)):
        save = args.outfile
    plot_models(
        model_data=X_test,
        model_name=model_name,
        col_name=column,
        color_category="province",
        markersize=args.markersize,
        alpha=0.5,
        figsize=(40, 40),
        central_longitude=0,
        save=save,
        sample_plot=sample_plot,
        sample_metadata=smd,
        accuracy=model_report["accuracy"],
        add_gridlines=args.gridlines,
        markersize_sample=markersize_sample,
        currents=currents,
    )
    print_with_timestamp("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot province models.")
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Scale environmental data before fitting the model.",
    )
    parser.add_argument(
        "--sample-model",
        type=float,
        help="Number of data points to fit the model to.",
        default=1e5,
    )
    parser.add_argument(
        "--sample-plot",
        type=float,
        help="Number of data points to use for plotting.",
        default=1e5,
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        help="Metadata file to look for labels.",
        default="/local/path/to/data/metadata_1454_cluster_labels.csv",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        default=True,
        help="Path to output file. If not provided, will be automatically generated.",
    )
    parser.add_argument(
        "-m",
        "--markersize",
        type=float,
        help="Marker size for each data point in test dataset.",
        default=1,
    )
    parser.add_argument(
        "-ms",
        "--markersize-sample",
        type=float,
        help="Marker size for each data point the sample dataset.",
        default=12,
    )
    parser.add_argument(
        "-t",
        "--test-size",
        type=float,
        help="Size of test dataset in relation to training dataset.",
        default=0.2,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Whether to force to create the model.",
    )
    parser.add_argument(
        "-gl",
        "--gridlines",
        action="store_true",
        help="Whether to add gridlines.",
        default=False,
    )
    parser.add_argument(
        "-r",
        "--random-state",
        type=int,
        default=0,
        help="Random state to use for the model. Default is 42.",
    )
    parser.add_argument(
        "-s",
        "--env-data-glob-string",
        type=str,
        default="/local/path/to/data/environmental/*baseline*.nc",
        help="Glob string to find the environmental datasets used for testing.",
    )
    parser.add_argument(
        "-e",
        "--env-cols",
        type=str,
        default="Salinity Nitrate OceanTemperature DissolvedMolecularOxygen Silicate pH DissolvedIron Phosphate SeaIceCover Chlorophyll",
        help="Environmental variables to use for the model (space-separated). Default is 'Salinity Nitrate OceanTemperature DissolvedMolecularOxygen Silicate pH DissolvedIron Phosphate SeaIceCover Chlorophyll'.",
    )
    parser.add_argument(
        "--currents",
        action="store_true",
        help="Whether to plot ocean currents.",
    )
    args = parser.parse_args()
    main(args)
