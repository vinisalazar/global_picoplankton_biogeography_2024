#!/usr/bin/env python
# coding: utf-8

"""
Plot taxonomic profile.
"""

import os
import sys


from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
from functools import lru_cache

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

os.environ["PATH"] = (
    os.environ["PATH"]
    + ":/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/texlive/20230313/bin/x86_64-linux/"
)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.usetex"] = True
mpl.rc('text.latex', preamble=r'\usepackage{cmbright}')

env = "local"
wd = "/local/path/to/" if env == "local" else "/local/path/to/"

sys.path.insert(0, f"{wd}/scripts/")
from plotting_utils import palettes


smd = pd.read_csv(f"{wd}/provinces_final/data/metadata_1454_cluster_labels.csv", index_col=0, dtype={"sourmash_k_10_1487_25m": "object"})
gmd = pd.read_csv(f"{wd}/provinces_final/data/genome_metadata.tsv", sep="\t", index_col=0)
tables_clean = {
    "genomes": pd.read_csv(f"{wd}/provinces_final/data/counts/genomes_trimmed_mean_formatted_clean.csv", index_col=0),
    "genus": pd.read_csv(f"{wd}/provinces_final/data/counts/genus_trimmed_mean_formatted_clean.csv", index_col=0),
    "family": pd.read_csv(f"{wd}/provinces_final/data/counts/family_trimmed_mean_formatted_clean.csv", index_col=0),
    "class": pd.read_csv(f"{wd}/provinces_final/data/counts/class_trimmed_mean_formatted_clean.csv", index_col=0),
    "order": pd.read_csv(f"{wd}/provinces_final/data/counts/order_trimmed_mean_formatted_clean.csv", index_col=0),
    "phylum": pd.read_csv(f"{wd}/provinces_final/data/counts/phylum_trimmed_mean_formatted_clean.csv", index_col=0),
    # "BRITE": pd.read_csv(f"{wd}/provinces_final/data/counts/BRITE_trimmed_mean_formatted_clean.csv", index_col=0),
}

cim = pd.read_csv(f"{wd}/provinces_final/data/R/drafts/splsda_all_genomes_ncomp10_keepX3_cim.csv", index_col=0)

for k, v in tables_clean.items():
    unclassified = [i for i in v.columns if "unclassified" in i]
    tables_clean[k] = v.drop(columns=unclassified)


# New genus entry for most abundant genome
gmd.loc["GCA_902624425", "genus"] = "g__GCA_902624425_gen_nov"
tables_clean["genus"]["g__GCA_902624425_gen_nov"] = tables_clean["genomes"]["GCA_902624425"]

ncomp = 20
keepX = 5
rank = "genomes"
ranks = "domain phylum class order family genus sci_names".split()

for rank_ in ranks[:-1]:
    for genome in tables_clean["genomes"].columns:
        parent = gmd.loc[genome, rank_]
        if parent == f"{rank_[0]}__":
            parent = gmd.loc[genome, rank_] = f"{rank_[0]}__{genome}"
            tables_clean[rank_][f"{rank_[0]}__{genome}"] = tables_clean["genomes"][genome]

tables_clean["domain"] = tables_clean["genomes"].T.join(gmd["domain"]).groupby("domain").sum().T


def get_ordered_labels(df):
    distance_matrix = pdist(df, metric="euclidean")
    Z = linkage(distance_matrix, method="average")
    dendro = dendrogram(Z, no_plot=True, labels=df.index)
    ordered_labels = dendro['ivl']
    return ordered_labels


# Function to sort each category group based on the predefined order
def sort_by_order(group, order):
    return group.loc[order]


def get_rel_abun(df):
    df = (df.T / df.T.sum()).T
    return df


ranks = "domain phylum class order family genus genomes".split()

colormaps = {
    'p__Proteobacteria': '#1f77b4',  # Blue
    'p__Bacteroidota': 'maroon',
    'p__Cyanobacteria': '#2ca02c',   # Green
    'p__Actinobacteriota': 'darkorange',
    'p__Firmicutes': 'teal',
    'p__Halobacteriota': 'slateblue',
    'p__Marinisomatota': 'steelblue',
    'p__Verrucomicrobiota': 'darkviolet',
    'p__Thermoplasmatota': 'darkviolet',
    'p__Methanobacteriota': 'slategrey',
    'p__Thermoproteota': 'darkmagenta',
    'p__Nanoarchaeota': 'olive',
    'p__SAR324': 'darkgoldenrod',
    'p__Planctomycetota': 'darkgoldenrod'
}

tab20_colors = mpl.colormaps["tab20"].colors[::2] + mpl.colormaps["tab20"].colors[1::2]
colormaps = dict(zip(colormaps.keys(), tab20_colors))

tab20_colors_bar = mpl.colormaps["tab20"].colors[::2] + mpl.colormaps["tab20"].colors[1::2]
colormaps_bar = dict(zip(colormaps.keys(), tab20_colors_bar))


def get_parent(taxa, rank):
    if rank == "domain":
        return "root"
    parent_rank = ranks[ranks.index(rank) - 1]
    return gmd.reset_index(names=["genomes"])[gmd.reset_index(names=["genomes"])[rank] == taxa].drop_duplicates(rank)[parent_rank].values[0]


def get_phylum(taxa, rank):
    if rank == "phylum":
        return taxa
    if rank == "domain":
        return "root"
    if rank == "genomes":
        return gmd.loc[taxa, "phylum"]
    return gmd.drop_duplicates(rank).set_index(rank).loc[taxa, "phylum"]


def hex_to_rgb(hex_color):
    """Convert hex color string to an RGB tuple."""
    return mpl.colors.hex2color(hex_color)

def adjust_brightness(color, factor):
    """
    Adjust the brightness of an RGB color towards white.
    
    Parameters:
    - color: tuple, the original RGB color.
    - factor: float, range [0, 1], where 0 is the original color and 1 is white.
    
    Returns:
    - A new RGB tuple representing the adjusted color.
    """
    return tuple((1 - factor) * c + factor * 1 for c in color)

@lru_cache
def create_brightness_colormap(hex_color, brightness_factor=0.8, name='custom_brightness'):
    """
    Create a colormap that transitions from the given hex color to a brighter version.
    
    Parameters:
    - hex_color: str, the starting hex color.
    - brightness_factor: float, range [0, 1], how close the final color is to white.
    - name: str, name of the colormap.
    
    Returns:
    - A LinearSegmentedColormap object.
    """
    if isinstance(hex_color, str):
        start_color = hex_to_rgb(hex_color) if hex_color.startswith("#") else hex_to_rgb(mpl.colors.to_hex(hex_color))
    elif isinstance(hex_color, tuple):
        start_color = hex_color
    elif hex_color != hex_color:
        start_color = pd.Series(colormaps).sample(1).iloc[0]
    end_color = adjust_brightness(start_color, brightness_factor)
    
    # Create a colormap from these colors
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, [start_color, end_color])
    return cmap

@lru_cache
def get_colormap_colors(color, num_colors):

    # Generate the colors
    cmap = create_brightness_colormap(color)

    colors = [mpl.colors.to_hex(cmap(i)) for i in list(np.linspace(0.0, 1, num_colors))]
    
    return colors





def create_pie_data(province="9", N=10, colormaps=colormaps, add_archaea=False, add_other_phyla=False, add_other_phyla_N=10):

    # Select province samples
    samples = smd[smd["sourmash_k_10_1487_25m"] == province].index
    pie_data = tables_clean["genomes"].loc[samples]

    # Convert to relative abundance
    pie_data = get_rel_abun(pie_data)

    # Select 10 most abundant genomes (they're all bacterial)
    most_abun = pie_data.median().nlargest(N).index.tolist()

    # Add 2 most abundant Archaeal genomes
    if add_archaea:
        most_abun += pie_data.median().loc[gmd[gmd["domain"] == "d__Archaea"].index].nlargest(2).index.tolist()
    if add_other_phyla:
        most_abun += pie_data.median().loc[gmd[~gmd["phylum"].isin(add_other_phyla)].index].nlargest(add_other_phyla_N).index.tolist()
    pie_data = pie_data[most_abun]
    pie_data = pie_data.median()

    # Get lineages
    lineages = gmd.loc[pie_data.index, ranks[:-1]]
    lineages = {col: lineages[col].unique().tolist() for col in lineages.columns}

    for k, v in lineages.items():
        # This is to debug unindentified lineages
        try:
            lineages[k] = get_rel_abun(tables_clean[k].loc[samples])[v].median()
        except KeyError:
            print(k, v)
            raise
    lineages["genomes"] = pie_data
    for k, v in lineages.items():
        lineages[k] = lineages[k].reset_index()
        lineages[k].columns = ["taxa", "value"]
        lineages[k]["rank"] = k

    pie_data = pd.concat([lineages[k] for k in lineages.keys()]).reset_index(drop=True)

    pie_data["parent"] = pie_data.apply(lambda x: get_parent(x["taxa"], x["rank"]), axis=1)

    pie_data["rank"] = pd.Categorical(pie_data["rank"], categories=ranks, ordered=True)
    pie_data = pie_data.sort_values("rank").reset_index(drop=True)

    # colormaps = dict(zip(color_ranks, colormaps))
    colormaps["root"] = "#9a9a9a"

    pie_data["color"] = pie_data.apply(lambda row: get_phylum(row["taxa"], row["rank"]), axis=1).map(colormaps)
    pie_data["color"] = pie_data.apply(lambda row: get_colormap_colors(row["color"], len(ranks))[ranks.index(row["rank"])], axis=1)

    return pie_data





# Phylum
print("Formatting data for phylum.\n")
stacked_bar_phylum = tables_clean["phylum"]
stacked_bar_phylum = (stacked_bar_phylum.T / stacked_bar_phylum.T.sum()).T
most_abun = stacked_bar_phylum.median().nlargest(10).index.tolist()
stacked_bar_phylum = stacked_bar_phylum[most_abun].sample(len(stacked_bar_phylum)) * 100
stacked_bar_phylum["Others"] = 100 - stacked_bar_phylum.sum(axis=1)
stacked_bar_phylum = stacked_bar_phylum.join(smd["sourmash_k_10_1487_25m"])
stacked_bar_phylum["sourmash_k_10_1487_25m"] = stacked_bar_phylum["sourmash_k_10_1487_25m"].astype(int).map({k: v["label"] for k, v in palettes["k_10"].items()})
stacked_bar_phylum["sourmash_k_10_1487_25m"] =  pd.Categorical(stacked_bar_phylum["sourmash_k_10_1487_25m"],
                                                             categories="APLR BPLR BALT CTEM OTEM STEM MTEM TGYR TRHI TRLO".split(),
                                                             ordered=True)
sort_order_phylum = {}
for category in stacked_bar_phylum["sourmash_k_10_1487_25m"].unique():
    sort_order_phylum[category] = get_ordered_labels(stacked_bar_phylum[stacked_bar_phylum["sourmash_k_10_1487_25m"] == category].drop("sourmash_k_10_1487_25m", axis=1))
stacked_bar_phylum = stacked_bar_phylum.groupby("sourmash_k_10_1487_25m", group_keys=False).apply(lambda x: sort_by_order(x, sort_order_phylum[x["sourmash_k_10_1487_25m"].iloc[0]]))
stacked_bar_phylum.columns = [i[3:] if i[1] == "_" else i for i in stacked_bar_phylum.columns]


colordict_phylum = {k: colormaps_bar["p__" + k] for k in stacked_bar_phylum.columns[:-1] if k != "Others"}
colordict_phylum["Others"] = "#9a9a9a"





# Family
print("Formatting data for family.\n")
stacked_bar_family = tables_clean["family"]
stacked_bar_family = (stacked_bar_family.T / stacked_bar_family.T.sum()).T
most_abun = stacked_bar_family.median().nlargest(10).index.tolist()
stacked_bar_family = stacked_bar_family[most_abun].sample(len(stacked_bar_family)) * 100
stacked_bar_family["Others"] = 100 - stacked_bar_family.sum(axis=1)
stacked_bar_family = stacked_bar_family.join(smd["sourmash_k_10_1487_25m"])
stacked_bar_family["sourmash_k_10_1487_25m"] = stacked_bar_family["sourmash_k_10_1487_25m"].astype(int).map({k: v["label"] for k, v in palettes["k_10"].items()})
stacked_bar_family["sourmash_k_10_1487_25m"] =  pd.Categorical(stacked_bar_family["sourmash_k_10_1487_25m"],
                                                             categories="APLR BPLR BALT CTEM OTEM STEM MTEM TGYR TRHI TRLO".split(),
                                                             ordered=True)
sort_order_family = {}
for category in stacked_bar_family["sourmash_k_10_1487_25m"].unique():
    sort_order_family[category] = get_ordered_labels(stacked_bar_family[stacked_bar_family["sourmash_k_10_1487_25m"] == category].drop("sourmash_k_10_1487_25m", axis=1))
stacked_bar_family = stacked_bar_family.groupby("sourmash_k_10_1487_25m", group_keys=False).apply(lambda x: sort_by_order(x, sort_order_family[x["sourmash_k_10_1487_25m"].iloc[0]]))
stacked_bar_family.columns = [i[3:] if i[1] == "_" else i for i in stacked_bar_family.columns]

colordict_family = [
    'Pelagibacteraceae',
    'Flavobacteriaceae',
    'Cyanobiaceae',
    'D2472',
    'Rhodobacteraceae',
    'TMED189',
    'AAA536-G10',
    'GCA-002718135',
    'AG-339-G14',
    'Puniceispirillaceae'
    ]

# # Old

colordict_family = {k: colormaps_bar[gmd.drop_duplicates("family").set_index("family").loc[f"f__{k}", "phylum"]] for k in stacked_bar_family.columns[:-1] if k != "Others"}
colordict_family["Others"] = "#9a9a9a"
for phylum, phylum_color in colordict_phylum.items():
    phylum_brightness = 0
    for family, family_color in colordict_family.items():
        if "f__" + family in gmd[gmd["phylum"] == f"p__{phylum}"]["family"].to_list():
            colordict_family[family] = adjust_brightness(family_color, phylum_brightness)
            phylum_brightness += 0.15





rename_families = gmd.drop_duplicates("family").set_index("family").loc[["f__" + i for i in stacked_bar_family.columns[:-2]]].iloc[[3, 6, 7, 8]]["order"].to_dict()
rename_families = {k[3:]: f"{k[3:]}\n({v[3:]})" for k, v in rename_families.items()}
rename_families = {k: v.replace("\n", " ") if "D2472" in v else v for k, v in rename_families.items()}
stacked_bar_family = stacked_bar_family.rename(columns=rename_families)





renamed_colordict_family = {rename_families[k]: v for k, v in colordict_family.items() if k in rename_families.keys()}
colordict_family = {**colordict_family, **renamed_colordict_family}
colordict_family = {k: colordict_family[k] for k in stacked_bar_family.columns[:-1] if k != "Others"}
colordict_family = {k: v for k, v in colordict_family.items() if k in stacked_bar_family.columns}
colordict_family["Others"] = "#9a9a9a"





median_archaea = get_rel_abun(tables_clean["phylum"])[gmd[gmd["domain"] == "d__Archaea"]["phylum"].unique()].sum(axis=1).to_frame().merge(smd["sourmash_k_10_1487_25m"], left_index=True, right_index=True).groupby("sourmash_k_10_1487_25m").median().reset_index()
median_archaea.columns = ["province", "archaea"]
median_archaea["province_label"] = median_archaea["province"].astype(int).map({k: v["label"] for k, v in palettes["k_10"].items()})
median_archaea[median_archaea["archaea"] >= 0.02]["province"].unique()





provinces_pie_data = dict()

for p in "14 16 10 2 3 7 5 11 9 0".split():
    print(p)
    add_archaea = True if p in median_archaea[median_archaea["archaea"] >= 0.02]["province"].unique() else False
    add_other_phyla = False if p not in "7 14 0 10 2".split() else ["p__Proteobacteria", "p__Bacteroidota"]
    add_other_phyla = add_other_phyla if p != "3" else ["p__Proteobacteria", "p__Bacteroidota", "p__Cyanobacteria"]
    provinces_pie_data[p] = create_pie_data(province=p, add_archaea=add_archaea, add_other_phyla=add_other_phyla, add_other_phyla_N=10)











def sunburst(pie_data, ax, size=0.15, radius=0.15, labeldistance=1, pctdistance=1, remove_labels=True, remove_pct=True, remove_all_labels=False, rotatelabels=False, format_lineage=False, genome_fontsize=8):

    domain_data = pie_data[pie_data["rank"] == "domain"]
    domain_data.loc[0, "color"] = "#D3D3D3"
    if len(domain_data) > 1:
        domain_data.loc[1, "color"] = "#929292"
    wedges, texts, autotexts = ax.pie(domain_data["value"],
                           labels=domain_data["taxa"],
                           normalize=False,
                           autopct="%1.1f%%",
                           colors=domain_data["color"],
                           radius=radius,
                           labeldistance=0.1)
    rank_wedges = defaultdict(list)

    for wedge, text, autotext in zip(wedges, texts, autotexts):
        theta1, theta2 = wedge.theta1, wedge.theta2
        data = pie_data[pie_data["parent"] == text.get_text()]
        rank_wedges["phylum"].append(ax.pie(data["value"],
            labels=data["taxa"],
            colors=data["color"],           
            radius=radius+size,
            normalize=False,
            autopct="%1.1f%%",
            labeldistance=labeldistance,
            pctdistance=pctdistance,
            wedgeprops=dict(width=size, edgecolor='w'),
            startangle=theta1,
            )
        )

    for domain in rank_wedges["phylum"]:
        for wedge, text, autotext in zip(*domain):
            theta1, theta2 = wedge.theta1, wedge.theta2
            data = pie_data[pie_data["parent"] == text.get_text()]
            rank_wedges["class"].append(ax.pie(data["value"],
                labels=data["taxa"],
                colors=data["color"],
                radius=radius+2*size,
                normalize=False,
                autopct="%1.1f%%",
                labeldistance=labeldistance,
                pctdistance=pctdistance,
                wedgeprops=dict(width=size, edgecolor='w'),
                startangle=theta1,
                )
            )

    for phylum in rank_wedges["class"]:
        for wedge, text, autotext in zip(*phylum):
            theta1, theta2 = wedge.theta1, wedge.theta2
            data = pie_data[pie_data["parent"] == text.get_text()]
            rank_wedges["order"].append(ax.pie(data["value"],
                labels=data["taxa"],
                colors=data["color"],
                radius=radius+3*size,
                normalize=False,
                autopct="%1.1f%%",
                labeldistance=labeldistance,
                pctdistance=pctdistance,
                wedgeprops=dict(width=size, edgecolor='w'),
                startangle=theta1,
                )
            )

    for class_ in rank_wedges["order"]:
        for wedge, text, autotext in zip(*class_):
            theta1, theta2 = wedge.theta1, wedge.theta2
            data = pie_data[pie_data["parent"] == text.get_text()]
            rank_wedges["family"].append(ax.pie(data["value"],
                labels=data["taxa"],
                colors=data["color"],
                radius=radius+4*size,
                normalize=False,
                autopct="%1.1f%%",
                labeldistance=labeldistance,
                wedgeprops=dict(width=size, edgecolor='w'),
                startangle=theta1,
                )
            )

    for order in rank_wedges["family"]:
        for wedge, text, autotext in zip(*order):
            theta1, theta2 = wedge.theta1, wedge.theta2
            data = pie_data[pie_data["parent"] == text.get_text()]
            rank_wedges["genus"].append(ax.pie(data["value"],
                labels=data["taxa"],
                colors=data["color"],
                radius=radius+5*size,
                normalize=False,
                autopct="%1.1f%%",
                labeldistance=labeldistance,
                pctdistance=pctdistance,
                wedgeprops=dict(width=size, edgecolor='w'),
                startangle=theta1,
                )
            )

    for family in rank_wedges["genus"]:
        for wedge, text, autotext in zip(*family):
            theta1, theta2 = wedge.theta1, wedge.theta2
            data = pie_data[pie_data["parent"] == text.get_text()]
            rank_wedges["genomes"].append(ax.pie(data["value"],
                labels=data["taxa"],
                colors=data["color"],
                radius=radius+6*size,
                normalize=False,
                autopct="%1.1f%%",
                labeldistance=labeldistance,
                pctdistance=pctdistance,
                wedgeprops=dict(width=size, edgecolor='w'),
                textprops=dict(fontsize=genome_fontsize),
                startangle=theta1,
                rotatelabels=rotatelabels
                )
            )

    if remove_pct:
        for artist_text in ax.texts:
            text_ = artist_text.get_text()
            if "%" in text_:
                artist_text.remove()

    if remove_labels:
        for artist_text in ax.texts:
            text_ = artist_text.get_text()
            if text_[1] == "_":
                artist_text.remove()
            elif text_ in gmd.index:
                # artist_text.set_text("; ".join([text_] + gmd.loc[text_, ranks[:-1]].str[3:].to_list()))
                if format_lineage:
                    text_ = format_lineage(text_)
                    artist_text.set_text(text_)
                else:
                    artist_text.set_text(text_)

    if remove_all_labels:
        for artist_text in ax.texts:
            artist_text.remove()





def custom_join(lst, N=3, sep1="\n", sep2="; "):
    # Split the list into sublists of N elements each
    sublists = [lst[i:i + N] for i in range(0, len(lst), N)]
    
    # Join elements within each sublist using sep2
    joined_sublists = [sep2.join(map(str, sublist)) for sublist in sublists]
    
    # Join the sublists using sep1
    result = sep1.join(joined_sublists)
    
    return result





pd.set_option('display.max_rows', None)











provinces_pie_data["16"]





# Define layout
fig = plt.figure(figsize=(24, 16))
gs = mpl.gridspec.GridSpec(4, 5, figure=fig)
bar_width = 1
label_face_colors = {v["label"]: v["color"] for k, v in palettes["k_10"].items()}

# phylum plot
ax_phylum = fig.add_subplot(gs[0, :])

indices_phylum = np.arange(len(stacked_bar_phylum))
bottom_phylum = np.zeros(len(stacked_bar_phylum))

for i, (column, color) in enumerate(colordict_phylum.items()):
    ax_phylum.bar(indices_phylum, stacked_bar_phylum[column], bar_width, bottom=bottom_phylum, label=column, color=color)
    bottom_phylum += stacked_bar_phylum[column]

unique_categories = stacked_bar_phylum["sourmash_k_10_1487_25m"].unique()
category_positions = []
rectangles = []
for category in unique_categories:
    category_indices = stacked_bar_phylum.index[stacked_bar_phylum['sourmash_k_10_1487_25m'] == category].tolist()
    start_pos = indices_phylum[stacked_bar_phylum.index.get_loc(category_indices[0])]
    end_pos = indices_phylum[stacked_bar_phylum.index.get_loc(category_indices[-1])] + bar_width
    midpoint = ((start_pos + end_pos) / 2)
    category_positions.append(midpoint)
    rectangles.append((start_pos - bar_width/2, end_pos - start_pos + bar_width - 1))

for start_pos, width in rectangles:
    ax_phylum.add_patch(plt.Rectangle((start_pos, 0), width, bottom_phylum.max(), fill=False, edgecolor='black', linewidth=2))

ax_phylum.set_title('10 most abundant phyla')
ax_phylum.set_xlabel('')
ax_phylum.set_ylabel('Relative abundance (\\%)')
ax_phylum.set_ylim(0, 100)
ax_phylum.set_xlim(min(indices_phylum) - bar_width/2, max(indices_phylum) + bar_width/2)
ax_phylum.set_xlim(min(indices_phylum) - bar_width/2, 1615)
ax_phylum.spines["top"].set_visible(False)
ax_phylum.spines["bottom"].set_visible(False)
ax_phylum.spines["right"].set_visible(False)
ax_phylum.set_xticks(category_positions)
# Manually add xtick labels with background colors
for pos, category in zip(category_positions, unique_categories):
    ax_phylum.text(pos, -0.05 * bottom_phylum.max(), category, ha='center', va='top', backgroundcolor=label_face_colors[category], fontsize=8)
ax_phylum.legend(title='Phylum', loc="lower right")

# family plot
ax_family = fig.add_subplot(gs[1, :])

indices_family = np.arange(len(stacked_bar_family))
bottom_family = np.zeros(len(stacked_bar_family))

for i, (column, color) in enumerate(colordict_family.items()):
    ax_family.bar(indices_family, stacked_bar_family[column], bar_width, bottom=bottom_family, label=column, color=color)
    bottom_family += stacked_bar_family[column]

unique_categories = stacked_bar_family["sourmash_k_10_1487_25m"].unique()
category_positions = []
rectangles = []
for category in unique_categories:
    category_indices = stacked_bar_family.index[stacked_bar_family['sourmash_k_10_1487_25m'] == category].tolist()
    start_pos = indices_family[stacked_bar_family.index.get_loc(category_indices[0])]
    end_pos = indices_family[stacked_bar_family.index.get_loc(category_indices[-1])] + bar_width
    midpoint = ((start_pos + end_pos) / 2)
    category_positions.append(midpoint)
    rectangles.append((start_pos - bar_width/2, end_pos - start_pos + bar_width - 1))

for start_pos, width in rectangles:
    ax_family.add_patch(plt.Rectangle((start_pos, 0), width, bottom_family.max(), fill=False, edgecolor='black', linewidth=2))

ax_family.set_title('10 most abundant families')
ax_family.set_xlabel('')
ax_family.set_ylabel('Relative abundance (\\%)')
ax_family.set_ylim(0, 100)
ax_family.set_xlim(min(indices_family) - bar_width/2, max(indices_family) + bar_width/2)
ax_family.set_xlim(min(indices_family) - bar_width/2, 1615)
ax_family.spines["top"].set_visible(False)
ax_family.spines["bottom"].set_visible(False)
ax_family.spines["right"].set_visible(False)
ax_family.set_xticks(category_positions)
for pos, category in zip(category_positions, unique_categories):
    ax_family.text(pos, -0.05 * bottom_phylum.max(), category, ha='center', va='top', backgroundcolor=label_face_colors[category], fontsize=8)
ax_family.legend(title='Family', loc="lower right")


pie_chart_2_0 = fig.add_subplot(gs[2, 0])
pie_chart_2_1 = fig.add_subplot(gs[2, 1])
pie_chart_2_2 = fig.add_subplot(gs[2, 2])
pie_chart_2_3 = fig.add_subplot(gs[2, 3])
pie_chart_2_4 = fig.add_subplot(gs[2, 4])
pie_chart_3_0 = fig.add_subplot(gs[3, 0])
pie_chart_3_1 = fig.add_subplot(gs[3, 1])
pie_chart_3_2 = fig.add_subplot(gs[3, 2])
pie_chart_3_3 = fig.add_subplot(gs[3, 3])
pie_chart_3_4 = fig.add_subplot(gs[3, 4])

pie_charts = [pie_chart_2_0, pie_chart_2_1, pie_chart_2_2, pie_chart_2_3, pie_chart_2_4, pie_chart_3_0, pie_chart_3_1, pie_chart_3_2, pie_chart_3_3, pie_chart_3_4]

format_lineage = lambda x: custom_join(gmd.loc[x, ranks[:-1]].str[3:].to_list() + [x])
format_lineage = False
for ax, province in zip(pie_charts, "14 10 2 3 7 0 16 5 11 9".split()):
    color = palettes["k_10"][int(province)]["color"]
    sunburst(provinces_pie_data[province], ax, remove_labels=True, remove_pct=True, pctdistance=0.8, format_lineage=format_lineage, rotatelabels=False, genome_fontsize=6, remove_all_labels=True)
    ax.set_title(palettes["k_10"][int(province)]["label"], bbox=dict(facecolor=color, alpha=0.7))
    ax.set_frame_on(True)
    ax.spines["top"].set_color(color)
    ax.spines["bottom"].set_color(color)
    ax.spines["left"].set_color(color)
    ax.spines["right"].set_color(color)



plt.savefig("/local/path/to/figures/final_draft_imgs/fig3_tax_top_genomes_only.png", dpi=1200, bbox_inches="tight", transparent=True)


# ### Cluster Image Map




genome_labels = gmd.loc[cim.columns, [i for i in ranks if i != "genomes"]].apply(lambda row: "; ".join(list(dict.fromkeys([i[3:] if i[1] == "_" else i for i in row] + [row.name]))), axis=1)





col_colors = smd.loc[cim.index]["sourmash_k_10_1487_25m"].astype(int).map({k: v["color"] for k, v in palettes["k_10"].items()}).rename("Province")





cimplot = sns.clustermap(cim.T, method="average", cmap="Spectral_r", xticklabels=False, col_colors=col_colors, dendrogram_ratio=0.1, cbar_pos=None, figsize=(20, 20))


_ = cimplot.ax_heatmap.set_yticklabels(genome_labels, rotation=0)

# plt.savefig("./final_draft_imgs/fig3_tax_bottom.png", dpi=600, bbox_inches="tight", transparent=True)





pd.set_option('display.max_rows', 100)
ranks.pop()





gmd.loc[tables_clean["genomes"].median().nlargest(100).index, ranks]





gmd.drop_duplicates("genus").set_index("genus").loc[tables_clean["genus"].median().nlargest(20).index, [i for i in ranks if i != "genus"]]





gmd.drop_duplicates("genus").set_index("genus").loc[tables_clean["genus"].median().nlargest(100).index, [i for i in ranks if i != "genus"]]





(tables_clean["genus"] > 0).mean().sort_values(ascending=True).plot(kind="hist")





(tables_clean["genomes"] > 0).mean().sort_values(ascending=True).plot(kind="hist")





tables_clean["genomes"]["GCA_003662515"].describe()





from skbio import diversity











metric = "simpson"
rank = "genus"
s = pd.Series(diversity.alpha_diversity(metric, get_rel_abun(tables_clean[rank]).map(lambda x: 0.0 if x < 1e-6 else x)))
s.index = tables_clean[rank].index
s.name = metric





table = tables_clean[rank].join(s).join(smd["sourmash_k_10_1487_25m"].rename("province").astype(int).map({k: v["label"] for k, v in palettes["k_10"].items()}))





table.groupby("province")[metric].describe()





# Individual sunbursts





ranks





provinces_pie_data[province].query("rank == 'genomes'").set_index("taxa")





# Test one instance
for province in "0 14 16 10 2 3 7 5 11 9".split():
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    print(palettes["k_10"][int(province)]["label"])
    print(provinces_pie_data[province].query("rank == 'genomes'").set_index("taxa").join(gmd[ranks + ["sci_names"]]).set_index("value"))
    sunburst(provinces_pie_data[province],
             ax,
             remove_labels=True,
             remove_pct=False,
             pctdistance=0.8,
             format_lineage=lambda x: custom_join(gmd.loc[x, ranks].str[3:].to_list() + [x]),)
    ax.set_title(palettes["k_10"][int(province)]["label"], bbox=dict(facecolor=palettes["k_10"][int(province)]["color"], alpha=0.7))
    print()





gmd.loc[provinces_pie_data["0"].query("rank == 'genomes'")["taxa"], ranks[:-1] + ["sci_names"]]





fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sunburst(provinces_pie_data["0"],
         ax,
         remove_labels=True,
         remove_pct=False,
         pctdistance=0.8,
         format_lineage=lambda x: custom_join(gmd.loc[x, ranks[:-1]].str[3:].to_list() + [x]))
ax.set_title(palettes["k_10"][int("0")]["label"], bbox=dict(facecolor=palettes["k_10"][int("0")]["color"], alpha=0.7))


# plt.savefig("./sunburst_aplr.svg", format="svg")





get_ipython().system('ls ../drafts')





# Test one instance
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sunburst(provinces_pie_data["0"], ax, remove_labels=True, remove_pct=False, pctdistance=0.8)





# Test one instance
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sunburst(provinces_pie_data["7"], ax, remove_labels=True, remove_pct=False, pctdistance=0.8)





# Test one instance
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sunburst(provinces_pie_data["10"], ax, remove_labels=True, remove_pct=False, pctdistance=0.8)





# Test one instance
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sunburst(provinces_pie_data["5"], ax, remove_labels=True, remove_pct=False, pctdistance=0.8)





# Test one instance
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

sunburst(provinces_pie_data["7"], ax, remove_labels=True, remove_pct=False, pctdistance=0.8, format_lineage=lambda text_: custom_join(gmd.loc[text_, ranks[:-1]].map(lambda x: x[3:]).to_list() + [text_]))





genome_number_labels = provinces_pie_data.copy()

for k, v in genome_number_labels.items():
    genome_number_labels[k] = v[v["rank"] == "genomes"].drop_duplicates().sort_values("taxa").reset_index(drop=True).reset_index().set_index("taxa")




# Test one instance
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# TROP
sunburst(provinces_pie_data["9"], ax, remove_labels=False, remove_pct=True, pctdistance=0.8, format_lineage=False, rotatelabels=True)





# Test one instance
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# PEQD
sunburst(provinces_pie_data["11"], ax, remove_labels=False, remove_pct=True, pctdistance=0.8, format_lineage=False, rotatelabels=True)





# Test one instance
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# TCON
sunburst(provinces_pie_data["5"], ax, remove_labels=False, remove_pct=True, pctdistance=0.8, format_lineage=False, rotatelabels=True)


# ## Genus and genome-level heatmaps

# ### *Synechococcus*









data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["genus"].str.contains("Synechococcus")) |
                                         (gmd["genus"].str.contains("Cyanobium"))].index]).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_[data_.iloc[:, :-1].max().nlargest(10).index]
data_ = data_.rename(columns={**gmd.loc[data_.columns[:-1], "sci_names"].to_dict(), **{"GCA_000012505": "Synechococcus sp. CC9902"}})
sns.clustermap(data_.T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(7.5,5), xticklabels=False, yticklabels=True, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_synechococcus.svg", bbox_inches="tight")


# ### *Prochlorococcus*








data_ = get_rel_abun(tables_clean["genus"][gmd[(gmd["genus"].str.contains("Prochlorococcus"))]["genus"].unique()]).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(7.5,5), xticklabels=False, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_prochlorococcus_genus.svg", bbox_inches="tight")










data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["genus"].str.contains("Prochlorococcus_A"))].index])
data_ = data_[data_.mean().nlargest(5).index]
rename_dict = gmd.loc[data_.columns, "sci_names"].to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_prochlorococcus_A_genome.svg", bbox_inches="tight")





data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["genus"].str.contains("Prochlorococcus"))].index])
data_ = data_[data_.mean().nlargest(6).index]
rename_dict = gmd.loc[data_.columns, "sci_names"].to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_prochlorococcus_genome.svg", bbox_inches="tight")


# ### *Pelagibacter*




data_ = get_rel_abun(tables_clean["genomes"])[gmd[(gmd["family"].str.contains("Pelagibacteraceae"))].index]
data_ = data_[data_.max().nlargest(10).index]
rename_dict = gmd.loc[data_.columns, ["family", "genus", "sci_names"]].apply(lambda row: "; ".join(row), axis=1).to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.3)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_pelagibacter.eps", bbox_inches="tight")


# ### SAR86




data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["family"].str.contains("SAR86"))].index])
data_ = data_[data_.median().nlargest(10).index]
rename_dict = gmd.loc[data_.columns, ["family", "genus", "sci_names"]].apply(lambda row: "; ".join(row.to_list()), axis=1).to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_sar86.svg", bbox_inches="tight")





data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["family"].str.contains("HIMB59"))].index])
# data_ = data_[data_.median().nlargest(10).index]
rename_dict = gmd.loc[data_.columns, ["family", "genus", "sci_names"]].apply(lambda row: "; ".join(row.to_list()), axis=1).to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,7.5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_himb59.svg", bbox_inches="tight")








# ## Genus and genome-level heatmaps (PNG)

# ### *Synechococcus*




data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["genus"].str.contains("Synechococcus")) |
                                         (gmd["genus"].str.contains("Cyanobium"))].index]).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_[data_.iloc[:, :-1].max().nlargest(10).index]
data_ = data_.rename(columns={**gmd.loc[data_.columns[:-1], "sci_names"].to_dict(), **{"GCA_000012505": "Synechococcus sp. CC9902"}})
sns.clustermap(data_.T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(7.5,5), xticklabels=False, yticklabels=True, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_synechococcus.png", dpi=600, bbox_inches="tight")


# ### *Prochlorococcus*






data_ = get_rel_abun(tables_clean["genus"][gmd[(gmd["genus"].str.contains("Prochlorococcus"))]["genus"].unique()]).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(7.5,5), xticklabels=False, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_prochlorococcus_genus.png", dpi=600, bbox_inches="tight")






data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["genus"].str.contains("Prochlorococcus_A"))].index])
data_ = data_[data_.mean().nlargest(5).index]
rename_dict = gmd.loc[data_.columns, "sci_names"].to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_prochlorococcus_A_genome.png", dpi=600, bbox_inches="tight")





data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["genus"].str.contains("Prochlorococcus"))].index])
data_ = data_[data_.mean().nlargest(6).index]
rename_dict = gmd.loc[data_.columns, "sci_names"].to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_prochlorococcus_genome.png", dpi=600, bbox_inches="tight")


# ### *Pelagibacter*




data_ = get_rel_abun(tables_clean["genomes"])[gmd[(gmd["family"].str.contains("Pelagibacteraceae"))].index]
data_ = data_[data_.max().nlargest(10).index]
rename_dict = gmd.loc[data_.columns, ["family", "genus", "sci_names"]].apply(lambda row: "; ".join(row), axis=1).to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.3)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_pelagibacter.png", dpi=600, bbox_inches="tight")


# ### SAR86




data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["family"].str.contains("SAR86"))].index])
data_ = data_[data_.median().nlargest(10).index]
rename_dict = gmd.loc[data_.columns, ["family", "genus", "sci_names"]].apply(lambda row: "; ".join(row.to_list()), axis=1).to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.6, cbar_pos=None)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_sar86.png", dpi=600, bbox_inches="tight")





data_ = get_rel_abun(tables_clean["genomes"][gmd[(gmd["family"].str.contains("HIMB59"))].index])
# data_ = data_[data_.median().nlargest(10).index]
rename_dict = gmd.loc[data_.columns, ["family", "genus", "sci_names"]].apply(lambda row: "; ".join(row.to_list()), axis=1).to_dict()
rename_dict = {k: v + f"; {k}" for k, v in rename_dict.items()}
data_ = data_.rename(columns=rename_dict).join(
                                             smd["sourmash_k_10_1487_25m"].astype(int).map(
                                                 {k: v["label"] for k, v in palettes["k_10"].items()}).rename("province")).rename_axis("Sample")
data_ = data_.sort_values("province")
sns.clustermap(data_.iloc[:, :-1].T, col_colors=col_colors.loc[data_.index], cmap="Spectral_r", figsize=(10,7.5), yticklabels=True, xticklabels=False, vmin=0, vmax=0.6)
plt.savefig("/local/path/to/figures/final_draft_imgs/clustermap_himb59.png", dpi=600, bbox_inches="tight")







