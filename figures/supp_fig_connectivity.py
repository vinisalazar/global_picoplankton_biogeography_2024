#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np

import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter, LatitudeLocator

from itertools import combinations

from matplotlib import pyplot as plt

sys.path.insert(0, "../../../scripts/")

from plotting_utils import plot_colored_markers, palettes

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.usetex"] = True
mpl.rc('text.latex', preamble=r'\usepackage{cmbright}')


# In[2]:


data_dir = "/local/path/to/provinces_final"

md = pd.read_csv(f"{data_dir}/data/metadata_1454_cluster_labels.csv", index_col=0)
df = pd.read_csv(f"{data_dir}/../data/misc/sourmash.csv", index_col=0)

df = df.loc[md.index, md.index]


# In[3]:


distances = df.copy()
distances.values[np.triu_indices(len(df))] = np.nan
distances = distances.reset_index(names=["sample1"]).melt(id_vars="sample1", var_name="sample2", value_name="distance")
distances = distances[distances["distance"] == distances["distance"]].reset_index(drop=True)
distances["alpha"] = 1 - distances["distance"]


# # Connectivity

# In[4]:


distances["alpha"].plot(kind="hist", bins=100)
plt.xlim(0.0, 0.1)


# In[5]:


def qualitative_colormap(cmap_name, N):
    cmap = mpl.colormaps.get_cmap(cmap_name)
    colors = [cmap(i) for i in np.linspace(0, 1, N)]
    return colors


# In[9]:


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_global()
gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    color="black",
    alpha=0.7,
    linestyle="--",
    ylabel_style={"size": 12, "color": "black", "weight": "bold"},
    xlabel_style={"size": 12, "color": "black", "weight": "bold"},
    auto_inline=True,
    ylocs = [-60, -40, -23.5, 0, 23.5, 40, 60],
    zorder=1,
)
gl.xlines = False
gl.top_labels = False
gl.bottom_labels = True
gl.left_labels = True
gl.right_labels = False
gl.xformatter = LongitudeFormatter()
gl.yformatter = LatitudeFormatter()
ax.add_feature(cfeature.NaturalEarthFeature("physical", "land", "10m"),
               alpha=1, color="#EFEFDB", zorder=2, edgecolor="black", linewidth=0.5)
ax.add_feature(cfeature.LAKES, ec="black", lw=0.25, zorder=2)
ax.add_feature(cfeature.RIVERS, zorder=2)
ax.coastlines(resolution="10m", lw=0.5, edgecolor="black", zorder=2)

for ix, row in md.iterrows():
    ax.plot(row["longitude"], row["latitude"], color=palettes["k_10"][row["sourmash_k_10_1487_25m"]]["color"], marker=palettes["k_10"][row["sourmash_k_10_1487_25m"]]["marker"],markersize=10, markeredgewidth=0.5, markeredgecolor="black", zorder=2)

plot_data = distances[distances["alpha"] >= 0.07].reset_index(drop=True)

quantile_values = [0.5, 0.75, 0.9]
quantiles = plot_data["alpha"].quantile(quantile_values)

colors = qualitative_colormap("Greys", 4)

for ix, row in plot_data.iterrows():
    color = "black"
    if ix % 1000 == 0:
       print(f"Processed {ix} / {len(plot_data)} edges.", end="\r")
    if row["alpha"] <= quantiles.loc[quantile_values[0]]:
        alpha = row["alpha"]
        color = colors[1]
    elif row["alpha"] <= quantiles.loc[quantile_values[1]]:
        alpha = row["alpha"] + 0.1
        color = colors[2]
    elif row["alpha"] <= quantiles.loc[quantile_values[2]]:
        alpha = row["alpha"] + 0.25
        color = colors[3]
    else:
        alpha = 1
        color = colors[3]
        color = "blue"
    ax.plot([md.loc[row["sample1"], "longitude"], md.loc[row["sample2"], "longitude"]], [md.loc[row["sample1"], "latitude"], md.loc[row["sample2"], "latitude"]], color=color, alpha=alpha, lw=0.5, zorder=3)

ax.set_title("Genomic connectivity between samples", fontsize=20)

plt.savefig("/local/path/to/figures/final_draft_imgs/figS7_connectivity.svg", bbox_inches="tight")


# In[ ]:




