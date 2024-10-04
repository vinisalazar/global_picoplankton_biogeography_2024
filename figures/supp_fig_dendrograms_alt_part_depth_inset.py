#!/usr/bin/env python
# coding: utf-8

# # Clustering analysis
# 
# 1. Load distance data
# 2. Keep all samples
#     - Remove outliers (`k=8, N=10`)
#     - Dendrogram
#     - Large partition (`k=4`)
#         - Map
#     - Analyse tropical subtree (`k=6`, `N=10`)
#         - Fine-grained partition map (`k=7`)
#     - Depth graph
# 3. Filter `<25m depth` samples, remove outliers
#     - Dendrogram
# 4. Large partitioning `k=4`
#     - Map
# 5. Fine-grained partioning `k=10`
#     - Map
# 6. Export cluster tables

# In[1]:


import os
import sys
import itertools


import pandas as pd
import numpy as np
import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs

import matplotlib as mpl
from matplotlib import pyplot as plt

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, set_link_color_palette
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sys.path.insert(0, "/local/path/to/scripts/")
from plotting_utils import plot_colored_markers, palettes

os.environ['PATH'] = os.environ['PATH'] + ':/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/texlive/20230313/bin/x86_64-linux/'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.usetex'] = True
get_cmap_ix = lambda ix, cmap: mpl.colors.rgb2hex(plt.get_cmap(cmap).colors[ix])


# # 1. Load distance data

# In[ ]:


df = pd.read_csv("/local/path/to/data/distances/sourmash.csv", index_col=0)
md = pd.read_csv("~/biogo-hub/provinces_final/data/metadata_2132.csv", index_col=0)

df = df.loc[md.index, md.index]


# # 2. Keep all samples
# ## Remove outliers

# In[ ]:


X = squareform(df.values)
Z = linkage(X, method='average')
print_clusters = False

# Use this to determine k
if print_clusters:
    for k in range(5, 11):
        K = cut_tree(Z, n_clusters=k)
        labels = pd.DataFrame(K, index=df.index)[0]
        print(f"k = {k}")
        k_gt_ten = labels.value_counts()[labels.value_counts() > 10].index.__len__()
        print(f"Groups larger than 10: {k_gt_ten}.")
        print(labels.value_counts())
        print()
else:
    k_gt_ten = 4


# In[ ]:


k, N = 8, 10
K = cut_tree(Z, n_clusters=k)
cut_value = Z[-(k - 1), 2]
labels = pd.DataFrame(K, index=df.index)[0]
label_name = f"sourmash_k_{k_gt_ten}_{len(labels)}"
md[label_name] = labels
outlier_provs = md[label_name].value_counts()[md[label_name].value_counts() < N].index
outlier_samples = md[md[label_name].isin(outlier_provs)]
md.loc[outlier_samples.index, label_name] = 99
robust_provs = md[label_name].value_counts()[md[label_name].value_counts() >= N].index


# ## Dendrogram

# In[ ]:


def plot_dendrogram_all_samples(label_name, save=False):
    fig, ax = plt.subplots(figsize=(20, 240))

    D = dendrogram(Z,
                color_threshold=cut_value,
                labels= "B" + md[label_name].astype(str) + "_" + md.index,
                orientation='left',
                leaf_font_size=8,
                ax=ax)
    _ = ax.set_xlim(1, 0.75)

    if save:
        plt.savefig(save, dpi = 400 if save.endswith(".png") else "figure", bbox_inches='tight')

# Supp fig. pt 1
# plot_dendrogram_all_samples(label_name)
# plt.savefig("/local/path/to/figures/img/final_drafts/supp_fig_full_dendrogram_w_subsurface.pdf", bbox_inches='tight')


# In[ ]:


subtree = 3
md_subtree = md[md[label_name] == subtree]
subtree = md_subtree.index

X = df.loc[subtree, subtree]
X = squareform(X.values)
Z = linkage(X, method='average')

if print_clusters:
    for k in range(5, 21):
        K = cut_tree(Z, n_clusters=k)
        labels = pd.DataFrame(K, index=subtree)[0]
        print(f"k = {k}")
        k_gt_ten = labels.value_counts()[labels.value_counts() > N].index.__len__()
        print(f"Groups larger than 10: {k_gt_ten}.")
        print(labels.value_counts())
    print()
else:
    k_gt_ten = 3

k = 6
K = cut_tree(Z, n_clusters=k)
cut_value = Z[-(k - 1), 2]
labels = pd.DataFrame(K, index=subtree)[0]

# Identify labels from subtree
labels = labels + 20
md_subtree.loc[labels.index, label_name] = labels
outlier_provs = md_subtree[label_name].value_counts()[md_subtree[label_name].value_counts() < N].index
outlier_samples = md_subtree[md_subtree[label_name].isin(outlier_provs)]
md_subtree.loc[outlier_samples.index, label_name] = 99

# Join subtree with others
md.loc[labels.index, label_name] = labels
robust_provs = md[label_name].value_counts()[md[label_name].value_counts() >= N].index


# ### Map

# In[ ]:


# Subtree only
# plot_colored_markers(md_subtree[md_subtree[label_name].isin(robust_provs)], color_category=label_name, jitter=1, cmap="Dark2")

md_robust = md[md[label_name].isin(robust_provs)]

# All samples
# plot_colored_markers(md_robust, color_category=label_name, jitter=1, cmap="Dark2")

md_depths_k_7 = md_robust.copy()


# ### Depth plots

# In[ ]:


plot_data = md_depths_k_7.copy()


# In[ ]:


plot_data[plot_data["depth"] > 0]["depth"].describe()


# In[ ]:


boxenplot_colors = dict(zip(plot_data[label_name].astype(int).unique(), sns.color_palette("Dark2").as_hex()))


# In[ ]:





# In[ ]:


plot_data.groupby("sourmash_k_4_2132")["depth"].describe().drop("count", axis=1)


# In[ ]:


sns.boxenplot(data = plot_data, x=label_name, y="depth", showfliers=False, color="gray", width=0.5, palette=boxenplot_colors, hue=label_name, legend=False)

depth_inset_data = plot_data


# ## Statistical tests between samples

# In[ ]:


plot_data["depth"]


# In[ ]:


# Shapiro-Wilk test for each group

# TRY TRANSFORMATIONS TO MAKE IT NORMAL, E.G. BOX-COX

# plot_data["depth"] = plot_data["depth"].apply(lambda x: np.log(x + 1))
normality_results = plot_data.groupby("sourmash_k_4_2132")['depth'].apply(lambda x: stats.shapiro(x))
print("Shapiro-Wilk p-values for normality:\n", normality_results)

# Levene's test
levene_stat, levene_p = stats.levene(*[group["depth"].values for name, group in plot_data.groupby("sourmash_k_4_2132")])
print(f"Levene's test statistic: {levene_stat}")
print(f"Levene's test p-value: {levene_p}")

if all(normality_results.apply(lambda x: x[1]) > 0.05) and levene_p > 0.05:
    # ANOVA
    anova_stat, anova_p = stats.f_oneway(*[group["depth"].values for name, group in plot_data.groupby("sourmash_k_4_2132")])
    print(f"ANOVA p-value: {anova_p}")
else:
    # Kruskal-Wallis test
    kruskal_stat, kruskal_p = stats.kruskal(*[group["depth"].values for name, group in plot_data.groupby("sourmash_k_4_2132")])
    print(f"Kruskal-Wallis statistic: {kruskal_stat}")
    print(f"Kruskal-Wallis p-value: {kruskal_p}")


# In[ ]:


# Mann-Whitney U tests with Bonferroni correction
groups = plot_data['sourmash_k_4_2132'].unique()
pairwise_results = {}
for group1, group2 in itertools.combinations(groups, 2):
    group1_values = plot_data[plot_data['sourmash_k_4_2132'] == group1]['depth']
    group2_values = plot_data[plot_data['sourmash_k_4_2132'] == group2]['depth']
    u_stat, p_val = stats.mannwhitneyu(group1_values, group2_values)
    pairwise_results[(group1, group2)] = p_val * len(groups)  # Bonferroni correction
print("Pairwise Mann-Whitney U test results (Bonferroni corrected p-values):\n")

for k, v in pairwise_results.items():
    if v < 0.01:
        print(f"{k[0]} vs {k[1]}: {v}")


# In[ ]:


# Tukey's HSD post-hoc test
tukey = pairwise_tukeyhsd(endog=plot_data['depth'], groups=plot_data['sourmash_k_4_2132'], alpha=0.05)
print(tukey)


# In[ ]:


plot_data.drop_duplicates("coords").groupby("sourmash_k_4_2132").size()


# In[ ]:


sns.boxenplot(data = plot_data, x="sourmash_k_4_2132", y="depth", showfliers=False, color="gray", width=0.5, palette=boxenplot_colors, hue="sourmash_k_4_2132", legend=False)


# In[ ]:


# for value in plot_data["sourmash_k_4_2132"].unique():
#     data = plot_data[plot_data["sourmash_k_4_2132"] == value]["depth"]
#     fig, ax = plt.subplots()
#     sns.histplot(data, ax=ax)
#     ax.set_title(f"Group {value}")


# # 2. Filter samples to 25m depth

# In[ ]:


md = md[md["depth"] < 25].drop(label_name, axis=1)
df = df.loc[md.index, md.index]


# ## Remove outliers

# In[ ]:


X = squareform(df.values)
Z = linkage(X, method='average')
print_clusters = False

# Use this to determine k
if print_clusters:
    for k in range(5, 11):
        K = cut_tree(Z, n_clusters=k)
        labels = pd.DataFrame(K, index=df.index)[0]
        print(f"k = {k}")
        k_gt_ten = labels.value_counts()[labels.value_counts() > 10].index.__len__()
        print(f"Groups larger than 10: {k_gt_ten}.")
        print(labels.value_counts())
        print()
else:
    k_gt_ten = 4


# In[ ]:


k, N = 6, 10
K = cut_tree(Z, n_clusters=k)
cut_value = Z[-(k - 1), 2]
labels = pd.DataFrame(K, index=df.index)[0]
label_name = f"sourmash_k_{k_gt_ten + 2}_{len(labels)}_25m"
md[label_name] = labels
outlier_provs = md[label_name].value_counts()[md[label_name].value_counts() < N].index
outlier_samples = md[md[label_name].isin(outlier_provs)]
md.loc[outlier_samples.index, label_name] = 99
robust_provs = md[label_name].value_counts()[md[label_name].value_counts() >= N].index
robust_samples = md[md[label_name].isin(robust_provs)].index


# ## Dendrogram

# In[ ]:


# Supp fig pt 2
# plot_dendrogram_all_samples(label_name)
# plt.savefig("/local/path/to/figures/img/final_drafts/supp_fig_full_dendrogram_without_subsurface.pdf")


# ## *k*=4 partition map

# In[ ]:


md_k_4 = md[md[label_name].isin(robust_provs)]
plot_colored_markers(md_k_4, color_category=label_name, jitter=1, cmap="Dark2")


# ### Deprecated

# In[ ]:


# ds = xr.open_dataset("/data/scratch/projects/punim1293/vini/data/bio-oracle/ph_baseline_2000_2018_depthsurf_xr.nc")
# ds.to_dataframe()
# ds["ph_max"].plot()


# In[ ]:


# df = pd.read_csv("~/biogo-hub/data/models/model_data/sourmash_k_10_1487_25m_not_scaled_16270117_points_X_test.csv")
# ds = df.set_index(["time", "latitude", "longitude"]).to_xarray()
# ds["Salinity"].plot()


# In[ ]:


# ds.dropna(dim="latitude", how="all").dropna(dim="longitude", how="all")
# ds['province_obj'] = ds['province'].astype(str)
# ds["province"].plot(levels=8, subplot_kws=dict(projection=ccrs.Orthographic()), transform=ccrs.PlateCarree())


# In[ ]:


def create_sequential_colormap(color_hex):
    # Convert color hex to RGB
    r, g, b = tuple(int(color_hex[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    # Define colormap dictionary
    colormap_dict = {'red':   ((0.0, r, r),
                               (1.0, 1.0, 1.0)),
                     'green': ((0.0, g, g),
                               (1.0, 1.0, 1.0)),
                     'blue':  ((0.0, b, b),
                               (1.0, 1.0, 1.0))}

    # Create colormap
    colormap = mpl.colors.LinearSegmentedColormap('sequential_colormap', colormap_dict)

    return colormap


# In[ ]:


palette = {
        16:{"description": "Baltic Sea", "label": "BALT", "category": "BALT", "counts": 51, "color": get_cmap_ix(10, "tab20")},
        11:{"description": "Pacific Equatorial Divergence/Countercurrent", "label": "PEQD", "category": "TROP", "counts": 54, "color": get_cmap_ix(8, "tab20") },
        5:{"description": "Tropical Convergence", "label": "TGYR", "category": "TROP", "counts": 161, "color": get_cmap_ix(1, "tab20") },
        9:{"description": "Broad Tropical", "label": "TROP", "category": "TROP", "counts": 818, "color": get_cmap_ix(0, "tab20") },
        0:{"description": "Antarctic Polar", "label": "APLR", "category": "POLR", "counts": 30, "color": get_cmap_ix(15, "tab20") },
        14:{"description": "Arctic Polar", "label": "BPLR", "category": "POLR", "counts": 42, "color": get_cmap_ix(14, "tab20") },
        10:{"description": "Upwelling Areas", "label": "CTEM", "category": "TEMP", "counts": 139, "color": get_cmap_ix(4, "tab20") },
        2:{"description": "S. Subtropical Convergence", "label": "SANT", "category": "TEMP", "counts": 43, "color": get_cmap_ix(6, "tab20") },
        3:{"description": "North Atlantic Drift/Agulhas", "label": "NADR", "category": "TEMP", "counts": 34, "color": get_cmap_ix(7, "tab20") },
        7:{"description": "Mediterranean", "label": "MEDI", "category": "TEMP", "counts": 82, "color": get_cmap_ix(6, "tab20") }
    }


# In[ ]:


# for p in "0 5 7 9 10 11 14 16".split():
#     fig, ax = plt.subplots(figsize=(40, 40))
#     ds[p].plot(cmap=create_sequential_colormap(palette[int(p)]["color"][1:]).reversed(), add_colorbar=False)
#     ax.set_facecolor("k")


# In[ ]:


# fig, ax = plt.subplots(figsize=(40, 40))
# ax.set_facecolor("k")

# for p in "0 5 7 9 10 11 14 16".split():
#     ds[p].plot(cmap=create_sequential_colormap(palette[int(p)]["color"][1:]).reversed(), add_colorbar=False, ax=ax)


# In[ ]:




