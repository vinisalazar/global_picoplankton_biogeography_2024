#!/usr/bin/env python
# coding: utf-8

"""
Plot model RF classification model performance.
"""



import os
import sys

from glob import glob
from itertools import cycle

import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc

os.environ['PATH'] = os.environ['PATH'] + ':/apps/easybuild-2022/easybuild/software/Compiler/GCC/11.3.0/texlive/20230313/bin/x86_64-linux/'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.usetex'] = True

sys.path.insert(0, "/local/path/to/scripts/")
sys.path.insert(0, "/local/path/to/scripts/")
from plotting_utils import palettes, split_on_capitals






env_cols = "Salinity Nitrate OceanTemperature DissolvedMolecularOxygen Silicate pH DissolvedIron Phosphate SeaIceCover Chlorophyll".split()
# env_data_glob_string = "/local/path/to/data/environmental/*baseline*.nc"

data_dir = "/local/path/to/"
data_dir = "/local/path/to/"

smd = pd.read_csv(f"{data_dir}/provinces_final/data/metadata_1454_cluster_labels.csv", index_col=0)

smd["cluster"] = smd["sourmash_k_10_1487_25m"].map({k: v["label"] for k, v in palettes["k_10"].items()})

test_size = 0.5
random_state = 100
X = smd.drop_duplicates("coords")[env_cols]
y = smd.drop_duplicates("coords")["sourmash_k_10_1487_25m"].astype(str)
y = smd.drop_duplicates("coords")["cluster"]
scaler = StandardScaler()
scale = False
palette = {v["label"]: v["color"] for k, v in palettes["k_10"].items()}


# ## Repeated Stratified *k*-fold with modified RF Classifier




# Assuming you have your data in X and y
# Define the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_samples=0.75)

# Define the cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=9, n_repeats=10, random_state=random_state)


confusion_matrices, accuracies = [], []
feature_importances = []
for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Optionally scale the data
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Fit the classifier
    clf.fit(X_train, y_train)

    # Predict the test set
    y_pred = clf.predict(X_test)
    confusion_matrix_ = confusion_matrix(y_test, y_pred, normalize='pred')
    confusion_matrices.append(confusion_matrix_)

    # Evaluate the performance
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    # Get the feature importances
    feature_importances.append(clf.feature_importances_)
    print(f"Fold {i}: Accuracy on the test set: {accuracy}")





feature_importances = pd.DataFrame(feature_importances)
feature_importances.columns = clf.feature_names_in_





feature_importances_order = feature_importances.mean().sort_values(ascending=False).index





corr = X.corr()
corr = corr.loc[feature_importances_order, feature_importances_order]





fig, axs = plt.subplots(2, 1, figsize=(10, 16))

sns.boxenplot(data=feature_importances[feature_importances_order], palette="Blues_r", orient="h", ax=axs[0])
axs[0].set_xlabel("Feature relative importance")
axs[0].set_title("Feature importance across 100 folds of cross-validation")
sns.heatmap(corr, cmap="coolwarm_r", annot=True, fmt=".2f", ax=axs[1])
axs[1].set_title("Correlation between predictor features")

plt.annotate("A", (0.05, 0.875), xycoords="figure fraction", weight="bold", size=16)
plt.annotate("B", (0.05, 0.475), xycoords="figure fraction", weight="bold", size=16)

# plt.savefig("/local/path/to/figures/final_draft_imgs/fig_S5_feature_importance_correlation.png", dpi=600, bbox_inches="tight")
# plt.savefig("/local/path/to/figures/final_draft_imgs/fig_S5_feature_importance_correlation.svg", bbox_inches="tight")


# ## Averaged confusion matrix




confusion = np.mean(np.stack(confusion_matrices), axis=0)
confusion = pd.DataFrame(confusion, index=clf.classes_, columns=clf.classes_)

confusion_std = np.std(np.stack(confusion_matrices), axis=0)
confusion_std = pd.DataFrame(confusion_std, index=clf.classes_, columns=clf.classes_)





# Function to create the annotation text with an optional diagonal parameter
def create_annot_text(mean_df, std_df, diagonal=True):
    annot = mean_df.round(2).astype(str)
    if diagonal:
        for i in range(len(mean_df)):
            annot.iloc[i, i] += " ± " + str(round(std_df.iloc[i, i], 2))
    else:
        annot += " ± " + std_df.round(2).astype(str)
    return annot

# Create annotation text
annot_text = create_annot_text(confusion, confusion_std)

# Build the plot
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(confusion, annot=annot_text, fmt="", cmap=plt.cm.Blues, linewidths=0.2, ax=ax,
            annot_kws={'size': 10})

# Add labels to the plot
class_names = clf.classes_
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
ax.set_xticks(tick_marks2, class_names, rotation=0)
ax.set_yticks(tick_marks2, class_names, rotation=0)
ax.set_xlabel('Predicted label')
ax.set_ylabel('True label')
# ax.set_title('Confusion matrix for Random Forest model', loc='left')
ax.set_title('Cross-validation strategy: repeated stratified \\textit{k}-fold. 9 splits, 10 repeats.\nRF params: Max._depth=10, Min._samples_split=5, Min._samples_leaf=2, Max._samples=0.7'.replace("_", " "), loc='left')

outfile = f"{data_dir}/provinces_final/figures/final_draft_imgs/figS4_confusion_matrix_repeated_kfold.svg"
# plt.savefig(outfile, dpi="figure" if outfile[-3:] != "png" else 600, bbox_inches="tight")


# ## OvR ROC curve – all provinces




def plot_roc_curve(clf, X_train, y_train, X_test, y_test, classes, ax, binary_clf=False, set_ylabel=True, set_xlabel=True, title='Receiver Operating Characteristic (ROC) - Multiclass'):
    clf.fit(X_train, y_train)
    clf.classes_ = classes

    y_score = clf.predict_proba(X_test) if not binary_clf else clf.predict_proba(X_test)[:, 1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if binary_clf:
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='gray', lw=2, label='ROC curve (area = {0:0.2f})'.format(roc_auc))
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([-0.01, 1.0])
        ax.set_ylim([0.0, 1.05])
        if title:
            ax.set_title(title)
    else:
        for i, class_ in enumerate(classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        for i, class_ in enumerate(classes):
            ax.plot(fpr[i], tpr[i], color=palette[class_], lw=2,
                    label=f'{class_} (area = {round(roc_auc[i], 2)})')

        ax.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4,
                label=f"Micro-average (area = {round(roc_auc['micro'], 2)})")

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([-0.05, 1.0])
        ax.set_ylim([0.0, 1.05])
        if title:
            ax.set_title(title)
    if set_xlabel:
        ax.set_xlabel('False Positive Rate')
    if set_ylabel:
        ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")





# Create a GridSpec with 2 rows and 2 columns
fig = plt.figure(figsize=(32, 20))
gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[2.5, 1], height_ratios=[1, 1, 1], hspace=0.2, wspace=0.1)

# Create the axes using add_subplot from GridSpec
ax1 = fig.add_subplot(gs[:, 0])  # First plot (top-left)
ax2 = fig.add_subplot(gs[0, 1],)  # Second plot (top-right)
ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)  # Third plot (bottom-left)
ax4 = fig.add_subplot(gs[2, 1], sharex=ax2)  # Fourth plot (bottom-right)

# Create annotation text
annot_text = create_annot_text(confusion, confusion_std)

# Build the plot
sns.set(font_scale=1.4)
sns.heatmap(confusion, annot=annot_text, fmt="", cmap=plt.cm.Blues, linewidths=0.2, ax=ax1,
            annot_kws={'size': 15}, cbar=False)

# Add labels to the plot
class_names = clf.classes_
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
ax1.set_xticks(tick_marks2, class_names, rotation=0)
ax1.set_yticks(tick_marks2, class_names, rotation=0)
ax1.set_xlabel('Predicted label')
ax1.set_ylabel('True label')
ax1.set_title('Cross-validation strategy: repeated stratified \\textit{k}-fold. 9 splits, 10 repeats.\nRF params: Max._depth=10, Min._samples_split=5, Min._samples_leaf=2, Max._samples=0.7'.replace("_", " "), loc='left')

X_all = smd.drop_duplicates("coords")[env_cols]
y_all = smd.drop_duplicates("coords")["sourmash_k_10_1487_25m"].astype(str)
y_all = smd.drop_duplicates("coords")["cluster"]

# All classes
classes_all = np.unique(y_all)
y_bin_all = label_binarize(y_all, classes=classes_all)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_bin_all, test_size=test_size, random_state=0, stratify=y
)
clf_ = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state))
plot_roc_curve(clf_, X_train, y_train, X_test, y_test, classes_all, ax2, set_xlabel=False, set_ylabel=False, title='Receiver Operating Characteristic (ROC) curve – all provinces')

# Temperate classes
prov_cat_df_temp = smd[smd["cluster"].isin("CTEM MTEM STEM OTEM".split())]
X_temp = prov_cat_df_temp.drop_duplicates("coords")[env_cols]
y_temp = prov_cat_df_temp.drop_duplicates("coords")["sourmash_k_10_1487_25m"].astype(str)
y_temp = prov_cat_df_temp.drop_duplicates("coords")["cluster"]
classes_temp = np.unique(y_temp)
y_bin_temp = label_binarize(y_temp, classes=classes_temp)
X_train, X_test, y_train, y_test = train_test_split(
    X_temp, y_bin_temp, test_size=test_size, random_state=random_state, stratify=y_temp
)
plot_roc_curve(clf_, X_train, y_train, X_test, y_test, classes_temp, ax3, set_xlabel=False, set_ylabel=False, title='ROC curve – Temperate')

# Tropical classes
prov_cat_df_trop = smd[smd["cluster"].isin("TGYR TRHI TRLO".split())]
X_trop = prov_cat_df_trop.drop_duplicates("coords")[env_cols]
y_trop = prov_cat_df_trop.drop_duplicates("coords")["sourmash_k_10_1487_25m"].astype(str)
y_trop = prov_cat_df_trop.drop_duplicates("coords")["cluster"]
classes_trop = np.unique(y_trop)
y_bin_trop = label_binarize(y_trop, classes=classes_trop)
X_train, X_test, y_train, y_test = train_test_split(
    X_trop, y_bin_trop, test_size=test_size, random_state=random_state, stratify=y_trop
)
plot_roc_curve(clf_, X_train, y_train, X_test, y_test, classes_trop, ax4, title='ROC curve – Tropical')

ax2.tick_params(labelbottom=False)
ax3.tick_params(labelbottom=False)
plt.annotate("\\textbf{A}", (0.05, 0.81), xycoords="figure fraction", weight="bold", size=24)
plt.annotate("\\textbf{B}", (0.62, 0.81), xycoords="figure fraction", weight="bold", size=24)

# plt.savefig("/local/path/to/figures/final_draft_imgs/figS3_confusion_matrix_roc_curve.svg", bbox_inches="tight")
plt.savefig("/local/path/to/figures/final_draft_imgs/figS3_confusion_matrix_roc_curve.png", dpi=600, bbox_inches="tight")





# Assuming you have your data in X (pandas DataFrame) and y
# Define the base classifier
clf_ = RandomForestClassifier(
    n_estimators=100, random_state=random_state, 
    max_depth=10, min_samples_split=5, min_samples_leaf=2, max_samples=0.75
)

# Define the cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=9, n_repeats=10, random_state=random_state)

# Lists to store results
confusion_matrices, accuracies = [], []
class_feature_importances = []

for i, (train_index, test_index) in enumerate(cv.split(X, y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Optionally scale the data
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Fit the OneVsRestClassifier
    onevsr = OneVsRestClassifier(clf_).fit(X_train, y_train)

    # Predict the test set
    y_pred = onevsr.predict(X_test)
    confusion_matrix_ = confusion_matrix(y_test, y_pred, normalize='pred')
    confusion_matrices.append(confusion_matrix_)

    # Evaluate the performance
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
    # Get the correctly ordered class labels
    ordered_class_labels = onevsr.classes_

    # Get feature importances for each class using ordered class labels and feature names
    class_importances = {}
    for class_index, class_label in enumerate(ordered_class_labels):
        # Get feature importances for the class
        importances = onevsr.estimators_[class_index].feature_importances_
        # Map feature importances to feature names
        feature_importances_with_labels = pd.Series(importances, index=X.columns)
        class_importances[class_label] = feature_importances_with_labels
    class_feature_importances.append(class_importances)
    
    print(f"Fold {i}: Accuracy on the test set: {accuracy}")

# After the loop, you can analyze feature importances for each class across folds using correct class labels and feature names





class_feature_importances_dict = {class_: pd.DataFrame(pd.DataFrame(class_feature_importances).loc[:, class_].to_list()) for class_ in y.unique()}





for class_, df in class_feature_importances_dict.items():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    sns.boxenplot(data=df, palette="Blues_r", orient="h", ax=ax)
    ax.set_title(f"Feature importance for class {class_}")





corr





class_feature_importances_dict





feature_importances.columns = [split_on_capitals(i) for i in feature_importances.columns]
feature_importances_order = [split_on_capitals(i) for i in feature_importances_order]
corr.index = [split_on_capitals(i) for i in corr.index]
corr.columns = [split_on_capitals(i) for i in corr.columns]
for df_ in class_feature_importances_dict.values():
    df_.columns = [split_on_capitals(i) for i in df_.columns]





df





fig = plt.figure(figsize=(22.5, 20))

# Create a 2-row gridspec
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.225)

# First row - divide into 2 equal subplots
gs_first_row = gs[0].subgridspec(1, 2, wspace=0.2)  # 1 row, 2 columns
ax1 = fig.add_subplot(gs_first_row[0, 0])  # First row, first half
ax2 = fig.add_subplot(gs_first_row[0, 1])  # First row, second half

# Second gridspec for the second and third rows combined (5 plots each)
gs_second_row = gs[1].subgridspec(2, 5)  # 2 rows, 5 columns

# Second row - 5 equally wide plots
ax3 = fig.add_subplot(gs_second_row[0, 0])
ax4 = fig.add_subplot(gs_second_row[0, 1])
ax5 = fig.add_subplot(gs_second_row[0, 2])
ax6 = fig.add_subplot(gs_second_row[0, 3])
ax7 = fig.add_subplot(gs_second_row[0, 4])

# Third row - 5 equally wide plots
ax8 = fig.add_subplot(gs_second_row[1, 0])
ax9 = fig.add_subplot(gs_second_row[1, 1])
ax10 = fig.add_subplot(gs_second_row[1, 2])
ax11 = fig.add_subplot(gs_second_row[1, 3])
ax12 = fig.add_subplot(gs_second_row[1, 4])

class_axes = [ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

sns.heatmap(corr, cmap="coolwarm_r", annot=True, fmt=".2f", ax=ax1, cbar=False)
ax1.set_title("Correlation between predictor features")
ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=12)  # Adjust fontsize as needed
ax1.xaxis.set_ticklabels(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=10)


sns.boxenplot(data=feature_importances[feature_importances_order], palette="Blues_r", orient="h", ax=ax2)
ax2.set_xlabel("Feature relative importance for classification between all classes")
ax2.set_title("Feature importance across 100 folds of cross-validation")
ax2.yaxis.set_ticklabels(ax2.yaxis.get_ticklabels(), fontsize=10)  # Adjust fontsize as needed

plt.annotate("\\textbf{A}", (0.05, 0.80), xycoords="figure fraction", fontweight="bold", size=16)
plt.annotate("\\textbf{B}", (0.515, 0.80), xycoords="figure fraction", fontweight="bold", size=16)
plt.annotate("\\textbf{C}", (0.05, 0.4), xycoords="figure fraction", fontweight="bold", size=16)

colour_dict = {v["label"]: v["color"] for k, v in palettes["k_10"].items()}
for ax_, (class_, df) in zip(class_axes, class_feature_importances_dict.items()):
    if ax_ not in [ax3, ax8]:
        ax_.yaxis.set_ticklabels([])
    if ax_ is ax10:
        ax_.set_xlabel("Feature relative importance per class (One vs. Rest classifier)")
    sns.boxenplot(data=df[feature_importances_order], orient="h", ax=ax_, color=colour_dict[class_])
    ax_.set_title(class_)

plt.savefig("/local/path/to/figures/final_draft_imgs/figS4_feature_importance_correlation.svg", bbox_inches="tight")
# plt.savefig("/local/path/to/figures/final_draft_imgs/figS4_feature_importance_correlation.png", dpi=600, bbox_inches="tight")







