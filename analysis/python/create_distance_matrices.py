#!/usr/bin/env python

"""
Generate distance matrices from all blocks of data.
"""

import sys
import argparse
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from functools import reduce
from itertools import combinations
from pathlib import Path

sys.path.insert(0, "/local/path/to/scripts/")
from utils import print_with_timestamp, cog_mappings

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    print_with_timestamp("Creating merged df. Reading input data.")
    tables = {
        "genomes": pd.read_csv(
            input_dir
            + "data/counts/genomes_trimmed_mean_formatted_clean_normalised.csv",
            index_col=0,
        ),
        "sourmash": pd.read_csv(input_dir + "data/distances/sourmash.csv", index_col=0),
        "env": pd.read_csv(
            input_dir + "data/R/env_data_clean_normalised.csv", index_col=0
        ).drop_duplicates(),
        "KEGG_ko": pd.read_csv(
            input_dir
            + "data/counts/KEGG_ko_trimmed_mean_formatted_clean_normalised.csv",
            index_col=0,
        ),
        "KEGG_Pathway": pd.read_csv(
            input_dir
            + "data/counts/KEGG_Pathway_trimmed_mean_formatted_clean_normalised.csv",
            index_col=0,
        ),
        "KEGG_rclass": pd.read_csv(
            input_dir
            + "data/counts/KEGG_rclass_trimmed_mean_formatted_clean_normalised.csv",
            index_col=0,
        ),
        "KEGG_Reaction": pd.read_csv(
            input_dir
            + "data/counts/KEGG_Reaction_trimmed_mean_formatted_clean_normalised.csv",
            index_col=0,
        ),
        "searoute": pd.read_csv(
            input_dir + "data/distances/searoute_norm.csv", index_col=0
        ),
        "BRITE": pd.read_csv(
            input_dir + "data/counts/BRITE_trimmed_mean_formatted_clean_normalised.csv",
            index_col=0,
        ),
    }
    selectVars_files = {
        "genomes": pd.read_csv(
            args.input_dir + "data/R/genomes_sPLS_selected_variables.csv", index_col=0
        ).columns.to_list(),
        "KEGG_ko": pd.read_csv(
            args.input_dir + "data/R/KEGG_ko_sPLS_selected_variables.csv", index_col=0
        ).columns.to_list(),
        "KEGG_Pathway": pd.read_csv(
            args.input_dir + "data/R/KEGG_Pathway_sPLS_selected_variables.csv",
            index_col=0,
        ).columns.to_list(),
        "KEGG_rclass": pd.read_csv(
            args.input_dir + "data/R/KEGG_rclass_sPLS_selected_variables.csv",
            index_col=0,
        ).columns.to_list(),
        "KEGG_Reaction": pd.read_csv(
            args.input_dir + "data/R/KEGG_Reaction_sPLS_selected_variables.csv",
            index_col=0,
        ).columns.to_list(),
    }

    print_with_timestamp("Calculating distance matrices.")
    n = len(tables["genomes"])
    for k, v in tables.items():
        print_with_timestamp(f"\tFor matrix '{k}'.")
        if k in ("sourmash", "searoute"):
            tables[k] = (
                tables[k]
                .loc[tables["genomes"].index, tables["genomes"].index]
                .drop_duplicates()
            )
        else:
            counts_data = tables[k].copy()
            tables[k] = pd.DataFrame(
                squareform(pdist(counts_data)),
                index=tables[k].index,
                columns=tables[k].index,
            )
            tables[k].to_csv(input_dir + output_dir + f"{k}_{n}.csv")
            if k in selectVars_files.keys():
                formatted_selectVar_names = [
                    i[:-2] if (i.endswith(".1")) or (i.endswith(".2")) else i
                    for i in selectVars_files[k]
                ]
                try:
                    X = counts_data[formatted_selectVar_names].copy()
                except KeyError:
                    breakpoint()
                selectVars_dist = pd.DataFrame(
                    squareform(pdist(X)), index=X.index, columns=X.index
                )
                selectVars_dist.to_csv(
                    input_dir + output_dir + f"{k}_{n}_selectVars.csv"
                )
            tables[k] = tables[k] / tables[k].max().max()
            tables[k].to_csv(input_dir + output_dir + f"{k}_{n}_normalised.csv")
            if k in selectVars_files.keys():
                selectVars_dist = selectVars_dist / selectVars_dist.max().max()
                selectVars_dist.to_csv(
                    input_dir + output_dir + f"{k}_{n}_normalised_selectVars.csv"
                )

    print_with_timestamp("Merging distance matrices.")
    for k, v in tables.items():
        tables[k] = (
            v.reset_index(names="query")
            .melt(id_vars="query", var_name="subject", value_name=k)
            .set_index(["query", "subject"])
        )

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True),
        tables.values(),
    )

    # Filter same-sample comparisons
    merged_df = merged_df[merged_df.sum(axis=1) > 0]
    outfile = input_dir + output_dir + f"merged_distances_{n}_filtered_zeros.csv"
    merged_df.to_csv(outfile)
    print_with_timestamp(f"Done. Saved to '{outfile}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create distance matrices.")
    parser.add_argument(
        "-i", "--input_dir", help="Input (parent) directory.", default="/local/path/to/"
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output directory, relative to parent.",
        default="data/distances/",
    )
    args = parser.parse_args()
    main(args)
