#!/usr/bin/env python

import argparse
import pandas as pd
from pathlib import Path
from glob import glob


def process_rank(df, rank, metric):
    selected_rows = df.loc[df.index.get_level_values("rank") == rank]
    files = [i for i in selected_rows[metric].values if i is not None]
    if files:
        cat_df = pd.concat([pd.read_csv(file) for file in files])
        return cat_df.groupby(rank).sum().T
    else:
        return None


def main(input_dir, outdir, mode="tax"):
    if mode == "tax":
        ranks = "phylum class order family genus species taxid genome".split()
        ranks = "genome".split()
        ranks = "phylum genus species genome".split()
    elif mode == "func":
        ranks = "BRITE COG_category Description EC GOs KEGG_Reaction KEGG_rclass KEGG_ko KEGG_Pathway max_annot_level PFAMs Preferred_name".split()
        ranks = "BRITE COG_category Description EC KEGG_Reaction KEGG_rclass KEGG_ko KEGG_Pathway max_annot_level Preferred_name".split()
        ranks = "BRITE KEGG_Reaction KEGG_rclass KEGG_ko KEGG_Pathway COG_category".split()
    else:
        raise ValueError(
            f"Mode '{mode}' not recognized. Must be either 'tax' or 'func'."
        )
    metrics = "Mean Trimmed_Mean Variance Read_Count Reads_per_base RPKM TPM".split()
    metrics = "Trimmed_Mean".split()
    metrics = [i.lower() for i in metrics]

    input_dir = Path(input_dir)
    samples = glob(str(input_dir.joinpath("*.csv")))
    samples = {Path(i).stem.split("_gb")[0] for i in samples}
    samples = {i: dict() for i in samples}
    for k, v in samples.items():
        for rank in ranks:
            samples[k][rank] = dict()
            for metric in metrics:
                file = (input_dir / "_".join([k, "gb", rank, metric])).with_suffix(
                    ".csv"
                )
                if Path(file).exists():
                    samples[k][rank][metric] = file
                else:
                    samples[k][rank][metric] = None

    table_df = pd.DataFrame.from_dict(
        {(i, j): samples[i][j] for i in samples.keys() for j in samples[i].keys()},
        orient="index",
    )

    table_df.index.rename(["sample", "rank"], inplace=True)
    for rank in ranks:
        for metric in metrics:
            outfile = f"{outdir}/{rank}_{metric}.csv"
            if Path(outfile).exists() and not args.force:
                print(f"File '{outfile}' already exists. Skipping.")
                continue
            else:
                print(f"Processing rank '{rank}', metric '{metric}'.")
                if (rank_df := process_rank(table_df, rank, metric)) is not None:
                    rank_df.to_csv(outfile)
                else:
                    print(f"No files found for rank '{rank}', metric '{metric}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data files.")
    parser.add_argument("-i", "--input_dir", help="Directory containing data files.")
    parser.add_argument("-o", "--outdir", help="Output directory.")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Whether to force overwriting existing files.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        default="tax",
        help="Whether to run in taxonomy or functional mode.",
    )
    args = parser.parse_args()

    main(args.input_dir, args.outdir, args.mode)
