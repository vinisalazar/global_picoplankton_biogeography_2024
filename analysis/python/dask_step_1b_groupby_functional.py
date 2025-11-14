#!/usr/bin/env python

"""
Dask code for mass processing of counts tables (functional).
"""

import argparse
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask import dataframe as dd

import pandas as pd

from glob import glob
from tqdm import tqdm
from pathlib import Path


def start_cluster(cores, memory, jobs, _print=True):
    """
    Start a dask cluster on the HPC.

    Parameters
    ----------
        cores : int
            The number of cores to use per job.
        memory : str
            The amount of memory to use per job.
        jobs : int
            The number of jobs to start.

    Returns
    -------
        dask.distributed.Client
            The dask client connected to the cluster.
    """
    slurm_log_output = (
        "/data/gpfs/projects/punim1989/hub_provinces/notebooks/dask-logs/%x_%A_%a.out"
    )
    cluster = SLURMCluster(
        cores=cores,
        memory=memory,
        queue="cascade,mig",
        account="punim0613",
        walltime="8:00:00",
        job_extra_directives=[
            f"--output {slurm_log_output}",
        ],
    )

    if _print:
        print(
            f"Starting cluster with {jobs} jobs, {cores} cores, and {memory} memory per job."
        )
    cluster.scale(jobs=jobs)
    client = Client(cluster)
    print(client.dashboard_link)
    return client


get_sample_from_file = lambda file: file.split("/")[-1].split("_T")[0]


# @dask.delayed
def load(file):
    ddf = dd.read_csv(
        file,
        sep="\t",
        dtype={
            "Mean": "float64",
            "RPKM": "float64",
            "Reads per base": "float64",
            "TPM": "float64",
            "Trimmed Mean": "float64",
            "Variance": "float64",
        },
    )
    ddf = ddf.set_index("Contig")
    return ddf


# @dask.delayed
def merge(left, right):
    return left.merge(right, left_index=True, right_index=True, how="inner")


# @dask.delayed
def groupby(df, by, metric):
    return df.groupby(by)[metric].sum()


def comma_explode(df, column):
    """Explode a column with comma-separated values into multiple rows."""
    df[column] = df[column].str.split(",")
    return df.explode(column).groupby(column).sum()


def str_split_explode(df, column):
    df[column] = df[column].apply(lambda s: list(s))
    return df.explode(column).groupby(column).sum()


def f(filenames, fun, outdir, force=False):
    results = dict()
    outdir = Path(outdir)
    metrics = [
        #    "Mean",
        "Trimmed Mean",
        #    "Variance",
        #    "Read Count",
        #    "Reads per base",
        #    "RPKM",
        #    "TPM",
    ]
    comma_columns = [
        #    "GOs",
        "KEGG_Pathway",
        "KEGG_Reaction",
        "BRITE",
        #    "PFAMs",
        #    "EC",
        "KEGG_ko",
        "KEGG_rclass",
    ]
    single_value_cols = [
        #    "max_annot_lvl",
        # "Preferred_name",
        # "EC",
        "KEGG_ko",
        # "Description",
    ]
    str_split_cols = ["COG_category"]
    all_categories = comma_columns + single_value_cols + str_split_cols
    for file in tqdm(filenames):
        sample = Path(file).stem
        if (
            all(
                outdir.joinpath(
                    f"{sample}_gb_{category}_{metric.lower()}.csv".replace(" ", "_")
                ).exists()
                for category in all_categories
                for metric in metrics
            )
            and not force
        ):
            print(f"Found all files for {sample}. Skipping.", end="\r")
            continue
        data = load(file)
        large_join = merge(data, fun)
        data = large_join.persist()
        dask.distributed.wait(data)
        for category in all_categories:
            for metric in metrics:
                outfile = outdir.joinpath(
                    f"{sample}_gb_{category}_{metric.lower()}.csv".replace(" ", "_")
                )
                if not outfile.exists() or force:
                    gb = groupby(data, category, metric)
                    gb = gb.compute()
                    if category in comma_columns:
                        gb = comma_explode(gb.reset_index(), category)[metric]
                    elif category in str_split_cols:
                        gb = str_split_explode(gb.reset_index(), category)[metric]
                    elif category in single_value_cols:
                        pass
                    try:
                        gb = gb[(gb.index.notnull()) & (gb > 0) & (gb.index != "-")]
                    except ValueError:
                        breakpoint()
                    gb.name = sample
                    results[sample] = gb
                    results[sample].to_csv(outfile)
                else:
                    print(f"{outfile} already exists. Skipping.", end="\r")
    return results


def main(args):
    print("Loading functions file from disk.")
    fun = dd.read_csv(args.functions, sep="\t")
    fun = fun.set_index("#query")
    pattern = args.input
    filenames = glob(pattern)
    print(f"Found {len(filenames)} files.")
    print("Starting computation.")
    results = dask.compute(f(filenames, fun, args.outdir, args.force))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask groupby")
    parser.add_argument(
        "-c", "--cores", type=int, default=8, help="Number of cores per job."
    )
    parser.add_argument(
        "-m", "--memory", type=str, default="16GB", help="Amount of memory per job."
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=64, help="Number of jobs to start."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="/data/gpfs/projects/punim0639/vini/coverm/*.coverm",
        help="Pattern to input files.",
    )
    parser.add_argument(
        "-fn",
        "--functions",
        type=str,
        default="/data/gpfs/projects/punim1989/databases/genes/genome_reps_emapper.emapper.annotations_fmt_filtered",
        help="Path to taxonomy file.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="/local/path/to/",
        help="Path to output directory.",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite of existing files."
    )
    args = parser.parse_args()
    with start_cluster(args.cores, args.memory, args.jobs) as client:
        main(args)
