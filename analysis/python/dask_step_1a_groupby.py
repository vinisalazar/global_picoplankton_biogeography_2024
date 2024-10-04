#!/usr/bin/env python
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
    slurm_log_output = "/data/gpfs/projects/punim1989/biogo-hub/data/slurm-logs/%x_%A_%a.out"
    cluster = SLURMCluster(cores=cores, 
                           memory=memory,
                           queue="cascade,mig",
                           account="punim0613",
                           walltime="8:00:00", 
                           job_extra_directives=[f"--output {slurm_log_output}",],)

    if _print:
        print(f"Starting cluster with {jobs} jobs, {cores} cores, and {memory} memory per job.")
    cluster.scale(jobs=jobs)
    client = Client(cluster)
    print(client)
    print(cluster)
    return client


get_sample_from_file = lambda file: file.split("/")[-1].split("_T")[0]


#@dask.delayed
def load(file):
    ddf = dd.read_csv(file, sep="\t", dtype={'Mean': 'float64',
                                             'RPKM': 'float64',
                                             'Reads per base': 'float64',
                                             'TPM': 'float64',
                                             'Trimmed Mean': 'float64',
                                             'Variance': 'float64'})
    ddf = ddf.set_index("Contig")
    return ddf

#@dask.delayed
def merge(left, right):
    return left.merge(right, left_index=True, right_index=True, how="inner")

#@dask.delayed
def groupby(df, by, metric):
    return df.groupby(by)[metric].sum()

#@dask.delayed
def persist(df):
    return df.persist()


def f(filenames, tax, outdir, force=False):
    results = dict()
    outdir = Path(outdir)
    metrics = ["Trimmed Mean", "Read Count", "TPM"]
    groupby_categories = ["phylum", "class", "order", "family", "genus", "species", "taxid", "genome"]
    for file in tqdm(filenames):
        sample = Path(file).stem
        if all(outdir.joinpath(f"{sample}_gb_{category}_{metric.lower()}.csv".replace(" ", "_")).exists() for category in groupby_categories for metric in metrics) and not force:
            print(f"Found all files for {sample}. Skipping.")
            continue
        data = load(file)
        large_join = merge(data, tax)
        data = large_join.persist()
        dask.distributed.wait(data)
        for category in groupby_categories:
            for metric in metrics:
                outfile = outdir.joinpath(f"{sample}_gb_{category}_{metric.lower()}.csv".replace(" ", "_"))
                if not outfile.exists() or force:
                    gb = groupby(data[[metric, category]], category, metric)
                    gb = gb.compute()
                    gb.name = sample
                    results[sample] = gb
                    if category == "genus":
                        try:
                            results[sample] = results[sample].drop("g__unclassified")
                        except KeyError:
                            pass
                    results[sample].to_csv(outfile)
                else:
                    print(f"{outfile} already exists. Skipping.")
    return results


def main(args):
    print("Loading taxonomy file from disk.")
    tax = dd.read_csv(args.taxonomy, sep="\t", dtype={"genome": "object",
                                                      "genome_name": "object"})
    tax = tax.set_index("seq_uuid")
    pattern = args.input
    filenames = glob(pattern)
    print(f"Found {len(filenames)} files.")
    print("Starting computation.")
    results = dask.compute(f(filenames, tax, args.outdir, args.force))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dask groupby")
    parser.add_argument("-c", "--cores", type=int, default=8, help="Number of cores per job.")
    parser.add_argument("-m", "--memory", type=str, default="64GB", help="Amount of memory per job.")
    parser.add_argument("-j", "--jobs", type=int, default=12, help="Number of jobs to start.")
    parser.add_argument("-i", "--input", type=str, default="/data/scratch/projects/punim1293/vini/data/bio-go-data/coverm/*.coverm", help="Pattern to input files.")
    parser.add_argument("-t", "--taxonomy", type=str, default="/data/gpfs/projects/punim1989/databases/genes/genome_reps_filtered_w_genome.txt", help="Path to taxonomy file.")
    parser.add_argument("-o", "--outdir", type=str, default="/data/scratch/projects/punim1293/vini/data/bio-go-data/dask-tables/tax/bysample/", help="Path to output directory.")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite of existing files.")
    args = parser.parse_args()
    with start_cluster(args.cores, args.memory, args.jobs) as client:
        main(args)
