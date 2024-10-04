import re
import json
import argparse
import pandas as pd
from glob import glob
from pathlib import Path
from tqdm import tqdm


# BRITE mappings
with open("/data/gpfs/projects/punim1989/hub_provinces/data/misc/brite_descriptions.json") as f:
        brite_cols = json.load(f)
        
brite_cols = {k: v if v else k for k, v in brite_cols.items()}

mappings = {
    "COG_category": {
        "A": "RNA processing and modification",
        "B": "Chromatin structure and dynamics",
        "C": "Energy production and conversion",
        "D": "Cell cycle control, cell division, chromosome partitioning",
        "E": "Amino acid transport and metabolism",
        "F": "Nucleotide transport and metabolism",
        "G": "Carbohydrate transport and metabolism",
        "H": "Coenzyme transport and metabolism",
        "I": "Lipid transport and metabolism",
        "J": "Translation, ribosomal structure and biogenesis",
        "K": "Transcription",
        "L": "Replication, recombination and repair",
        "M": "Cell wall/membrane/envelope biogenesis",
        "N": "Cell motility",
        "O": "Post-translational modification, protein turnover, chaperones",
        "P": "Inorganic ion transport and metabolism",
        "Q": "Secondary metabolites biosynthesis, transport and catabolism",
        "R": "General function prediction only",
        "S": "Function unknown",
        "T": "Signal transduction mechanisms",
        "U": "Intracellular trafficking, secretion, and vesicular transport",
        "V": "Defense mechanisms",
        "W": "Extracellular structures",
        "Y": "Nuclear structure",
        "Z": "Cytoskeleton",
    },
    "KEGG_ko": pd.read_csv("/local/path/to/data/misc/renaming/ko_entries.tsv", sep="\t", index_col=0, header=None).iloc[:, 0].to_dict(),
    "KEGG_Pathway": pd.read_csv("/local/path/to/data/misc/renaming/pathway_entries.tsv", sep="\t", index_col=0, header=None).iloc[:, 0].to_dict(),
    "KEGG_Reaction": pd.read_csv("/local/path/to/data/misc/renaming/reaction_entries.tsv", sep="\t", index_col=0, header=None, on_bad_lines="skip").iloc[:, 0].to_dict(),
    "KEGG_rclass": pd.read_csv("/local/path/to/data/misc/renaming/rclass_entries.tsv", sep="\t", index_col=0, header=None).iloc[:, 0].to_dict(),
    "BRITE": brite_cols
}

mappings["KEGG_rclass"] = {(k[3:] if k.startswith("rc:") else k): v for k, v in mappings["KEGG_rclass"].items()}
mappings["KEGG_rclass"] = {k:(k if v == "[]" else v) for k, v in mappings["KEGG_rclass"].items()}
kegg_ko_2nd_attempt = pd.read_csv("/local/path/to/data/misc/renaming/ko_entries_2nd_try.csv", sep="\t", index_col=0, header=None).iloc[:, 0].to_dict()
mappings["KEGG_ko"] = {**mappings["KEGG_ko"], **kegg_ko_2nd_attempt}


def format_colname(string):
    # Define the pattern to match ".X" where X is any integer
    pattern = r'\.\d+$'
    # Use regex to find matches
    try:
        match = re.search(pattern, string)
        # If match found, strip it from the string
        if match:
            return string[:match.start()]
        else:
            return string
    except TypeError:  # when we have a float
        return str(string)


def process_data(input_dir, output_dir, force=False, mode="tax"):
    metrics = "tpm trimmed_mean rpkm read_count".split()
    metrics = ["trimmed_mean",]
    if mode == "tax":
        ranks = {Path(i).stem.split("_")[0] for i in glob(input_dir + "/*")}
    elif mode == "func":
        ranks = "BRITE COG_category KEGG_Reaction KEGG_rclass KEGG_ko KEGG_Pathway".split()

    md = pd.read_csv("/local/path/to/data/metadata_1454_cluster_labels.csv")
    for rank in tqdm(ranks):
        for metric in tqdm(metrics):
            input_file = f"{input_dir}/{rank}_{metric}.csv"
            if not Path(input_file).exists():
                print(f"File '{input_file}' does not exist. Skipping.")
                continue
            outfile = f"{output_dir}/{rank}_{metric}_formatted.csv"
            if Path(outfile).exists() and not force:
                print(f"'{outfile}' exists. Skipping.")
                continue
            else:
                pass
            df = pd.read_csv(input_file, index_col=0)
            if rank in mappings.keys():
                if rank == "KEGG_ko":
                    df.columns = [i.replace("ko:", "") for i in df.columns]
                df = df.rename(columns=mappings[rank])

            df["sample_name"] = [i.split("_T")[0] for i in df.index]
            df["sample_name"] = df["sample_name"].str.split("_").apply(lambda l: l[0] if (not l[0].startswith("Arc")) and (len(l) > 1) else "_".join(l))
            df["sample_name"] = df["sample_name"].apply(lambda s: "_".join(s.split("_")[:-1]) if len(s.split("_")) > 3 else s)
            df = df.groupby("sample_name").sum()
            df = df.merge(md, left_index=True, right_on="sample_name", how="inner")
            df = df.set_index("index")
            try:
                df = df[[i for i in df.columns if i not in md.columns]]
            except:
                breakpoint()

            # Sum columns with same name
            df.columns = [format_colname(s) for s in df.columns]
            if 'nan' in df.columns:
                df = df.drop('nan', axis=1)
            df = df.T.groupby(df.columns).sum().T

            df.to_csv(outfile)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and format it.")
    parser.add_argument("-i", "--input_dir", type=str, help="Path to the input directory containing the CSV files.")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to the output directory for saving formatted files.")
    parser.add_argument("-m", "--mode", type=str, default="tax", help="Whether to run in taxonomy or functional mode.")
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    process_data(args.input_dir, args.output_dir, args.force, args.mode)

