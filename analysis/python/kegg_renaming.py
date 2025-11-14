#!/usr/bin/env python

"""
Rename KEGG features (remove punctuation etc.). This script is deprecated.
"""

from pathlib import Path
from glob import glob

import requests
import pandas as pd



datadir = "/local/path/to/data/counts/"
files = glob(datadir + "*_normalised.csv")
files = {Path(f).stem.replace("_trimmed_mean_formatted_clean_normalised", ""): f for f in files if "KEGG" in f}
files = {k: pd.read_csv(v, index_col=0, nrows=1).columns.to_list() for k, v in files.items()}
for k, v in files.items():
    print(k, len(v))

with open('/data/gpfs/projects/punim1293/vini/db/kegg/genes/ko/ko', 'r', encoding="utf-8", errors="replace") as ko_file:
    lines = ko_file.readlines()

ko_entries = dict()
for ix, line in enumerate(lines):
    if line.startswith("ENTRY"):
        ko_id = line.split()[1]
        ko_entries[ko_id] = lines[ix+2]

ko_entries = {k: v.replace("NAME        ", "").split(" [EC")[0] for k, v in ko_entries.items()}
ko_entries = {i: ko_entries[i[3:]] for i in files["KEGG_ko"] if i[3:] in ko_entries.keys()}
ko_missing = [i for i in files["KEGG_ko"] if i not in ko_entries.keys()]
ko_all = {**ko_entries, **{i: i for i in ko_missing}}

missing = pd.read_csv("/local/path/to/data/misc/renaming/ko_missing_2nd_try.csv", header=None)[0].to_list()


def search_kegg_entry(kegg_identifier):
    base_url = "https://rest.kegg.jp/find/ko/"
    full_url = f"{base_url}{kegg_identifier}"

    try:
        response = requests.get(full_url)
        if response.status_code == 200:
            # Parse the response content (assuming it's in plain text format)
            entry_info = response.text.strip()
            return entry_info
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error: {e}"

failed = 0
success = dict()

with open("/local/path/to/data/misc/renaming/ko_entries_2nd_try.csv", "a") as f:
    for i, kegg_identifier in enumerate(missing):
        if kegg_identifier in success.keys():
            continue
        if i % 10 == 0:
            print(f"Processed {i}/{len(missing)} entries.", end="\r")
        try:
            result = search_kegg_entry(kegg_identifier)
            success[kegg_identifier] = result
            f.write(result+"\n")
        except:
            failed += 1
            pass

print("Done. Failed:", failed)