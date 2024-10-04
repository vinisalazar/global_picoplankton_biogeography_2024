#!/usr/bin/env python

import sys
import json
import pandas as pd
sys.path.insert(0, "/local/path/to/scripts/")
from utils import print_with_timestamp, cog_mappings

env_cols = [
    'Salinity',
    'OceanTemperature',
    'DissolvedMolecularOxygen',
    'Silicate',
    'pH',
    'SeaWaterSpeed',
    'DissolvedIron',
    'Phosphate',
    'Nitrate',
    'SeaIceCover',
    'Chlorophyll',
    'PhotosyntheticallyAvailableRadiation',
    'DiffuseAttenuationCoefficientPAR',
    'MixedLayerDepth',
    'Terrain',
    'AirTemperature',
    'TotalPhytoplankton',
    'SeaIceThickness',
    'TotalCloudFraction',
]

def main():
    parentdir = "/local/path/to/"
    # Sample and genome metadata
    print_with_timestamp("Loading metadata.")
    smd = pd.read_csv(parentdir + "provinces_final/data/metadata_1454_cluster_labels.csv", index_col="index")
    with open("/data/gpfs/projects/punim1989/hub_provinces/data/misc/brite_descriptions.json") as f:
        brite_cols = json.load(f)
        
    brite_cols = {k: v if v else k for k, v in brite_cols.items()}

    # Count tables
    print_with_timestamp("Loading counts data.")
    genomes = pd.read_csv(parentdir + "provinces_final/data/counts/genome_trimmed_mean_formatted.csv")
    genus = pd.read_csv(parentdir + "provinces_final/data/counts/genus_trimmed_mean_formatted.csv")
    family = pd.read_csv(parentdir + "provinces_final/data/counts/family_trimmed_mean_formatted.csv")
    klass = pd.read_csv(parentdir + "provinces_final/data/counts/class_trimmed_mean_formatted.csv")
    order = pd.read_csv(parentdir + "provinces_final/data/counts/order_trimmed_mean_formatted.csv")
    phylum = pd.read_csv(parentdir + "provinces_final/data/counts/phylum_trimmed_mean_formatted.csv")
    cog = pd.read_csv(parentdir + "provinces_final/data/counts/COG_category_trimmed_mean_formatted_filtered.csv")
    brite = pd.read_csv(parentdir + "provinces_final/data/counts/BRITE_trimmed_mean_formatted.csv")
    kegg_Pathway = pd.read_csv(parentdir + "provinces_final/data/counts/KEGG_Pathway_trimmed_mean_formatted_filtered.csv")
    kegg_rclass = pd.read_csv(parentdir + "provinces_final/data/counts/KEGG_rclass_trimmed_mean_formatted.csv")
    kegg_ko = pd.read_csv(parentdir + "provinces_final/data/counts/KEGG_ko_trimmed_mean_formatted.csv")
    kegg_Reaction = pd.read_csv(parentdir + "provinces_final/data/counts/KEGG_Reaction_trimmed_mean_formatted.csv")

    # Check number of samples and features
    print_with_timestamp("Shape of data tables:")
    for df in "genomes cog brite kegg_Pathway kegg_rclass kegg_ko kegg_Reaction".split():
        print_with_timestamp(f"{df}: {eval(df).shape[0]} x {eval(df).shape[1] - 1}")

    # Conciliate row names
    print_with_timestamp("Conciliating row names.")
    genomes = genomes.groupby("index").sum(numeric_only=True)
    genus = genus.groupby("index").sum(numeric_only=True)
    family = family.groupby("index").sum(numeric_only=True)
    klass = klass.groupby("index").sum(numeric_only=True)
    order = order.groupby("index").sum(numeric_only=True)
    phylum = phylum.groupby("index").sum(numeric_only=True)
    cog = cog.groupby("index").sum(numeric_only=True)
    brite = brite.groupby("index").sum(numeric_only=True)
    kegg_Pathway = kegg_Pathway.groupby("index").sum(numeric_only=True)
    kegg_rclass = kegg_rclass.groupby("index").sum(numeric_only=True)
    kegg_ko = kegg_ko.groupby("index").sum(numeric_only=True)
    kegg_Reaction = kegg_Reaction.groupby("index").sum(numeric_only=True)

    smd = smd[smd.index.isin(cog.index)].sort_index()

    # Apply metadata cutoffs
    smd = smd.query("depth <= 200")
    genomes = genomes.loc[smd.index]
    genus = genus.loc[smd.index]
    family = family.loc[smd.index]
    klass = klass.loc[smd.index]
    order = order.loc[smd.index]
    phylum = phylum.loc[smd.index]
    cog = cog.loc[smd.index]
    brite = brite.loc[smd.index].rename(columns=brite_cols)
    kegg_Pathway = kegg_Pathway.loc[smd.index]
    kegg_rclass = kegg_rclass.loc[smd.index]
    kegg_ko = kegg_ko.loc[smd.index]
    kegg_Reaction = kegg_Reaction.loc[smd.index]

    # Additional column filtering
    kegg_Pathway = kegg_Pathway[[i for i in kegg_Pathway.columns if not i.startswith("map")]]
    kegg_Pathway = kegg_Pathway[[i for i in kegg_Pathway.columns if not i.startswith("ko")]]

    cog.columns = [i for ix, i in enumerate(cog.columns)]
    brite.columns = [i[:50] + "_" + str(ix) for ix, i in enumerate(brite.columns)]
    kegg_Pathway.columns = [i for ix, i in enumerate(kegg_Pathway.columns)]
    kegg_rclass.columns = [i[:20] + "_" + str(ix) for ix, i in enumerate(kegg_rclass.columns)]
    kegg_ko.columns = [i[:50] + "_" + str(ix) for ix, i in enumerate(kegg_ko.columns)]
    kegg_Reaction.columns = [i[:20] + "_" + str(ix) for ix, i in enumerate(kegg_Reaction.columns)]

    print_with_timestamp("Writing R tables.")
    genomes.to_csv(parentdir + f"provinces_final/data/counts/genomes_trimmed_mean_formatted_clean.csv")
    genus.to_csv(parentdir + f"provinces_final/data/counts/genus_trimmed_mean_formatted_clean.csv")
    family.to_csv(parentdir + f"provinces_final/data/counts/family_trimmed_mean_formatted_clean.csv")
    klass.to_csv(parentdir + f"provinces_final/data/counts/class_trimmed_mean_formatted_clean.csv")
    order.to_csv(parentdir + f"provinces_final/data/counts/order_trimmed_mean_formatted_clean.csv")
    phylum.to_csv(parentdir + f"provinces_final/data/counts/phylum_trimmed_mean_formatted_clean.csv")
    brite.to_csv(parentdir + f"provinces_final/data/counts/BRITE_trimmed_mean_formatted_clean.csv")
    cog.to_csv(parentdir + f"provinces_final/data/counts/COG_category_trimmed_mean_formatted_filtered_clean.csv")
    kegg_Pathway.to_csv(parentdir + f"provinces_final/data/counts/KEGG_Pathway_trimmed_mean_formatted_filtered_clean.csv")
    kegg_rclass.to_csv(parentdir + f"provinces_final/data/counts/KEGG_rclass_trimmed_mean_formatted_clean.csv")
    kegg_ko.to_csv(parentdir + f"provinces_final/data/counts/KEGG_ko_trimmed_mean_formatted_clean.csv")
    kegg_Reaction.to_csv(parentdir + f"provinces_final/data/counts/KEGG_Reaction_trimmed_mean_formatted_clean.csv")
    smd[[i for i in env_cols if i in smd.columns]].to_csv(parentdir + f"provinces_final/data/R/env_data_clean.csv")
    smd.to_csv(parentdir + f"provinces_final/data/R/sample_metadata_clean.csv")


if __name__ == "__main__":
    main()
    print_with_timestamp("All done.")
    sys.exit()
