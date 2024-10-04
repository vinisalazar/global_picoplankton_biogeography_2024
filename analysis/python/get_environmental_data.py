import argparse
import logging
from pathlib import Path
import pandas as pd
from erddapy import ERDDAP
import pyo_oracle as bopy
from datetime import datetime


def initialize_erddap(dataset_id, constraints):
    """
    Initialize ERDDAP object with the given dataset_id and constraints.

    Args:
        dataset_id (str): ID of the dataset.
        constraints (dict): Constraints for the ERDDAP object.

    Returns:
        ERDDAP: Initialized ERDDAP object.
    """
    e = ERDDAP(server="https://erddap.bio-oracle.org/erddap/", protocol="griddap")
    e.dataset_id = dataset_id
    e.griddap_initialize()
    
    constraints = {**e._constraints_original, **constraints}
    for k, v in constraints.items():
        for d in e._constraints_original, e.constraints:
            if k in d.keys():
                d[k] = v

    return e


def main(args):
    """
    Main function to process data.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    log_file = args.log_file or f"get_env_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename=log_file)
    console_handler = logging.StreamHandler()  # For logging to console
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    logging.info("Reading input table.")
    df = pd.read_csv(args.input_table, index_col=0)

    constraints = {
        "latitude>=": df["latitude"].describe().loc["min"],
        "latitude<=": df["latitude"].describe().loc["max"],
        "latitude_step": 1,
        "longitude>=": df["longitude"].describe().loc["min"],
        "longitude<=": df["longitude"].describe().loc["max"],
        "longitude_step": 1
    }
    logging.info("Preprocessed input data.")

    layers = bopy.list_layers(time_period="future")
    sample_layers = [i for i in layers["datasetID"] if (("surf" in i))]
    logging.info(f"Found {len(sample_layers)} sample layers.")

    for ix, layer in enumerate(sample_layers):
        out_dataframe = f"{layer}_xr_dataframe.csv"
        out_dataframe = Path(args.output_dir.joinpath(out_dataframe))
        out_nc = f"{layer}_xr.nc"
        out_nc = Path(args.output_dir.joinpath(out_nc))
        # if not (out_dataframe.exists() and out_nc.exists()):
        if not (out_nc.exists()):
            try:
                logging.info(f"Processing layer '{layer}'.")
                e = initialize_erddap(layer, constraints)
                ds = e.to_xarray()
                ds.to_netcdf(out_nc)
                logging.info("Output files saved.")
            except:
                logging.error(f"Couldn't process layer '{layer}'.")
        else:
            logging.info(f"Skipping layer '{layer}'. Output files already exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data.")
    parser.add_argument("-i", "--input_table", type=Path, help="Path to input table file.")
    parser.add_argument("-o", "--output_dir", type=Path, help="Path to output directory.")
    parser.add_argument("-l", "--log_file", type=Path, default=None, help="Path to log file. Defaults to ''output_<timestamp>.log'.")
    args = parser.parse_args()
    main(args)
