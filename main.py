from glob import glob
import os.path as path
import pandas as pd

# GLOBAL CONSTANTS
AGGREGATED_STORE_FILE = "store-data-aggregate.csv"
HOUSING_PRICES_FILE = "./housing-data/fulton_county_data.csv"


def main():
    stores_df = read_store_data()
    housing_df = read_housing_data()

    print(stores_df)

def read_store_data() -> pd.DataFrame:
    all_store_data = retrieve_store_file()
    return all_store_data[all_store_data["state_alpha"] == "GA"]
    
def retrieve_store_file() -> pd.DataFrame:
    # Checks for aggregated first, if not found,
    # then aggregated will be generated and returned
    if not path.isfile(AGGREGATED_STORE_FILE):
        return aggregate_seperated_store_files()
    # Otherwise we grab the aggregated data and return it
    # as a data frame
    else:
        return csv_to_df(AGGREGATED_STORE_FILE)

def aggregate_seperated_store_files() -> pd.DataFrame:
    # Get a reference to all csv files in the store-data
    # folder that have a .csv file extension (stores them)
    # in a list
    store_csvs = glob("store-data/*.csv")

    # Takes each file reference and converts them to a data
    # frame. Final result is a list of data frames
    seperated_store_dfs = list(map(csv_to_df, store_csvs))

    # Concats each individual df into a single large one
    stores_df = pd.concat(seperated_store_dfs)

    # Save to file so that this computation does not have
    # to run every time
    stores_df.to_csv(AGGREGATED_STORE_FILE)

    return stores_df

def read_housing_data() -> pd.DataFrame:
    return csv_to_df(HOUSING_PRICES_FILE)

# Assumes , as delimiter by default
def csv_to_df(file_name: str, delimiter: str =",") -> None:
    return pd.read_csv(file_name, sep=delimiter)

main()