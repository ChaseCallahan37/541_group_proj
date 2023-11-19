from glob import glob
import os.path as path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re

# GLOBAL CONSTANTS
AGGREGATED_STORE_FILE = "store-data-aggregate.csv"
HOUSING_PRICES_FILE = "./housing-data/realtor-data.csv"
COUNTIES_FILE = "./counties-data/counties.csv"
RAW_COUNTIES_FILE = "./counties-data/raw-counties.csv"
ZIP_CODES_FILE = "./zip-code-data/zip_code_database.csv"


def main():
    stores_df = read_store_data()
    housing_df = read_housing_data()
    zip_codes_df = read_zip_data()
    county_df = read_county_data()

    # If stores are not already populated with county, then do so and save it
    if not "county" in stores_df.columns:
        stores_df["county"] = stores_df["postal_code"].apply(lambda x: locate_county(zip_codes_df, x))
        stores_df.to_csv(path_or_buf=AGGREGATED_STORE_FILE)


def locate_county(zip_codes_df, zip):
    try:
        first_section = zip.split(" ")[0]
        zip_int = int(first_section)
        return zip_codes_df.loc[zip_int]["county"]
    except:
        return np.NaN
    


def read_store_data() -> pd.DataFrame:
    stores_df = retrieve_store_file()
    return stores_df
    
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
    housing_df = csv_to_df(HOUSING_PRICES_FILE)
    # We convert all zips from float to int then string, at that point we make the string 5 long by adding 0s to the start
    # If the zip code is NaN then we maintain the NaN designation
    housing_df["zip_code"] = housing_df["zip_code"].apply(lambda x: str(int(x)).zfill(5) if not pd.isnull(x) else np.nan)
    return housing_df

def read_zip_data() -> pd.DataFrame:
    zip_code_df = pd.read_csv(ZIP_CODES_FILE)
    zip_code_df = zip_code_df.set_index(["zip"])
    zip_code_df["county"] = zip_code_df["county"].apply(lambda x: re.sub("County", "", x).strip() if pd.notnull(x) else np.NaN)
    return zip_code_df

def read_county_data() -> pd.DataFrame:
    county_df = read_county_file()
    return county_df

def read_county_file() -> pd.DataFrame:
    if path.isfile(COUNTIES_FILE):
        return pd.read_csv(COUNTIES_FILE)
    return prepare_counties_data(RAW_COUNTIES_FILE)

# Assumes , as delimiter by default
def csv_to_df(file_name: str, delimiter: str =",") -> pd.DataFrame:
    return pd.read_csv(file_name, sep=delimiter)

def prepare_counties_data(file_name: str) -> pd.DataFrame:
    counties_file = open(file_name)
    counties = []
    for line in counties_file:
        county_data = {}
        line = line.split(",")
        county_raw = line[0]
        county_data["county"] = re.sub("County", "", county_raw).strip()

        rest = "".join(line[1:])
        median = rest.split("$")[1].strip()
        
        county_data["median"] = int(median)

        counties.append(county_data)
    df = pd.DataFrame(counties)
    df.to_csv(path_or_buf=COUNTIES_FILE)
    return df


main()