from glob import glob
import os.path as path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols 
from scipy.optimize import curve_fit



# pd.set_option('display.max_rows', None)

# GLOBAL CONSTANTS
AGGREGATED_STORE_FILE = "store-data-aggregate.csv"
AGGREGATED_STORE_RAW_URL = "https://raw.githubusercontent.com/ChaseCallahan37/541_group_proj/main/store-data-aggregate.csv"

REALTOR_FILE = "./realtor-data/realtor-data.csv"
RAW_REALTOR_FILE = "https://raw.githubusercontent.com/ChaseCallahan37/541_group_proj/main/realtor-data/realtor-data.csv"

RAW_COUNTIES_FILE = "./counties-data/raw-counties.csv"
COUNTIES_FILE = "./counties-data/counties.csv"

ZIP_CODES_FILE = "./zip-code-data/zip_code_database.csv"

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'  # Reset color to default


def main():
    stores_df = read_store_data()
    housing_df = read_housing_data()
    counties_df = read_county_data()
    zip_codes_df = read_zip_data()

    # If stores are not already populated with county, then do so and save it
    if not "county" in stores_df.columns:
        stores_df["county"] = stores_df["postal_code"].apply(lambda x: locate_county(zip_codes_df, x))
        stores_df.to_csv(path_or_buf=AGGREGATED_STORE_FILE)

    stores_per_county = stores_df.groupby(["county","company_name"])[["county", "company_name"]].value_counts().to_frame().reset_index()
    
    print(stores_per_county)

    county_store_df = stores_per_county.pivot_table(index="county", columns="company_name", values="count", fill_value=0)
    county_store_df["median"] = county_store_df.apply(lambda x: get_county_median(counties_df, x.name), axis=1)
    county_store_df = county_store_df[county_store_df["median"].notna()]

    # Replace the column names with names that do not use spaces
    new_names = {}
    for name in county_store_df.columns:
        new_names[name] = re.sub("[^A-Za-z]", "_", name.strip())
    county_store_df.rename(columns=new_names, inplace=True)

    # CORRELATION COEFFICIENTS
    median_cor = county_store_df.corr(numeric_only=True)["median"].sort_values(ascending=True)
    # company_corr_df = median_cor.to_frame().reset_index().set_index("company_name").rename(columns={"median": "correlation_coefficient"})
    company_corr_df = median_cor.to_frame().reset_index().rename(columns={"median": "correlation_coefficient"})
    company_corr_df = company_corr_df[company_corr_df["company_name"] != "median"]

    # Factors for analysis
    dependent = "median"
    independents = list(filter(lambda x: (x != dependent), county_store_df.columns))

    user_choice = menu_choice()
    while user_choice != "exit":
        if(user_choice == "1"):
            overall_county_store_data(county_store_df)
        elif(user_choice == "2"):
            show_company_corr_coef(company_corr_df)
        elif(user_choice == "3"):
            all_store_analysis(county_store_df, dependent, independents)
        elif(user_choice == "4"):
            store_count_analysis(county_store_df, dependent, independents)
        elif(user_choice == "5"):
            high_store_corr_coef_analysis(county_store_df,company_corr_df, independents)
        elif(user_choice == "exit"):
            print("Bye!")
        else:
            print("INVALID INPUT!")
        press_enter()
        user_choice = menu_choice()

def menu_choice() -> str:
    print("1. View Overall County Store Data")
    print("2. View Company Correlation Coefficients")
    print("3. View Analysis for All Stores")
    print("4. View Analysis for Store Count")
    print("5. View Analysis for Stores with high Correlation Coefficients")
    print("exit to exit")
    return input("Select Option: ").lower()

    
def overall_county_store_data(county_store_df: pd.DataFrame):
    print(f"\n\n{Colors.CYAN}STRUCTURED STORE DATA WITH COUNTY{Colors.RESET}\n")
    print(county_store_df)

def show_company_corr_coef(company_corr_df: pd.DataFrame):
    print(f"\n\n{Colors.CYAN}CORRELATION COEFICIENTS{Colors.RESET}\n")
    print(company_corr_df)

    plt.bar(x=company_corr_df["company_name"], height=company_corr_df["correlation_coefficient"])
    plt.title("Company Count Correlation to Median Home Price")
    plt.xlabel("Companies")
    plt.xticks(rotation=60)
    plt.ylabel("Correlation Coefficient")
    plt.yticks(np.arange(0, 1.05, .05))
    plt.show()

def all_store_analysis(county_store_df: pd.DataFrame, dependent: str, independents):
    # OLS ANALYSIS FOR ALL STORES

    # Join all the column names together via a space (except for the dependent)
    # variable, so that they may be interpolated into the formula
    all_store_ols = ols(formula=f"{dependent} ~ {' + '.join(independents)}", data=county_store_df).fit()

    print(f"\n\n{Colors.CYAN}OLS MODEL SUMMARY{Colors.RESET}\n")
    print(all_store_ols.summary())

    # MEDIAN PREDICTION FOR ALL STORES COUNT
    all_stores_pred_df = county_store_df.reset_index()[["county", "median"]]
    all_stores_pred_df["pred_median"] = list(all_store_ols.predict(county_store_df[independents]))
    all_stores_pred_df.sort_values(["median"], inplace=True, ascending=True)
    print(all_stores_pred_df[["median", "pred_median"]])
    plt.title("")
    plt.scatter(all_stores_pred_df["county"], all_stores_pred_df["median"], alpha=1, s=2)
    plt.scatter(all_stores_pred_df["county"], all_stores_pred_df["pred_median"], alpha=1, s=2)
    plt.show()

    median = all_stores_pred_df["median"]
    pred_median = all_stores_pred_df["pred_median"]
    plt.scatter(median, pred_median, s=2)
    plt.axline((0, 0), (median.max(), pred_median.max()), color="red")

    plt.title("Accuracy of All Store Model")
    plt.xlabel("Actual Median")
    plt.ylabel("Predicted Median")
    plt.show()


def store_count_analysis(county_store_df: pd.DataFrame, dependent: str, independents):

    # OLS ANALYSIS FOR STORE COUNT
    store_count_pred_df = county_store_df.reset_index()[["county", "median"]]
    print(county_store_df[independents])
    store_count_pred_df["store_count"] = list(county_store_df[independents].sum(axis=1).to_frame()[0])
    store_count_ols = ols(formula=f"{dependent} ~ store_count", data=store_count_pred_df).fit()
    print(store_count_ols.summary())

    store_count_pred_df["price_pred"] = store_count_ols.predict(store_count_pred_df["store_count"])
    store_count_pred_df.sort_values(["median"], ascending=True, inplace=True)
    print(store_count_pred_df)
    plt.scatter(store_count_pred_df["county"], store_count_pred_df["median"], alpha=1, s=2)
    plt.scatter(store_count_pred_df["county"], store_count_pred_df["pred_median"], alpha=1, s=2)

    plt.show()
    
def high_store_corr_coef_analysis(county_store_df: pd.DataFrame, company_corr_df, independents):

    # OLS ANALYSIS FOR STORES WITH GREATER THAN .3 CORRELATION COEFFICIENTS
    county_store_df.sort_values(["median"], ascending=True, inplace=True)
    high_corr_independents = list(filter(lambda x:  (company_corr_df[company_corr_df['company_name'] == x]['correlation_coefficient'] > .3).all(), independents))
    high_corr_ols = ols(formula=f"median ~ {' + '.join(high_corr_independents)}", data=county_store_df).fit()

    high_corr_pred_df = county_store_df.reset_index()[["county", "median"]]
    high_corr_pred_df["pred_median"] = list(high_corr_ols.predict(county_store_df[high_corr_independents]))
    print(high_corr_pred_df)
    plt.scatter(high_corr_pred_df["county"], high_corr_pred_df["median"], alpha=1, s=2)
    plt.scatter(high_corr_pred_df["county"], high_corr_pred_df["pred_median"], alpha=1, s=2)
    plt.show()
    median = high_corr_pred_df["median"]
    pred_median = high_corr_pred_df["pred_median"]
    plt.scatter(median, pred_median, s=2)
    plt.axline((0, 0), (median.max(), pred_median.max()), color="red")

    plt.title("Accuracy of Model High Store Correlation Model")
    plt.xlabel("Actual Median")
    plt.ylabel("Predicted Median")
    plt.show()     


# Recieves the dataset with the counties information regarding median
# home prices and the specific county that we are looking for
def get_county_median(counties_df: pd.DataFrame, county: str) :
    try:
        # Attempt to locate the county within the dataset
        median = counties_df.loc[county]["median"]
        # If multiple counties with same name exist
        # Then return nan
        if isinstance(median, pd.Series):
            return np.NaN

        return median
    except:
        # If any failures occur, default to
        # nan
        return np.NaN
    
def locate_county(zip_codes_df, zip):
    try:
        first_section = zip.split(" ")[0]
        zip_int = int(first_section)
        return zip_codes_df.loc[zip_int]["county"].lower()
    except:
        return np.NaN
    


def read_store_data() -> pd.DataFrame:
    all_store_data = retrieve_store_file()
    return all_store_data

def retrieve_realtor_file() -> pd.DataFrame:
    # Checks for aggregated first, if not found,
    # then aggregated will be generated and returned
    if not path.isfile(REALTOR_FILE):
        return read_housing_data()
    # Otherwise we grab the aggregated data and return it
    # as a data frame
    else:
        return csv_to_df(REALTOR_FILE)
    
def retrieve_store_file() -> pd.DataFrame:
    if path.isfile(AGGREGATED_STORE_FILE):
        return csv_to_df(AGGREGATED_STORE_FILE)

    if len(list(glob("store-data/*.csv"))) > 0:
        return aggregate_seperated_store_files()

    return pull_down_store_csv()

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
    housing_df = read_housing_file()
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
    county_df = county_df.set_index("county")
    return county_df

def read_county_file() -> pd.DataFrame:
    if path.isfile(COUNTIES_FILE):
        return pd.read_csv(COUNTIES_FILE)
    return prepare_counties_data(RAW_COUNTIES_FILE)

def read_housing_file() -> pd.DataFrame:
    if not path.isfile(REALTOR_FILE):
        pulled_housing_data = pd.read_csv(RAW_REALTOR_FILE)
        pulled_housing_data.to_csv(path_or_buf="./housing-data/realtor-data.csv")
        return pulled_housing_data
    return pd.read_csv(REALTOR_FILE)

# Assumes , as delimiter by default
def csv_to_df(file_name: str, delimiter: str =",") -> pd.DataFrame:
    return pd.read_csv(file_name, sep=delimiter)

def prepare_counties_data(file_name: str) -> pd.DataFrame:
    counties_file = open(file_name)
    counties = []
    for line in counties_file:
        county = {}
        line = line.split(",")

        try: 
            # Clean the county name by taking out the word 'County'
            county["county"] = re.sub("County", "", line[0]).strip().lower()
        except:
            county["county"] = np.Nan

        try:
            # Join th rest of the data together to split on a '$' instead
            stringy_median = "".join(line[1:]).split("$")[1].strip()
            county["median"] = int(stringy_median)
        except:
            county["median"] = np.NaN
        
        counties.append(county)

    counties_df = pd.DataFrame(counties)

    # Save counties data to file so that this operation only
    # needs to be run to generate files
    counties_df.to_csv(path_or_buf=COUNTIES_FILE)
    
    return counties_df


def pull_down_store_csv():
    url = AGGREGATED_STORE_RAW_URL
    df = pd.read_csv(url)

    # Removing Store indexes that are nonunique
    df.drop(['Unnamed: 0'], axis=1, inplace=True)

    df.to_csv(AGGREGATED_STORE_FILE)
    return df

def press_enter():
    input("\nPress enter to continue...")
    for i in range(0, 100): 
        print("\n")

main()