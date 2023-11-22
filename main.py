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

forgotten = {}

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
    counties_df = read_county_data()
    zip_codes_df = read_zip_data()

    # If stores are not already populated with county, then do so and save it
    if not "county" in stores_df.columns:
        stores_df["county"] = stores_df["postal_code"].apply(lambda x: locate_county(zip_codes_df, x))
        stores_df.to_csv(path_or_buf=AGGREGATED_STORE_FILE)

    
    # PREPARE STORE COUNT DATA
    stores_per_county = stores_df.groupby(["county","company_name"])[["county", "company_name"]].value_counts().to_frame().reset_index()
    county_store_count_df = stores_per_county.pivot_table(index="county", columns="company_name", values="count", fill_value=0)
    county_store_count_df["median"] = county_store_count_df.apply(lambda x: get_county_median(counties_df, x.name), axis=1)
    county_store_count_df = county_store_count_df[county_store_count_df["median"].notna()]

    # PREPARE STORE TYPE DATA
    types_per_county = stores_df.groupby(["county", "type"])[["county", "type"]].value_counts().to_frame().reset_index()
    county_store_type_df = types_per_county.pivot_table(index="county", columns="type", values="count", fill_value=0)
    county_store_type_df["total_count"] = county_store_type_df.apply(lambda x: x.sum() , axis=1)
    county_store_type_df["median"] = county_store_type_df.apply(lambda x: get_county_median(counties_df, x.name), axis=1)
    county_store_type_df = county_store_type_df[county_store_type_df["median"].notna()]


    # PREPARE STORE SUBTYPE DATA
    subtypes_per_county = stores_df.groupby(["county", "subtype"])[["county", "subtype"]].value_counts().to_frame().reset_index()
    county_store_subtype_df = subtypes_per_county.pivot_table(index="county", columns="subtype", values="count", fill_value=0)
    county_store_subtype_df["total_count"] = county_store_subtype_df.apply(lambda x: x.sum() , axis=1)
    county_store_subtype_df["median"] = county_store_subtype_df.apply(lambda x: get_county_median(counties_df, x.name), axis=1)
    county_store_subtype_df = county_store_subtype_df[county_store_subtype_df["median"].notna()]

    print(county_store_subtype_df)

    # CORRELATION COEFFICIENTS
    median_cor = county_store_count_df.corr(numeric_only=True)["median"].sort_values(ascending=True)
    # company_corr_df = median_cor.to_frame().reset_index().set_index("company_name").rename(columns={"median": "correlation_coefficient"})
    company_corr_df = median_cor.to_frame().reset_index().rename(columns={"median": "correlation_coefficient"})
    company_corr_df = company_corr_df[company_corr_df["company_name"] != "median"]

    # Factors for analysis
    dependent = "median"
    store_factors = list(filter(lambda x: (x != dependent), county_store_count_df.columns))
    type_factors =  list(filter(lambda x: (not(x == dependent or x == "total_count")), county_store_type_df.columns))
    subtype_factors =  list(filter(lambda x: (not(x == dependent or x == "total_count")), county_store_subtype_df.columns))


    menu_options = [
        lambda: overall_county_store_data(county_store_count_df),
        lambda: all_store_analysis(county_store_count_df, dependent, store_factors),
        lambda: store_count_analysis(county_store_count_df, dependent, store_factors),
        lambda: high_store_corr_coef_analysis(county_store_count_df,company_corr_df, dependent, store_factors),
        lambda: view_store_makeup_analysis(county_store_count_df, dependent, store_factors),
        lambda: view_type_count_analysis(county_store_type_df, dependent, type_factors),
        lambda: view_type_makeup_analysis(county_store_type_df, dependent, type_factors),
        lambda: view_subtype_count_analysis(county_store_subtype_df, dependent, subtype_factors),
        lambda: view_subtype_makeup_analysis(county_store_subtype_df, dependent, subtype_factors),
    ]

    user_choice = menu_choice()
    while user_choice != "exit":
        menu_options[user_choice]()
        press_enter()
        user_choice = menu_choice()

def menu_choice():
    print("1. View Overall County Store Data")
    print("2. View Company Correlation Coefficients")
    print("3. View Analysis for All Stores")
    print("4. View Analysis for Store Count")
    print("5. View Analysis for Stores with high Correlation Coefficients")
    print("6. View Store Makup Analysis")
    print("7. View Store Type Count Analysis")
    print("8. View Store Type Makup Analysis")
    print("9. View Store Subtype Count Analysis")
    print("10. View Store Subtype Makup Analysis")
    print(f"{Colors.RED}exit to exit{Colors.RESET}")
    choice = input("\nSelect Option: ").lower()

    if(choice == "exit"):
        return choice
    
    try:
        num = (int(choice) - 1)
        return num
    except:
        print(f"{Colors.RED}{choice}{Colors.RESET} is not a valid optin!")
        return press_enter()

    
def overall_county_store_data(county_store_count_df: pd.DataFrame):
    print(f"\n\n{Colors.CYAN}STRUCTURED STORE DATA WITH COUNTY{Colors.RESET}\n")
    print(county_store_count_df)

def all_store_analysis(county_store_count_df: pd.DataFrame, dependent: str, store_factors: list[str]):
    display_ols_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title="Store Count Model")


def store_count_analysis(county_store_count_df: pd.DataFrame, dependent: str, store_factors: list[str]):
    store_count_pred_df = county_store_count_df.reset_index()[["county", "median"]]
    store_count_pred_df["store_count"] = list(county_store_count_df[store_factors].sum(axis=1).to_frame()[0])
    
    display_ols_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title="Total Store Count Model")
    
def high_store_corr_coef_analysis(county_store_count_df: pd.DataFrame, company_corr_df, dependent: str, store_factors: list[str]):
    county_store_count_df.sort_values(["median"], ascending=True, inplace=True)
    high_corr_store_factors = list(filter(lambda x:  (company_corr_df[company_corr_df['company_name'] == x]['correlation_coefficient'] > .3).all(), store_factors))

    display_ols_model(df=county_store_count_df, dependent=dependent, factors=high_corr_store_factors, title="High Correlation Coefficient Store Count Model")  

def view_store_makeup_analysis(county_store_count_df: pd.DataFrame, dependent: str, store_factors: list[str]):
    county_store_count_df["store_count"] = list(county_store_count_df[store_factors].sum(axis=1).to_frame()[0])
    county_store_count_df = county_store_count_df[county_store_count_df["store_count"] > 0]

    for variable in store_factors:
        county_store_count_df[variable] = county_store_count_df.apply(lambda x: x[variable] / x["store_count"], axis=1)
    
    display_ols_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title="Store Percentage Model")

def view_type_count_analysis(county_store_type_df: pd.DataFrame, dependent: str, type_factors: list[str]):
    display_ols_model(df=county_store_type_df, dependent=dependent, factors=type_factors, title="Store Type Count Model")

def view_type_makeup_analysis(county_store_type_df: pd.DataFrame, dependent: str, type_factors: list[str]):

    for variable in type_factors:
        county_store_type_df[variable] = county_store_type_df.apply(lambda x: x[variable]/x["total_count"], axis=1)
    
    display_ols_model(df=county_store_type_df, dependent=dependent, factors=type_factors, title="Store Type Percentage Model")


def view_subtype_count_analysis(county_store_subtype_df: pd.DataFrame, dependent: str, subtype_factors: list[str]):
    display_ols_model(df=county_store_subtype_df, dependent=dependent, factors=subtype_factors, title="Store Subtype Count Model")

def view_subtype_makeup_analysis(county_store_subtype_df: pd.DataFrame, dependent: str, subtype_factors: list[str]):

    for variable in subtype_factors:
        county_store_subtype_df[variable] = county_store_subtype_df.apply(lambda x: x[variable]/x["total_count"], axis=1)
    display_ols_model(df=county_store_subtype_df, dependent=dependent, factors=subtype_factors, title="Store Subtype Percentage Model")

def display_ols_model(df: pd.DataFrame, dependent: str, factors: list[str], title: str):
    df.sort_values([dependent], ascending=True, inplace=True)
    ols_model = ols(formula=f"{dependent} ~ {' + '.join(factors)}", data=df).fit()
    print(ols_model.summary())

    df_corr = df[factors + [dependent]].corr(numeric_only=True)[dependent].sort_values(ascending=True).to_frame()
    df_corr = df_corr[df_corr.index != dependent]
    print(f"{Colors.CYAN}\nCORRELATION COEFFICIENTS{Colors.RESET}")
    print(df_corr)

    plt.bar(x=df_corr.index, height=df_corr[dependent])
    plt.title(f"Correlation for {title} Factors to {dependent}")
    plt.xlabel(f"{title} Factors")
    plt.xticks(rotation=50)
    plt.ylabel("Correlation Coeficient")
    plt.show()

    df.sort_values([dependent], ascending=True, inplace=True)

    df[f"pred_{dependent}"] = ols_model.predict(df[factors])

    dependent_col = df[dependent]
    pred_dependent_col = df[f"pred_{dependent}"]
    counties = df.index

    plt.scatter(counties, dependent_col, alpha=.6, s=2, color="blue")
    plt.scatter(counties, pred_dependent_col, alpha=.6, s=2, color="orange")
    plt.title(title)
    plt.xlabel("Counties")
    plt.ylabel(f"{dependent}")
    plt.xlabel
    plt.show()

    plt.scatter(dependent_col, pred_dependent_col, s=2, alpha=.6)
    plt.axline((0, 0), (dependent_col.max(), dependent_col.max()), color="green")
    plt.title(f"{title} Accuracy")
    plt.xlabel(f"{dependent} Actual")
    plt.ylabel(f"{dependent} Predicted")
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
    all_store_data["company_name"] = all_store_data["company_name"].apply(lambda x: re.sub("[^A-Za-z]", "_", x.strip()))
    all_store_data["type"] = all_store_data["company_name"].apply(get_company_type)
    all_store_data["subtype"] = all_store_data["company_name"].apply(get_company_subtype)
    return all_store_data

def get_company_type(company_name):
    company_types = {
        'Arbys': 'fast_food',
        'Buffalo_Wild_Wings': 'casual_dining',
        'Build_A_Bear': 'retail',
        'Burger_King': 'fast_food',
        "Caribou": "cafe",
        'ChickFilA': 'fast_food',
        'Chipotle': 'fast_food',
        "Culvers": "fast_casual",
        "CVS": "retail",
        'Dairy_Queen': 'fast_food',
        "Denny_s": "casual_dining",
        "Dominos": "fast_food",
            "Dunkin": "cafe",
        'Hardee_s': 'fast_food',
        'IHOP': 'casual_dining',
        "Jack_in_the_Box": "fast_food",
        "Jimmy_Johns": "fast_casual",
        "KFC": "fast_food",
        "Krispie_Kreme": "cafe",
        'Little_Caesars': 'fast_food',
        'McDonalds': 'fast_food',
        'Olive_Garden': 'casual_dining',
        "Panda_Express": "fast_casual",
        "Panera": "fast_casual",
        'Papa_Johns': 'fast_food',
        "Pizza_Hut": "fast_food",
        'Popeyes': 'fast_food',
        "Sonic": "fast_food",
        'Starbucks': 'cafe',
        'Subway': 'fast_food',
        'Target': 'retail',
        "Taco_Bell": "fast_food",
        'Tesla': 'automotive',
        'Tim_Hortons': 'cafe',
        'Trader_Joes': 'retail',
        "Waffle_House": "casual_dining",
        "Walgreen_s": "retail",
        'Wendys': 'fast_food',
        "Whataburger": "fast_food",
        'White_Castle': 'fast_food',
        "Wingstop": "fast_casual",
        "Zaxbys": "fast_casual",
    }
    return company_types[company_name]
    
    

def get_company_subtype(company_name):
    company_subtypes = {
        'Arbys': 'sandwich_shop',
        'Buffalo_Wild_Wings': 'sports_bar',
        'Build_A_Bear': 'toy_store',
        'Burger_King': 'burger_restaurant',
        "Caribou": "coffee_shop",
        'ChickFilA': 'chicken_restaurant',
        'Chipotle': 'mexican_grill',
         "Culvers": "burger_restaurant",
          "CVS": "pharmacy_and_convenience_store",
        'Dairy_Queen': 'ice_cream_shop',
        "Denny_s": "breakfast_restaurant",
        "Dominos": "pizza_restaurant",
         "Dunkin": "coffee_and_doughnut_shop",
        'Hardee_s': 'burger_restaurant',
        'IHOP': 'breakfast_restaurant',
        "Jack_in_the_Box": "burger_restaurant",
         "Jimmy_Johns": "sandwich_shop",
        "KFC": "chicken_restaurant",
        "Krispie_Kreme": "doughnut_shop",
        'Little_Caesars': 'pizza_restaurant',
        'McDonalds': 'burger_restaurant',
        'Olive_Garden': 'italian_restaurant',
        "Panda_Express": "chinese_restaurant",
         "Panera": "bakery_cafe",
        'Papa_Johns': 'pizza_restaurant',
        "Pizza_Hut": "pizza_restaurant",
        'Popeyes': 'chicken_restaurant',
        "Sonic": "drive_in_restaurant",
        'Starbucks': 'coffee_shop',
        'Subway': 'sandwich_shop',
         "Taco_Bell": "mexican_fast_food",
        'Target': 'department_store',
        'Tesla': 'electric_vehicles',
        'Tim_Hortons': 'coffee_shop',
        'Trader_Joes': 'grocery_store',
        "Waffle_House": "breakfast_restaurant",
         "Walgreen_s": "pharmacy_and_convenience_store",
        'Wendys': 'burger_restaurant',
         "Whataburger": "burger_restaurant",
        'White_Castle': 'burger_restaurant',
        "Wingstop": "chicken_wings_restaurant",
        "Zaxbys": "chicken_restaurant",
    }
    return company_subtypes.get(company_name, 'Unknown')


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