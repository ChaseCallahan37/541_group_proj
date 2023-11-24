from glob import glob
import os.path as path
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols, logit
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report

# GLOBAL CONSTANTS
AGGREGATED_STORE_FILE = "store-data-aggregate.csv"
AGGREGATED_STORE_RAW_URL = "https://raw.githubusercontent.com/ChaseCallahan37/541_group_proj/main/store-data-aggregate.csv"

REALTOR_FILE = "./realtor-data/realtor-data.csv"
RAW_REALTOR_FILE = "https://raw.githubusercontent.com/ChaseCallahan37/541_group_proj/main/realtor-data/realtor-data.csv"

RAW_COUNTIES_FILE = "./counties-data/raw-counties.csv"
COUNTIES_FILE = "./counties-data/counties.csv"

ZIP_CODES_FILE = "./zip-postal-data/zip_postal_database.csv"

CENSUS_FILE = "./zip-data/census_zip.csv"

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
    census_zip_df = read_census_zip()

    # If stores are not already populated with county, then do so and save it
    if not "county" in stores_df.columns:
        stores_df["county"] = stores_df["postal_code"].apply(lambda x: locate_county(zip_codes_df, x))
        stores_df.to_csv(path_or_buf=AGGREGATED_STORE_FILE)

    county_store_count_df = get_prepared_store_count_data(stores_df, counties_df).reset_index().set_index(["county"])
    county_store_type_df = get_prepared_store_type_data(stores_df, counties_df).reset_index().set_index(["county"])
    county_store_subtype_df = get_prepared_store_subtype_data(stores_df, counties_df).reset_index().set_index(["county"])
    fast_food_df = get_prepared_fast_food_data(stores_df, counties_df, county_store_subtype_df).reset_index().set_index(["county"])
    stores_by_zip_df = get_prepared_stores_by_zip(stores_df, census_zip_df).reset_index().set_index(["postal_code"])

    # Factors for analysis
    dependent = "median"
    store_factors = list(filter(lambda x: (x != dependent and x != "county"), county_store_count_df.columns))
    type_factors =  list(filter(lambda x: (not(x == dependent or x == "total_count")), county_store_type_df.columns))
    subtype_factors =  list(filter(lambda x: (not(x == dependent or x == "total_count")), county_store_subtype_df.columns))
    fast_food_factors = list(filter(lambda x: (not (x == dependent or x == "total_count")), fast_food_df.columns))


    menu_options = [
        {"title": "Store Count Model", "function": lambda title: store_count_model(county_store_count_df, dependent, store_factors, title),},
        {"title": "Total Store Count Model", "function": lambda title: total_store_count_model(county_store_count_df, dependent, store_factors, title),},
        {"title": "Store Percentage Model", "function": lambda title: store_percentage_makeup_model(county_store_count_df, dependent, store_factors, title),},
        {"title": "Store Type Count Model", "function": lambda title: store_type_count_model(county_store_type_df, dependent, type_factors, title),},
        {"title": "Store Type Percentage Model", "function": lambda title: store_type_percentage_makeup_model(county_store_type_df, dependent, type_factors, title),},
        {"title": "Store Subtype Count Model", "function": lambda title: store_subtype_count_model(county_store_subtype_df, dependent, subtype_factors, title),},
        {"title": "Store Subtype Percentage Model", "function": lambda title: store_subtype_percentage_makeup_model(county_store_subtype_df, dependent, subtype_factors, title),},
        {"title": "Fast Food SubType Count Model", "function": lambda title: fast_food_subtype_count_model(fast_food_df, dependent, fast_food_factors, title),},
        {"title": "Fast Food SubType Count Model", "function": lambda title: fast_food_subtype_percentage_makeup_model(fast_food_df, dependent, fast_food_factors, title)},
        {"title": "Store Count for Zip Model", "function": lambda title: store_count_model(stores_by_zip_df, dependent="median_income", store_factors=store_factors, title=title)}
    ]


    for i in range(1, (len(menu_options)+1)):
        print(f"{i}. {menu_options[i-1]['title']}")

    user_choice = menu_choice()
    while user_choice != "exit":
        chosen = menu_options[user_choice]
        chosen["function"](chosen["title"])
        press_enter()

        for i in range(1, (len(menu_options)+1)):
            print(f"{i}. {menu_options[i-1]['title']}")

        user_choice = menu_choice()

def menu_choice():
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
    
def get_prepared_store_count_data(stores_df: pd.DataFrame, counties_df:pd.DataFrame):
    # PREPARE STORE COUNT DATA
    file_name = "./cached_dfs/prepare_store_count_data.csv"
    if(path.isfile(file_name)):
        return pd.read_csv(file_name, index_col=0)
    
    stores_per_county = stores_df.groupby(["county","company_name"])[["county", "company_name"]].value_counts().to_frame().reset_index()
    county_store_count_df = stores_per_county.pivot_table(index="county", columns="company_name", values="count", fill_value=0)
    county_store_count_df["median"] = county_store_count_df.apply(lambda x: get_county_median(counties_df, x.name), axis=1)
    county_store_count_df = county_store_count_df[county_store_count_df["median"].notna()]
    
    county_store_count_df.to_csv(file_name)
    return county_store_count_df

def get_prepared_store_type_data(stores_df: pd.DataFrame, counties_df: pd.DataFrame):
    # PREPARE STORE TYPE DATA
    file_name = "./cached_dfs/prepared_store_type_data.csv"
    if(path.isfile(file_name)):
        return pd.read_csv(file_name, index_col=0)

    
    types_per_county = stores_df.groupby(["county", "type"])[["county", "type"]].value_counts().to_frame().reset_index()
    county_store_type_df = types_per_county.pivot_table(index="county", columns="type", values="count", fill_value=0)
    county_store_type_df["total_count"] = county_store_type_df.apply(lambda x: x.sum() , axis=1)
    county_store_type_df["median"] = county_store_type_df.apply(lambda x: get_county_median(counties_df, x.name), axis=1)
    county_store_type_df = county_store_type_df[county_store_type_df["median"].notna()]

    county_store_type_df.to_csv(file_name)

    return county_store_type_df

def get_prepared_store_subtype_data(stores_df: pd.DataFrame, counties_df: pd.DataFrame):
    # PREPARE STORE SUBTYPE DATA
    file_name = "./cached_dfs/prepared_store_subtype_data.csv"
    if(path.isfile(file_name)):
        return pd.read_csv(file_name, index_col=0)

    subtypes_per_county = stores_df.groupby(["county", "subtype"])[["county", "subtype"]].value_counts().to_frame().reset_index()
    county_store_subtype_df = subtypes_per_county.pivot_table(index="county", columns="subtype", values="count", fill_value=0)
    county_store_subtype_df["total_count"] = county_store_subtype_df.apply(lambda x: x.sum() , axis=1)
    county_store_subtype_df["median"] = county_store_subtype_df.apply(lambda x: get_county_median(counties_df, x.name), axis=1)
    county_store_subtype_df = county_store_subtype_df[county_store_subtype_df["median"].notna()]

    county_store_subtype_df.to_csv(file_name)
    return county_store_subtype_df

def get_prepared_fast_food_data(stores_df: pd.DataFrame, counties_df: pd.DataFrame, county_store_subtype_df: pd.DataFrame):
    # PREPARE FAST FOOD DATA
    file_name = "./cached_dfs/prepared_fast_food_data.csv"
    if(path.isfile(file_name)):
        return pd.read_csv(file_name, index_col=0)
    
    fast_food_stores = stores_df[stores_df["type"] == "fast_food"].groupby(["county", "subtype"])[["county", "subtype"]].value_counts().to_frame().reset_index()
    fast_food_df = fast_food_stores.pivot_table(index="county", columns="subtype", values="count", fill_value=0)
    fast_food_df["total_count"] = fast_food_df.apply(lambda x: x.sum(), axis=1)
    fast_food_df["median"] = county_store_subtype_df.apply(lambda x: get_county_median(counties_df, x.name), axis=1)
    fast_food_df = fast_food_df[fast_food_df["median"].notna()]

    fast_food_df.to_csv(file_name)
    return fast_food_df

def get_prepared_stores_by_zip(stores_df: pd.DataFrame, census_zip_df: pd.DataFrame):
    # PREPARE CENSUS ZIP DATA
    file_name = "./cached_dfs/prepared_store_zip_data.csv"
    if(path.isfile(file_name)):
        return pd.read_csv(file_name, index_col=0)

    stores_by_zip = stores_df.groupby(["postal_code", "company_name"])[["postal_code", "company_name"]].value_counts().to_frame().reset_index()
    stores_by_zip_df = stores_by_zip.pivot_table(index="postal_code", columns="company_name", values="count", fill_value=0)
    stores_by_zip_df["median_income"] = list(stores_by_zip_df.apply(lambda x: get_zip_median_income(x.name, census_zip_df), axis=1))
    stores_by_zip_df = stores_by_zip_df[stores_by_zip_df["median_income"].notna()]

    stores_by_zip_df.to_csv(file_name)
    return stores_by_zip_df

def store_count_model(county_store_count_df: pd.DataFrame, dependent: str, store_factors: list[str], title: str):
    display_ols_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title=title)
    display_logit_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title=title)

def total_store_count_model(county_store_count_df: pd.DataFrame, dependent: str, store_factors: list[str], title: str):
    store_count_pred_df = county_store_count_df.reset_index()[["county", "median"]]
    store_count_pred_df["store_count"] = list(county_store_count_df[store_factors].sum(axis=1).to_frame()[0])
    
    display_ols_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title=title)
    display_logit_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title=title)
    
def high_store_count_corr_coef_model(county_store_count_df: pd.DataFrame, company_corr_df, dependent: str, store_factors: list[str], title: str):
    county_store_count_df.sort_values(["median"], ascending=True, inplace=True)
    high_corr_store_factors = list(filter(lambda x:  (company_corr_df[company_corr_df['company_name'] == x]['correlation_coefficient'] > .3).all(), store_factors))

    display_ols_model(df=county_store_count_df, dependent=dependent, factors=high_corr_store_factors, title=title)
    display_logit_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title=title)


def store_percentage_makeup_model(county_store_count_df: pd.DataFrame, dependent: str, store_factors: list[str], title: str):
    county_store_count_df["store_count"] = list(county_store_count_df[store_factors].sum(axis=1).to_frame()[0])
    county_store_count_df = county_store_count_df[county_store_count_df["store_count"] > 0]

    for variable in store_factors:
        county_store_count_df[variable] = county_store_count_df.apply(lambda x: x[variable] / x["store_count"], axis=1)
    
    display_ols_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title=title)
    display_logit_model(df=county_store_count_df, dependent=dependent, factors=store_factors, title=title)

def store_type_count_model(county_store_type_df: pd.DataFrame, dependent: str, type_factors: list[str], title: str):
    display_ols_model(df=county_store_type_df, dependent=dependent, factors=type_factors, title=title)

def store_type_percentage_makeup_model(county_store_type_df: pd.DataFrame, dependent: str, type_factors: list[str], title: str):

    for variable in type_factors:
        county_store_type_df[variable] = county_store_type_df.apply(lambda x: x[variable]/x["total_count"], axis=1)
    
    display_ols_model(df=county_store_type_df, dependent=dependent, factors=type_factors, title=title)
    display_logit_model(df=county_store_type_df, dependent=dependent, factors=type_factors, title=title)


def store_subtype_count_model(county_store_subtype_df: pd.DataFrame, dependent: str, subtype_factors: list[str], title: str):
    display_ols_model(df=county_store_subtype_df, dependent=dependent, factors=subtype_factors, title=title)
    display_logit_model(df=county_store_subtype_df, dependent=dependent, factors=subtype_factors, title=title)

def store_subtype_percentage_makeup_model(county_store_subtype_df: pd.DataFrame, dependent: str, subtype_factors: list[str], title):
    for variable in subtype_factors:
        county_store_subtype_df[variable] = county_store_subtype_df.apply(lambda x: x[variable]/x["total_count"], axis=1)
    display_ols_model(df=county_store_subtype_df, dependent=dependent, factors=subtype_factors, title=title)
    display_logit_model(df=county_store_subtype_df, dependent=dependent, factors=subtype_factors, title=title)

def fast_food_subtype_count_model(fast_food_df: pd.DataFrame, dependent: str, fast_food_factors: list[str], title: str):
    display_ols_model(df=fast_food_df, dependent=dependent, factors=fast_food_factors, title=title)
    display_logit_model(df=fast_food_df, dependent=dependent, factors=fast_food_factors, title=title)

def fast_food_subtype_percentage_makeup_model(fast_food_df: pd.DataFrame, dependent: str, fast_food_factors: list[str], title: str):
    for variable in fast_food_factors:
        fast_food_df[variable] = fast_food_df.apply(lambda x: x[variable]/x["total_count"], axis=1)
    display_ols_model(df=fast_food_df, dependent=dependent, factors=fast_food_factors, title=title)
    display_logit_model(df=fast_food_df, dependent=dependent, factors=fast_food_factors, title=title)

def display_ols_model(df: pd.DataFrame, dependent: str, factors: list[str], title: str):
    index_name = df.index.name
    df.sort_values([dependent], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    ols_model = ols(formula=f"{dependent} ~ {' + '.join(factors)}", data=df).fit()
    print(ols_model.summary())

    df_corr = df[factors + [dependent]].corr(numeric_only=True)[dependent].to_frame()
    df_corr.sort_values([dependent], inplace=True, ascending=True)
    df_corr = df_corr[df_corr.index != dependent]
    print(f"{Colors.CYAN}\nCORRELATION COEFFICIENTS{Colors.RESET}")
    print(df_corr)

    
    plt.bar(x=df_corr.index, height=df_corr[dependent])
    plt.title(f"Correlation Coefficient for {title} Factors to {dependent}")
    plt.xlabel(f"{title} Factors")
    plt.xticks(rotation=50)
    plt.ylabel("Correlation Coefficient")
    plt.yticks(np.arange(-1, 1.1, .1))
    plt.show()


    df[f"pred_{dependent}"] = ols_model.predict(df[factors])

    dependent_col = df[dependent]
    pred_dependent_col = df[f"pred_{dependent}"]
    index = df.index


    plt.scatter(index, dependent_col, alpha=.6, s=2, color="blue", label=f"Actual {dependent}")
    plt.scatter(index, pred_dependent_col, alpha=.6, s=2, color="orange", label=f"Predicted {dependent}")
    plt.title(title)
    plt.xlabel(index_name.capitalize())
    plt.ylabel(f"{dependent}")
    plt.xticks([])
    plt.legend()
    plt.show()

    plt.scatter(dependent_col, pred_dependent_col, s=2, alpha=.6)
    plt.axline((0, 0), (dependent_col.max(), dependent_col.max()), color="green", label="Line of Perfect Prediction")
    plt.title(f"{title} Accuracy")
    plt.xlabel(f"{dependent} Actual")
    plt.ylabel(f"{dependent} Predicted")
    plt.legend()
    plt.show()

def display_logit_model(df: pd.DataFrame, dependent: str, factors: list[str], title: str):
    df.sort_values([dependent], ascending=True, inplace=True)

    df_corr = df[factors + [dependent]].corr(numeric_only=True)[dependent].sort_values(ascending=True).to_frame()

    positive_corr_coef_factors = list(filter(lambda x: df_corr.loc[x][dependent] > .1, factors))

    dependent_median = df[dependent].median()
    df[f"high_{dependent}"] = df[dependent].apply(lambda x: 1 if x >= dependent_median else 0)

    logit_model = logit(formula=f"high_{dependent} ~ {' + '.join(positive_corr_coef_factors)}", data=df).fit()

    df[f"prob_high_{dependent}"] = logit_model.predict(df[positive_corr_coef_factors])
    df[f"pred_high_{dependent}"] = df[f"prob_high_{dependent}"].apply(lambda x: 1 if x >= .5 else 0)

    conf_mat= confusion_matrix(df[f"high_{dependent}"], df[f"pred_high_{dependent}"])
    class_report = classification_report(df[f"high_{dependent}"], df[f"pred_high_{dependent}"])

    print(conf_mat)
    print(class_report)

    plt.scatter(df[dependent], df[f'prob_high_{dependent}'], s=1, alpha=.3, color="green", label=f"Probability of High {dependent}")
    plt.scatter(df[dependent], df[f"high_{dependent}"], s=1.2,  alpha=1, color="blue", label=f"High {dependent}")
    plt.scatter(df[dependent], df[f'pred_high_{dependent}'], s=1, alpha=.8, color="orange", label=f"Predicted High {dependent}")
    plt.legend()
    plt.title(f"Logit Prediction for {title}")
    plt.xlabel(dependent)
    plt.ylabel(f"High {dependent} Prediction")

    plt.show()


def get_zip_median_income(name, census_zip_df):
    try:
        found = census_zip_df[census_zip_df["zip_code"] == name]
        num = int(found["median_income"])
        return num
    except:
        return np.nan


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
        # If any failures occur, default to nan
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

def read_census_zip():
    orig_census_df =pd.read_csv(CENSUS_FILE)
    census_df = pd.DataFrame()
    census_df["zip_code"] = orig_census_df["NAME"].apply(lambda x: x.split(" ")[1] if pd.notnull(x) else np.nan)
    census_df["median_income"] = orig_census_df["S1902_C03_001E"].apply(convert_to_num).to_frame()
    census_df = census_df[census_df["median_income"].notnull()]
    return census_df

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
        
def convert_to_num(value):
    try:
        return int(value)
    except:
        return np.nan


main()
