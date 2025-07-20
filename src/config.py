import os

# === Base Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# === Raw Data Paths ===
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
FRAUD_DATA_PATH = os.path.join(RAW_DATA_DIR, "Fraud_Data.csv")
CREDITCARD_DATA_PATH = os.path.join(RAW_DATA_DIR, "creditcard.csv")
IP_COUNTRY_PATH = os.path.join(RAW_DATA_DIR, "IpAddress_to_Country.csv")

# === Processed Data Paths ===
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FRAUD_CLEANED_PATH = os.path.join(PROCESSED_DATA_DIR, "fraud_cleaned.csv")
CREDITCARD_CLEANED_PATH = os.path.join(PROCESSED_DATA_DIR, "creditcard_cleaned.csv")
FRAUD_WITH_GEO_PATH = os.path.join(PROCESSED_DATA_DIR, "fraud_with_geo.csv")
