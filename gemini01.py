import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. DATA LOADING & CLEANING
# ==========================================
# Function to load data from Excel sheets while skipping the metadata rows
def load_clean_excel(file_path, sheet_name, header_row):
    # Reading Excel file, assuming header is at 'header_row' (0-indexed)
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
    # Filter out rows that might be metadata artifacts (if any remain)
    df = df.dropna(how="all")
    return df


# Load Datasets from Excel file
# Adjust header rows based on visual inspection of your Excel file
xlsx_file = "BJKT_Questions_List_r01.xlsx"
df_core = load_clean_excel(xlsx_file, sheet_name="Core", header_row=3)
df_atm = load_clean_excel(xlsx_file, sheet_name="EDC & ATM", header_row=3)
df_jom = load_clean_excel(xlsx_file, sheet_name="JOM", header_row=3)
df_acct = load_clean_excel(xlsx_file, sheet_name="DATA ACCOUNT", header_row=3)
df_cust = load_clean_excel(xlsx_file, sheet_name="DATA NASABAH", header_row=4)

# --- Clean Core Banking Data ---
# Convert Amounts to numeric
df_core["DTAMT"] = pd.to_numeric(df_core["DTAMT"], errors="coerce").fillna(0)
# Convert Date
df_core["DTDATE"] = pd.to_datetime(df_core["DTDATE"], format="%Y%m%d", errors="coerce")
# Standardize Account Number
df_core["DTACCT"] = df_core["DTACCT"].astype(str).str.replace(r"\.0$", "", regex=True)

# --- Clean Nasabah (Customer) Data ---
# Calculate Age
df_cust["birth_dt"] = pd.to_datetime(df_cust["birth_dt"], errors="coerce")
current_year = datetime.now().year
df_cust["Age"] = current_year - df_cust["birth_dt"].dt.year
# Standardize CIF
df_cust["cif_no"] = df_cust["cif_no"].astype(str)

# --- Clean Account Data ---
df_acct["acct_no"] = df_acct["acct_no"].astype(str).str.replace(r"\.0$", "", regex=True)
df_acct["cif_no"] = df_acct["cif_no"].astype(str)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

# A. Transaction Aggregates (Core Banking)
# Group by Account to get total Inflow (Credit) and Outflow (Debit)
trx_agg = (
    df_core.groupby("DTACCT")
    .agg(
        Total_Inflow=("DTAMT", lambda x: x[df_core["DTDBCR"] == "C"].sum()),
        Total_Outflow=("DTAMT", lambda x: x[df_core["DTDBCR"] == "D"].sum()),
        Trx_Count=("DTAMT", "count"),
        Avg_Trx_Size=("DTAMT", "mean"),
    )
    .reset_index()
)

# B. Channel Usage (JOM - Mobile App)
# Group by Source Account to see if they use the App
jom_agg = (
    df_jom.groupby("sourceoffund")
    .agg(Mobile_Trx_Count=("jenistrx", "count"), Mobile_Volume=("nilai", "sum"))
    .reset_index()
)
jom_agg["sourceoffund"] = (
    jom_agg["sourceoffund"].astype(str).str.replace(r"\.0$", "", regex=True)
)

# C. Merge Features into One Customer Table
# 1. Merge Account -> Customer (to link Acct No to Profile)
master_df = pd.merge(
    df_acct[["acct_no", "cif_no", "prod_nm"]], df_cust, on="cif_no", how="left"
)

# 2. Merge Transaction Aggregates
master_df = pd.merge(
    master_df, trx_agg, left_on="acct_no", right_on="DTACCT", how="left"
)

# 3. Merge Mobile Usage
master_df = pd.merge(
    master_df, jom_agg, left_on="acct_no", right_on="sourceoffund", how="left"
)

# Fill NaNs
numeric_cols = [
    "Total_Inflow",
    "Total_Outflow",
    "Trx_Count",
    "Avg_Trx_Size",
    "Mobile_Trx_Count",
    "Mobile_Volume",
]
master_df[numeric_cols] = master_df[numeric_cols].fillna(0)

# D. Create Specific "Sweeper" Features
# Retention Ratio: How much money stays in the account?
# (Inflow - Outflow) / Inflow. If close to 0 or negative, they define "Sweepers".
master_df["Retention_Rate"] = (
    master_df["Total_Inflow"] - master_df["Total_Outflow"]
) / master_df["Total_Inflow"]
master_df["Retention_Rate"] = master_df["Retention_Rate"].fillna(1)  # Handle div/0

# ==========================================
# 3. HYBRID SEGMENTATION
# ==========================================

# --- Step A: Rule-Based (The "Must Haves") ---


def assign_segment(row):
    # Rule 1: Government Aid / Kids (Based on your 'segmen' column or Age)
    if row["segmen"] == "PROGRAM PEMPROV" or row["Age"] < 18:
        return "Government Aid / Student"

    # Rule 2: Pensioners (Age Based)
    elif row["Age"] >= 60:
        return "Pensioner"

    # Default for now
    return "General Population"


master_df["Segment_Rule"] = master_df.apply(assign_segment, axis=1)

# --- Step B: Clustering (For "General Population" only) ---
# We isolate the general population to find "Sweepers" vs "Savers"
general_pop = master_df[master_df["Segment_Rule"] == "General Population"].copy()

if not general_pop.empty:
    # Select features for clustering
    # We focus on: Retention (Sweeper behavior), Mobile Usage (Tech savvy), Volume (Wealth)
    features = ["Total_Inflow", "Retention_Rate", "Mobile_Trx_Count", "Trx_Count"]

    # Handle infinite values if any
    general_pop = general_pop.replace([np.inf, -np.inf], 0)

    # Normalize Data (K-Means requires scaling)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(general_pop[features])

    # Run K-Means
    # We aim for 3 clusters: 1. Salary Sweepers, 2. Digital Spenders, 3. Passive/Low Value
    kmeans = KMeans(n_clusters=3, random_state=42)
    general_pop["Cluster_Label"] = kmeans.fit_predict(scaled_data)

    # Map Clusters to Business Names (Automated Interpretation)
    # We calculate the mean Retention Rate for each cluster to name them dynamically
    cluster_means = general_pop.groupby("Cluster_Label")[features].mean()

    # Sort clusters by Retention Rate
    # Lowest Retention = Sweepers
    sweeper_cluster = cluster_means["Retention_Rate"].idxmin()
    high_val_cluster = cluster_means["Total_Inflow"].idxmax()

    def name_cluster(label):
        if label == sweeper_cluster:
            return "Salary Sweeper (Churn Risk)"
        elif label == high_val_cluster:
            return "Prime Customer (High Volume)"
        else:
            return "General Saver"

    general_pop["Segment_Final"] = general_pop["Cluster_Label"].apply(name_cluster)

    # Merge back to main dataframe
    master_df.loc[general_pop.index, "Segment_Rule"] = general_pop["Segment_Final"]

# ==========================================
# 4. REPORTING & ANALYSIS
# ==========================================

# Summary Report
segment_summary = (
    master_df.groupby("Segment_Rule")
    .agg(
        Customer_Count=("cif_no", "nunique"),
        Avg_Age=("Age", "mean"),
        Avg_Inflow=("Total_Inflow", "mean"),
        Avg_Retention=("Retention_Rate", "mean"),
        Mobile_Adoption=("Mobile_Trx_Count", lambda x: (x > 0).mean()),  # % using App
    )
    .sort_values(by="Avg_Inflow", ascending=False)
)

print("=== CUSTOMER SEGMENTATION RESULTS ===")
print(segment_summary)

# Optional: Export for Marketing Team
# master_df[['cif_no', 'acct_no', 'Segment_Rule', 'Total_Inflow']].to_csv('Target_List_Campaign.csv')
