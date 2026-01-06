# Customer Segmentation Notebook Analysis & Suggestions

## Executive Summary

This notebook performs customer segmentation on 1M+ bank transactions using RFM (Recency, Frequency, Monetary) analysis, K-Means clustering, and PCA dimensionality reduction. While the overall approach is sound, there are several critical issues and opportunities for improvement.

---

## 1. Data Collection & Understanding

### Current Approach
- Dataset: 1M+ transactions from 800K+ customers in India
- Features: Customer demographics, transaction details, account balances

### Issues Identified
1. **No data dictionary provided** - Makes it difficult to understand field meanings
2. **TransactionTime column dropped** without proper investigation
3. **Missing exploratory analysis** on the raw transaction data before aggregation

### Suggestions
- Create a data dictionary documenting all fields
- Investigate TransactionTime format before dropping (could be valuable for time-based patterns)
- Add initial transaction-level EDA before customer aggregation
- Document data source and collection period

---

## 2. Data Cleaning

### Critical Issues

#### Issue 1: Missing Data Handling
```python
df.dropna(inplace=True)
```
**Problem**: Drops all rows with ANY missing value without:
- Analyzing which columns have missing data
- Understanding why data is missing (MCAR, MAR, MNAR)
- Calculating how much data is lost
- Considering imputation strategies

**Suggestion**:
```python
# Analyze missing data first
missing_analysis = pd.DataFrame({
    'column': df.columns,
    'missing_count': df.isnull().sum(),
    'missing_pct': (df.isnull().sum() / len(df)) * 100
})
print(missing_analysis[missing_analysis.missing_count > 0])

# Then decide on strategy per column
# Consider imputation for low missing % columns
```

#### Issue 2: Gender Data Quality
```python
df.CustGender.value_counts()
df.drop(df[df['CustGender']=='T'].index,inplace=True)
```
**Problem**:
- No investigation into what 'T' represents (could be "Third Gender", data entry error, etc.)
- No documentation of how many rows were removed
- Potential bias introduction

**Suggestion**:
- Document the count and percentage of 'T' values
- Investigate if 'T' is a valid category in Indian banking context
- Consider creating an "Other" category instead of dropping

#### Issue 3: Negative Age Values
**Problem**: The notebook states "age is negative because anyone can open a Life Saver youth savings account on behalf of a child or grandchild" but:
- This doesn't explain NEGATIVE ages
- No validation that these are truly valid records
- Could indicate data quality issues

**Suggestion**:
```python
# Analyze age distribution
print(f"Negative ages: {(df['CustomerAge'] < 0).sum()}")
print(f"Age range: {df['CustomerAge'].min()} to {df['CustomerAge'].max()}")

# Consider handling:
# - If child account: use account holder's age OR create separate flag
# - If data error: investigate and clean
df['IsChildAccount'] = df['CustomerAge'] < 0
df['CustomerAge'] = df['CustomerAge'].abs()  # Or use actual child's age
```

#### Issue 4: Outlier Analysis
**Problem**: The notebook claims "there are no outliers" but doesn't properly analyze them
- BoxPlots show clear outliers in several features
- Calculated outlier percentages (up to 25%+) but ignored them

**Suggestion**:
- Be honest about outliers - they exist and are visible
- Analyze if outliers are valid extreme values or data errors
- Consider robust scaling methods or capping strategies
- Document decision to keep/remove outliers with business justification

---

## 3. Feature Engineering - RFM Analysis

### Critical Issues

#### Issue 1: RFM Calculation Logic Error
```python
MRF_df['Recency'] = MRF_df['TransactionDate2'] - MRF_df['TransactionDate1']
```
**MAJOR PROBLEM**: This calculates the **span between first and last transaction**, NOT recency!

**What Recency Should Be**: Days since the LAST transaction to NOW (or end of analysis period)

**Correct Implementation**:
```python
# Define analysis date (last date in dataset or today)
analysis_date = df['TransactionDate'].max()

# Recency = days since last transaction
MRF_df = df.groupby("CustomerID").agg({
    "TransactionDate": "max"  # Last transaction date
}).reset_index()

MRF_df['Recency'] = (analysis_date - MRF_df['TransactionDate']).dt.days
```

**Impact**: This fundamental error invalidates the entire RFM analysis and clustering results!

#### Issue 2: Converting 0 to 1 Days
```python
def rep_0(i):
    if i==0:
        return 1
```
**Problem**: Arbitrary transformation without justification. With correct recency calculation, 0 days is valid (transaction happened today).

#### Issue 3: Aggregation Strategy Issues
```python
"CustAccountBalance": "mean",
"TransactionAmount (INR)": "mean",
"CustomerAge": "median"
```

**Problems**:
1. **Inconsistent aggregation**: Why mean for some, median for others?
2. **Account Balance**: Mean balance across transactions may not represent current financial status
3. **Transaction Amount**: Mean doesn't capture spending patterns (high variance, occasional large purchases, etc.)
4. **Lost Information**: Aggregation loses temporal patterns, transaction types, etc.

**Suggestions**:
```python
MRF_df = df.groupby("CustomerID").agg({
    # Recency: Days since last transaction
    "TransactionDate": lambda x: (analysis_date - x.max()).days,

    # Frequency: Number of transactions
    "TransactionID": "count",

    # Monetary: Multiple metrics
    "TransactionAmount (INR)": ["sum", "mean", "std", "max"],

    # Account metrics
    "CustAccountBalance": ["last", "mean", "std"],  # Last = current balance

    # Demographics (static)
    "CustomerAge": "first",
    "CustGender": "first",
    "CustLocation": "first"
})

# Flatten multi-level columns
MRF_df.columns = ['_'.join(col).strip() for col in MRF_df.columns.values]
```

#### Issue 4: Transaction Date Handling
```python
"TransactionDate": "median"
```
**Problem**: Median transaction date is kept for visualization but:
- Not clear what insight this provides
- Uses valuable memory
- Could be better represented as "days active" or "transaction frequency rate"

---

## 4. Exploratory Data Analysis

### Strengths
- Good variety of visualizations
- Time series analysis included
- Distribution analysis performed

### Issues & Suggestions

#### Issue 1: Limited Statistical Analysis
**Missing**:
- Hypothesis testing
- Statistical significance tests
- Correlation strength interpretation

**Add**:
```python
# Statistical tests for differences between segments
from scipy.stats import f_oneway, chi2_contingency

# Test if age differs significantly across locations
locations = MRF_df['CustLocation'].value_counts().head(5).index
age_by_location = [MRF_df[MRF_df['CustLocation']==loc]['CustomerAge']
                   for loc in locations]
f_stat, p_value = f_oneway(*age_by_location)
print(f"Age difference across locations: F={f_stat:.2f}, p={p_value:.4f}")
```

#### Issue 2: Correlation Analysis
**Problem**: Heatmap shows very weak correlations (max ~0.2) but no discussion of implications

**Interpretation**:
- Weak correlations suggest features capture different customer aspects (GOOD for clustering)
- May indicate need for non-linear analysis
- Could suggest data quality issues if expected correlations are missing

#### Issue 3: Time Series Analysis
**Problem**:
- Only shows means over time
- Doesn't analyze trends, seasonality, or patterns
- Missing transaction volume analysis

**Suggestions**:
```python
# Decompose time series
from statsmodels.tsa.seasonal import seasonal_decompose

# Analyze transaction volume trends
daily_transactions = df.groupby('TransactionDate').size()
decomposition = seasonal_decompose(daily_transactions, model='additive', period=30)

# Plot trend, seasonal, residual components
```

---

## 5. Modeling - K-Means Clustering

### Critical Issues

#### Issue 1: Sampling Strategy
```python
df_scaled = df_scaled.sample(n=100000, random_state=42).reset_index(drop=True)
```

**Problems**:
1. **No justification** for 100K sample size
2. **Random sampling** may miss important rare customer segments
3. **No validation** that sample is representative
4. **Information loss** - 87.5% of data discarded

**Better Approaches**:
```python
# Option 1: Stratified sampling (preserve distributions)
from sklearn.model_selection import train_test_split
df_sample = df_scaled.groupby('some_category', group_keys=False).apply(
    lambda x: x.sample(frac=0.125, random_state=42)
)

# Option 2: Use Mini-Batch K-Means for full dataset
from sklearn.cluster import MiniBatchKMeans
model = MiniBatchKMeans(n_clusters=5, batch_size=10000, random_state=42)

# Option 3: Hierarchical sampling
# Sample customers, not transactions
```

#### Issue 2: Feature Scaling with Gender
```python
MRF_df['CustGender'] = MRF_df['CustGender'].map({'M':1, 'F':0})
df_scaled = StandardScaler().fit_transform(MRF_df)
```

**Problems**:
1. **Categorical variable treated as numeric** - Gender is nominal, not ordinal
2. **Scale distortion** - Binary 0/1 gets standardized differently than continuous variables
3. **Semantic loss** - Scaled gender values lose interpretability

**Correct Approach**:
```python
# Separate categorical and numerical
categorical_features = ['CustGender']
numerical_features = ['Frequency', 'CustAccountBalance',
                      'TransactionAmount (INR)', 'CustomerAge', 'Recency']

# One-hot encode categoricals
df_encoded = pd.get_dummies(MRF_df[categorical_features], prefix='Gender')

# Scale numericals
scaler = StandardScaler()
df_num_scaled = pd.DataFrame(
    scaler.fit_transform(MRF_df[numerical_features]),
    columns=numerical_features
)

# Combine
df_final = pd.concat([df_num_scaled, df_encoded], axis=1)
```

#### Issue 3: K-Means Initialization
```python
kmeans_set = {"init": "random", ...}
```

**Problem**: Random initialization is inferior to k-means++ (default in scikit-learn)

**Fix**:
```python
kmeans_set = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 42}
```

#### Issue 4: Optimal K Selection
**Current**: Uses Elbow, Silhouette, and Dendrogram

**Issues**:
- All three methods show K=5, but **silhouette scores are LOW** (~0.15-0.25)
  - Score < 0.25 indicates "weak structure"
  - Suggests data may not have natural clusters
- No consideration of business interpretability
- No stability analysis (do clusters remain stable across different samples?)

**Suggestions**:
```python
# Add more validation metrics
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

scores = {
    'k': [],
    'silhouette': [],
    'calinski_harabasz': [],
    'davies_bouldin': []
}

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_set)
    labels = kmeans.fit_predict(df_scaled)

    scores['k'].append(k)
    scores['silhouette'].append(silhouette_score(df_scaled, labels))
    scores['calinski_harabasz'].append(calinski_harabasz_score(df_scaled, labels))
    scores['davies_bouldin'].append(davies_bouldin_score(df_scaled, labels))

# Lower Davies-Bouldin is better
# Higher Calinski-Harabasz is better

# Test cluster stability
from sklearn.metrics import adjusted_rand_score
# Run clustering multiple times with different samples and compare
```

#### Issue 5: Cluster Interpretation
**Current**: Radar plot with minimal interpretation

**Missing**:
1. **Cluster profiles** - Statistical summary of each cluster
2. **Business interpretation** - What do these clusters mean for banking?
3. **Actionable insights** - How should bank treat each segment?
4. **Cluster sizes** - How many customers in each segment?
5. **Cluster stability** - Are clusters robust?

**Add**:
```python
# Detailed cluster profiles
df_scaled['Label'] = kmeans.labels_

for cluster in range(5):
    print(f"\n{'='*60}")
    print(f"CLUSTER {cluster}")
    print(f"{'='*60}")
    cluster_data = df_scaled[df_scaled['Label'] == cluster]

    print(f"Size: {len(cluster_data)} ({len(cluster_data)/len(df_scaled)*100:.1f}%)")
    print("\nProfile:")
    print(cluster_data.describe())

    print("\nTop Characteristics:")
    # Compare cluster mean to global mean
    for col in numerical_features:
        cluster_mean = cluster_data[col].mean()
        global_mean = df_scaled[col].mean()
        diff_pct = ((cluster_mean - global_mean) / global_mean) * 100
        if abs(diff_pct) > 20:  # Significant difference
            print(f"  - {col}: {diff_pct:+.1f}% vs average")

# Business interpretation mapping
cluster_names = {
    0: "High-Value Active Customers",
    1: "Frequent Small Transactors",
    2: "Dormant/Inactive Accounts",
    3: "Occasional Large Spenders",
    4: "New/Young Customers"
}

# Actionable recommendations
recommendations = {
    0: "Priority service, exclusive offers, retention focus",
    1: "Encourage higher-value transactions, upsell opportunities",
    2: "Reactivation campaigns, win-back offers",
    3: "Targeted marketing for specific needs",
    4: "Onboarding programs, educational content"
}
```

---

## 6. PCA Analysis

### Issues

#### Issue 1: Component Selection
```python
pca = PCA(n_components=4)
```

**Problems**:
1. **Only 6 features total** - PCA on 6 features may not be very effective
2. **Graph shows ~90% variance at 4 components** - Good, but not discussed
3. **No interpretation of principal components** - What do PC1, PC2, etc. represent?

**Suggestions**:
```python
# Analyze component loadings
pca = PCA(n_components=4)
pca_data = pca.fit_transform(df_scaled.iloc[:,:-1])

# Create loadings dataframe
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2', 'PC3', 'PC4'],
    index=df_scaled.columns[:-1]
)

print("PCA Component Loadings:")
print(loadings)

# Visualize loadings
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)

# Interpret components
print(f"\nPC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% variance")
print(f"PC1 is primarily driven by: {loadings['PC1'].abs().nlargest(3)}")
```

#### Issue 2: Clustering After PCA
**Current**: Applies K-Means to PCA-reduced data

**Questions Not Addressed**:
1. **Why use PCA if only 6 features?** - Dimensionality reduction benefit is minimal
2. **Information loss** - 4 components vs 6 features (lost 2 dimensions)
3. **Interpretability loss** - PC components are harder to interpret than original features
4. **No comparison** - Which approach gives better business insights?

**Better Approach**:
```python
# Compare both approaches
results_comparison = {
    'Method': ['K-Means Original', 'K-Means + PCA'],
    'Features': [6, 4],
    'Silhouette': [original_silhouette, pca_silhouette],
    'Interpretability': ['High', 'Low'],
    'Computation': ['Medium', 'Fast']
}

# Choose based on trade-offs and business needs
```

#### Issue 3: Same K Selection Process
**Problem**: Repeats entire Elbow/Silhouette/Dendrogram process for PCA data
- This is correct methodology
- But no discussion of whether optimal K changed
- No comparison of cluster quality between methods

---

## 7. Overall Methodology Issues

### Issue 1: No Train/Test Split or Validation
**Critical Problem**: Entire dataset used for clustering with no validation

**Why This Matters**:
- Cannot assess if clusters generalize to new customers
- No way to validate cluster stability
- Overfitting risk (though less critical for unsupervised learning)

**Solution**:
```python
# Split data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df_scaled, test_size=0.2, random_state=42)

# Fit on train
kmeans = KMeans(n_clusters=5, **kmeans_set)
kmeans.fit(train_data)

# Predict on test
test_labels = kmeans.predict(test_data)

# Validate cluster quality on test set
test_silhouette = silhouette_score(test_data, test_labels)
print(f"Test set silhouette score: {test_silhouette:.3f}")
```

### Issue 2: No Model Persistence
**Missing**: No code to save the trained model for production use

**Add**:
```python
import joblib

# Save model and preprocessing objects
joblib.dump(kmeans, 'models/kmeans_customer_segmentation.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(pca, 'models/pca.pkl')

# Save cluster profiles for reference
cluster_profiles = df_scaled.groupby('Label').describe()
cluster_profiles.to_csv('models/cluster_profiles.csv')
```

### Issue 3: No Business Metrics
**Missing**:
- Customer Lifetime Value (CLV) analysis per cluster
- Revenue contribution per segment
- Cost to serve analysis
- Churn risk per segment
- Marketing campaign effectiveness

**Add**:
```python
# Calculate business metrics per cluster
business_metrics = df_scaled.groupby('Label').agg({
    'TransactionAmount (INR)': ['sum', 'mean'],
    'Frequency': 'mean',
    'CustAccountBalance': 'mean'
})

# Calculate segment value
business_metrics['Total_Revenue'] = (
    business_metrics['TransactionAmount (INR)']['sum']
)
business_metrics['Avg_CLV'] = (
    business_metrics['TransactionAmount (INR)']['mean'] *
    business_metrics['Frequency']['mean'] * 12  # Annualized
)

# Rank segments by value
business_metrics.sort_values('Total_Revenue', ascending=False)
```

---

## 8. Code Quality Issues

### Issue 1: Hardcoded Values
```python
df_scaled = df_scaled.sample(n=100000, random_state=42)
```
**Problem**: Magic numbers without constants or configuration

**Fix**:
```python
# Configuration section at top
CONFIG = {
    'SAMPLE_SIZE': 100000,
    'RANDOM_STATE': 42,
    'N_CLUSTERS': 5,
    'PCA_COMPONENTS': 4,
    'PCA_VARIANCE_THRESHOLD': 0.90
}

# Use throughout
df_scaled = df_scaled.sample(n=CONFIG['SAMPLE_SIZE'],
                              random_state=CONFIG['RANDOM_STATE'])
```

### Issue 2: Inconsistent Naming
- Sometimes `df`, sometimes `MRF_df`, sometimes `df_scaled`
- `pca_data` vs `pca_df`
- Inconsistent column naming conventions

**Fix**: Establish naming convention
```python
# Raw data
df_raw = pd.read_csv(...)

# Customer-level aggregated data
df_customer = ...  # Instead of MRF_df

# Scaled data
df_customer_scaled = ...

# After clustering
df_customer_clustered = ...
```

### Issue 3: Lack of Functions
**Problem**: All code in notebook cells - difficult to reuse and test

**Improvement**:
```python
def calculate_rfm(transactions_df, analysis_date=None):
    """Calculate RFM metrics from transaction data."""
    if analysis_date is None:
        analysis_date = transactions_df['TransactionDate'].max()

    rfm = transactions_df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (analysis_date - x.max()).days,
        'TransactionID': 'count',
        'TransactionAmount (INR)': 'sum'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def perform_clustering(data, n_clusters, random_state=42):
    """Perform K-Means clustering with validation."""
    # ... clustering logic
    return model, labels, metrics

# Use functions
rfm_df = calculate_rfm(df)
model, labels, metrics = perform_clustering(df_scaled, n_clusters=5)
```

---

## 9. Missing Analyses (Original Notebook Goals)

The notebook states 5 goals but only accomplishes #1 (basic clustering):

### Goal 2: Location-wise Analysis ❌
**Missing**:
- Regional trend analysis
- Geographic clustering
- State/city-level insights
- Urban vs rural patterns

**Suggestion**: Add section
```python
# Regional analysis
regional_metrics = df.groupby('CustLocation').agg({
    'TransactionAmount (INR)': ['mean', 'sum', 'count'],
    'CustAccountBalance': 'mean',
    'CustomerAge': 'median'
})

# Top 10 cities by transaction volume
top_cities = regional_metrics.sort_values(
    ('TransactionAmount (INR)', 'sum'),
    ascending=False
).head(10)

# Geographic visualization
import plotly.express as px
fig = px.choropleth(
    regional_data,
    locations='CustLocation',
    color='Total_Transactions',
    title='Transaction Distribution Across India'
)
```

### Goal 3: Transaction-Related Analysis ❌
**Missing**:
- Transaction type analysis
- Time-based patterns (hourly, daily, monthly)
- Transaction amount distributions
- Payment method analysis

### Goal 4: RFM Analysis ⚠️
**Attempted but FLAWED** - See Section 3

### Goal 5: Network/Graph Analysis ❌
**Missing**: Completely absent

**Suggestion**: Add if data supports it
```python
import networkx as nx

# Example: Customer-Merchant network
# Or: Customer co-occurrence patterns
# Requires additional data or feature engineering
```

---

## 10. Recommendations for Improvement

### Immediate Fixes (High Priority)
1. **Fix RFM calculation** - Recency is fundamentally wrong
2. **Address missing data properly** - Analyze before dropping
3. **Handle categorical variables correctly** - Don't scale binary features
4. **Add cluster interpretation** - Make results actionable
5. **Document data quality issues** - Negative ages, gender 'T', etc.

### Methodology Improvements (Medium Priority)
6. **Implement proper sampling strategy** - Or use full dataset with Mini-Batch K-Means
7. **Add validation framework** - Train/test split for clusters
8. **Expand evaluation metrics** - Beyond silhouette
9. **Analyze cluster stability** - Multiple runs, different samples
10. **Compare clustering methods** - Try DBSCAN, Hierarchical, Gaussian Mixture

### Advanced Enhancements (Low Priority but Valuable)
11. **Add temporal analysis** - Customer lifecycle, trend analysis
12. **Implement all 5 stated goals** - Complete the promised analyses
13. **Include predictive modeling** - Churn prediction, CLV forecasting
14. **Add business metrics** - Revenue per segment, cost to serve
15. **Create deployment pipeline** - Save models, create prediction function

### Code Quality
16. **Refactor into functions** - Improve reusability
17. **Add configuration file** - Externalize parameters
18. **Include unit tests** - Validate key functions
19. **Add detailed comments** - Explain business logic
20. **Create requirements.txt** - Document dependencies

---

## 11. Suggested Notebook Structure

### Improved Organization
```
1. Executive Summary & Business Context
   - Problem statement
   - Success criteria
   - Expected outcomes

2. Data Understanding
   - Data dictionary
   - Initial exploration
   - Data quality report

3. Data Preparation
   - Missing value analysis & strategy
   - Outlier detection & treatment
   - Feature engineering with justification
   - Data validation

4. Exploratory Data Analysis
   - Univariate analysis
   - Bivariate analysis
   - Temporal patterns
   - Geographic patterns
   - Statistical testing

5. Feature Engineering
   - RFM calculation (CORRECT implementation)
   - Additional behavioral features
   - Feature selection
   - Feature scaling strategy

6. Modeling
   - Model selection rationale
   - Hyperparameter tuning
   - Multiple algorithm comparison
   - Validation strategy

7. Cluster Analysis
   - Cluster profiles
   - Business interpretation
   - Stability analysis
   - Comparison with PCA approach

8. Business Insights & Recommendations
   - Segment definitions
   - Value proposition per segment
   - Marketing strategies
   - Risk analysis

9. Model Deployment
   - Model persistence
   - Prediction pipeline
   - Monitoring strategy

10. Conclusions & Next Steps
    - Key findings
    - Limitations
    - Future work
```

---

## 12. Sample Code: Complete Improved Pipeline

```python
# Configuration
CONFIG = {
    'DATA_PATH': '/kaggle/input/bank-customer-segmentation/bank_transactions.csv',
    'SAMPLE_SIZE': 100000,  # Or None for full dataset with MiniBatch
    'RANDOM_STATE': 42,
    'N_CLUSTERS_RANGE': range(2, 11),
    'PCA_VARIANCE_THRESHOLD': 0.90
}

# 1. Load and Clean Data
def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Convert dates
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['CustomerDOB'] = pd.to_datetime(df['CustomerDOB'])

    # Calculate age correctly
    df['CustomerAge'] = (df['TransactionDate'] - df['CustomerDOB']).dt.days / 365.25

    # Handle gender
    valid_genders = ['M', 'F']
    df = df[df['CustGender'].isin(valid_genders)]

    # Handle missing values strategically
    df = df.dropna(subset=['TransactionAmount (INR)', 'TransactionDate'])
    # Impute other columns if needed

    return df

# 2. Calculate RFM (CORRECT VERSION)
def calculate_rfm_correct(df, analysis_date=None):
    if analysis_date is None:
        analysis_date = df['TransactionDate'].max()

    rfm = df.groupby('CustomerID').agg({
        'TransactionDate': lambda x: (analysis_date - x.max()).days,  # CORRECT Recency
        'TransactionID': 'count',  # Frequency
        'TransactionAmount (INR)': 'sum',  # Monetary
        'CustAccountBalance': 'last',  # Current balance
        'CustomerAge': 'first',
        'CustGender': 'first',
        'CustLocation': 'first'
    })

    rfm.columns = ['Recency', 'Frequency', 'Monetary',
                   'AccountBalance', 'Age', 'Gender', 'Location']

    return rfm.reset_index()

# 3. Prepare Features for Clustering
def prepare_features(rfm_df):
    # Separate numerical and categorical
    numerical_features = ['Recency', 'Frequency', 'Monetary', 'AccountBalance', 'Age']
    categorical_features = ['Gender']

    # One-hot encode categoricals
    df_cat = pd.get_dummies(rfm_df[categorical_features], drop_first=True)

    # Scale numericals
    scaler = StandardScaler()
    df_num = pd.DataFrame(
        scaler.fit_transform(rfm_df[numerical_features]),
        columns=numerical_features,
        index=rfm_df.index
    )

    # Combine
    df_final = pd.concat([df_num, df_cat], axis=1)

    return df_final, scaler

# 4. Find Optimal Clusters
def find_optimal_clusters(X, k_range):
    metrics = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        n_init=10, random_state=CONFIG['RANDOM_STATE'])
        labels = kmeans.fit_predict(X)

        metrics.append({
            'k': k,
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X, labels),
            'calinski': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels)
        })

    return pd.DataFrame(metrics)

# 5. Interpret Clusters
def interpret_clusters(df_original, labels, feature_names):
    df_clustered = df_original.copy()
    df_clustered['Cluster'] = labels

    interpretations = {}

    for cluster in sorted(df_clustered['Cluster'].unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
        global_data = df_clustered

        profile = {}
        profile['size'] = len(cluster_data)
        profile['percentage'] = len(cluster_data) / len(df_clustered) * 100

        # Key characteristics
        characteristics = []
        for feature in feature_names:
            cluster_mean = cluster_data[feature].mean()
            global_mean = global_data[feature].mean()
            diff_pct = ((cluster_mean - global_mean) / global_mean) * 100

            if abs(diff_pct) > 20:
                characteristics.append({
                    'feature': feature,
                    'cluster_mean': cluster_mean,
                    'global_mean': global_mean,
                    'diff_pct': diff_pct
                })

        profile['characteristics'] = characteristics
        interpretations[f'Cluster_{cluster}'] = profile

    return interpretations

# Main execution
if __name__ == "__main__":
    # Load
    df = load_and_clean_data(CONFIG['DATA_PATH'])

    # Calculate RFM
    rfm_df = calculate_rfm_correct(df)

    # Prepare features
    X, scaler = prepare_features(rfm_df)

    # Find optimal K
    metrics_df = find_optimal_clusters(X, CONFIG['N_CLUSTERS_RANGE'])
    print(metrics_df)

    # Fit final model
    optimal_k = 5  # Based on metrics
    final_model = KMeans(n_clusters=optimal_k, init='k-means++',
                         n_init=10, random_state=CONFIG['RANDOM_STATE'])
    labels = final_model.fit_predict(X)

    # Interpret
    interpretations = interpret_clusters(rfm_df, labels, X.columns.tolist())

    # Save
    joblib.dump(final_model, 'models/kmeans_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
```

---

## 13. Final Assessment

### Strengths
- Attempts industry-standard RFM analysis
- Uses multiple validation techniques (Elbow, Silhouette, Dendrogram)
- Includes both K-Means and PCA approaches
- Good visualization variety

### Critical Flaws
1. **Incorrect RFM calculation** - Invalidates core methodology
2. **Poor data cleaning practices** - Drops data without analysis
3. **Improper feature handling** - Scales categorical variables
4. **Weak cluster quality** - Low silhouette scores ignored
5. **Missing business context** - No actionable insights

### Overall Score: 4/10

**Rationale**:
- Fundamental methodology error (RFM) is critical
- Code runs but produces questionable results
- Missing most stated objectives (3 out of 5 goals not addressed)
- Lacks production readiness and business value

### Recommended Actions
1. **Immediate**: Fix RFM calculation and re-run entire analysis
2. **Short-term**: Address data quality and feature engineering issues
3. **Medium-term**: Add business interpretation and validation
4. **Long-term**: Refactor for production deployment

---

## 14. Learning Resources

### For RFM Analysis
- [DataCamp: Customer Segmentation in Python](https://www.datacamp.com/tutorial/introduction-customer-segmentation-python)
- [RFM Analysis Guide](https://www.putler.com/rfm-analysis/)

### For Clustering Evaluation
- [Scikit-learn Clustering Metrics](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- [Interpreting Silhouette Scores](https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c)

### For Customer Segmentation Best Practices
- [McKinsey: Customer Segmentation](https://www.mckinsey.com/capabilities/growth-marketing-and-sales/our-insights)
- [Harvard Business Review: Customer Analytics](https://hbr.org/topic/customer-analytics)

---

**Document prepared by**: AI Analysis
**Date**: 2025-12-18
**Version**: 1.0
