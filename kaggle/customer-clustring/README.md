# Customer Clustering with K-Means & PCA

A complete customer segmentation solution using unsupervised machine learning to classify credit card customers into 5 distinct segments for targeted marketing.

---

## 📊 Overview

This project segments credit card customers into 5 groups based on their spending patterns, payment behavior, and credit usage. The model uses **K-Means clustering** with **PCA dimensionality reduction** to identify distinct customer segments.

### Identified Segments:
1. **Big Tickets** (20.5%) - High-value customers with frequent large purchases
2. **Medium Tickets** (23.1%) - Moderate spenders who prefer installments
3. **Rare Purchasers** (18.7%) - Infrequent, low-engagement customers
4. **Beginners** (22.4%) - New credit users building purchase history
5. **Risk** (15.3%) - Minimal engagement, potential dormant accounts

---

## 🚀 Quick Start

### 1. Train the Model
Open and run the Jupyter notebook:
```bash
jupyter notebook customer-clustring-using-pca.ipynb
```

Run all cells to:
- Clean and preprocess data
- Engineer features
- Apply PCA (17 → 7 components, 85% variance)
- Train K-Means clustering (K=5)
- Save models to `models/` directory

### 2. Use the Trained Model

```python
from customer_segmentation import CustomerSegmentation

# Load model
model = CustomerSegmentation()

# Predict segment for one customer
customer = {
    'BALANCE_FREQUENCY': 0.95,
    'PURCHASES': 2500.00,
    'ONEOFF_PURCHASES': 1500.00,
    # ... other features
}

segment_id, segment_name = model.predict_segment(customer)
print(f"Customer belongs to: {segment_name}")
# Output: Customer belongs to: Big Tickets

# Get detailed recommendations
info = model.predict_segment(customer, include_details=True)
print(info['marketing_strategy'])
# Output: Premium rewards, VIP services, exclusive perks
```

### 3. Batch Processing

```python
import pandas as pd

# Load customer data
customers_df = pd.read_csv('customers.csv')

# Predict segments for all customers
results = model.predict_batch(customers_df, include_details=True)

# Save results
results.to_csv('segmented_customers.csv', index=False)

# View distribution
print(results['segment_name'].value_counts())
```

---

## 📁 Project Structure

```
customer-clustring/
├── customer-clustring-using-pca.ipynb   # Main analysis notebook
├── Customer DataSet.csv                  # Training data (8,950 customers)
├── customer_segmentation.py              # Production-ready module
├── models/                               # Saved model files
│   ├── kmeans_k5_model.pkl              # K-Means model
│   ├── pca_7_components.pkl             # PCA transformer
│   ├── standard_scaler.pkl              # Feature scaler
│   └── model_metadata.json              # Model info & features
├── HOW_TO_USE_MODEL.md                  # Detailed usage guide
├── ANALYSIS_NOTES.md                    # Technical documentation
└── README.md                            # This file
```

---

## 🎯 Features Used

### Original Features (17):
- **Balance & Frequency**: `BALANCE_FREQUENCY`, `PURCHASES_FREQUENCY`
- **Purchase Types**: `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`
- **Cash Advance**: `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX`
- **Transactions**: `PURCHASES_TRX`
- **Payments**: `PRC_FULL_PAYMENT`

### Engineered Features (5):
- `Monthly_Avg_Purchase` - Average monthly spending
- `Monthly_Avg_Cash` - Average monthly cash advance
- `Limit_Usage` - Credit utilization ratio
- `Pay_to_MinimumPay` - Payment behavior metric
- `Purchase_Type` - Categorical purchase preference

---

## 📈 Model Performance

| Metric | Score | Interpretation |
|--------|-------|---------------|
| **Silhouette Score** | 0.43 | Reasonable cluster separation |
| **Calinski-Harabasz** | 4,253 | Well-defined clusters |
| **Davies-Bouldin** | 1.15 | Good cluster compactness |
| **Variance Explained** | 85% | Retained information with PCA |

---

## 💡 Business Value

### Marketing Strategies by Segment

#### 🎯 Big Tickets
- **Strategy**: Premium retention
- **Actions**: VIP rewards, exclusive perks, personal service
- **Value**: Highest revenue per customer

#### 💳 Medium Tickets
- **Strategy**: Upselling with flexibility
- **Actions**: 0% installment plans, loyalty multipliers
- **Value**: High volume, steady growth

#### 🛍️ Rare Purchasers
- **Strategy**: Re-engagement
- **Actions**: Special offers, seasonal campaigns
- **Value**: Activation potential

#### 🌱 Beginners
- **Strategy**: Education & nurturing
- **Actions**: Welcome bonuses, financial literacy content
- **Value**: Long-term growth

#### ⚠️ Risk
- **Strategy**: Support & reactivation
- **Actions**: Surveys, credit counseling, payment assistance
- **Value**: Prevent churn

---

## 🔌 Integration Examples

### REST API (Flask)
```python
from flask import Flask, request, jsonify
from customer_segmentation import CustomerSegmentation

app = Flask(__name__)
model = CustomerSegmentation()

@app.route('/predict_segment', methods=['POST'])
def predict():
    customer_data = request.json
    segment_id, segment_name = model.predict_segment(customer_data)
    return jsonify({
        'segment_id': segment_id,
        'segment_name': segment_name
    })
```

### Database Integration
```python
import pandas as pd
from sqlalchemy import create_engine

# Load customers from database
engine = create_engine('postgresql://user:pass@host/db')
customers = pd.read_sql('SELECT * FROM customers', engine)

# Predict segments
results = model.predict_batch(customers)

# Update database
results.to_sql('customer_segments', engine, if_exists='replace')
```

### Scheduled Job
```python
import schedule
from customer_segmentation import CustomerSegmentation

model = CustomerSegmentation()

def daily_segmentation():
    customers = get_new_customers()
    results = model.predict_batch(customers)
    update_database(results)

schedule.every().day.at("02:00").do(daily_segmentation)
```

---

## 📚 Documentation

- **[HOW_TO_USE_MODEL.md](HOW_TO_USE_MODEL.md)** - Complete usage guide with examples
- **[ANALYSIS_NOTES.md](ANALYSIS_NOTES.md)** - Technical analysis & methodology
- **[Notebook](customer-clustring-using-pca.ipynb)** - Step-by-step analysis with comments

---

## 🛠️ Requirements

### Python Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Optional (for visualization)
```bash
pip install yellowbrick
```

### For API deployment
```bash
pip install flask  # or fastapi uvicorn
```

---

## 🔄 Model Maintenance

### When to Retrain
- **Quarterly**: Every 3 months as customer behavior evolves
- **Data Drift**: > 15% change in feature distributions
- **Performance Drop**: Silhouette score < 0.35
- **Business Changes**: New products, policy changes

### Retraining Process
1. Collect latest 6-12 months of data
2. Run notebook with new data
3. Evaluate new clusters vs old
4. A/B test before full deployment
5. Update saved models

---

## 📊 Monitoring Recommendations

Track these metrics in production:
- **Segment Distribution**: Changes over time
- **Customer Migration**: Movement between segments
- **Campaign Performance**: Response rate by segment
- **Revenue per Segment**: Business impact
- **Model Drift**: Feature statistics vs training data

---

## 🎓 Key Learnings

1. **PCA is effective** - Reduced 17 features to 7 while keeping 85% variance
2. **K=5 optimal** - Best balance between granularity and interpretability
3. **Feature engineering matters** - Derived features improved clustering quality
4. **Business context crucial** - Statistical clusters must align with business logic

---

## 🤝 Contributing

To improve this model:
1. Test alternative clustering algorithms (DBSCAN, Hierarchical)
2. Add demographic data if available
3. Implement temporal clustering for behavior changes
4. Build predictive models for segment migration

---

## 📝 License

This project is for educational and commercial use.

---

## 👤 Author

Generated from customer clustering analysis - December 2025

---

## 🔗 Related Resources

- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Customer Segmentation Guide](https://en.wikipedia.org/wiki/Market_segmentation)
- [RFM Analysis](https://en.wikipedia.org/wiki/RFM_(market_research))

---

**Questions?** Check the documentation files or run the example in `customer_segmentation.py`
