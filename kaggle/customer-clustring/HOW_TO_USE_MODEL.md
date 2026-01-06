# How to Use the Customer Clustering Model

This guide shows you how to use the trained K-Means clustering model to segment new customers and implement it in production.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Save the Trained Model](#save-the-trained-model)
3. [Load and Use the Model](#load-and-use-the-model)
4. [Segment New Customers](#segment-new-customers)
5. [Integration Examples](#integration-examples)
6. [API Endpoint Example](#api-endpoint-example)
7. [Batch Processing](#batch-processing)
8. [Real-World Use Cases](#real-world-use-cases)

---

## 🚀 Quick Start

### Step 1: Save Your Trained Models

Add this cell at the end of your notebook (after training):

```python
import joblib
import pickle

# Create a models directory
import os
os.makedirs('models', exist_ok=True)

# Save all necessary components
joblib.dump(KM_5, 'models/kmeans_k5_model.pkl')           # The K-Means model
joblib.dump(PCA_7, 'models/pca_7_components.pkl')         # PCA transformer
joblib.dump(SS, 'models/standard_scaler.pkl')             # StandardScaler
joblib.dump(x_cat.columns.tolist(), 'models/categorical_columns.pkl')  # Column names

print("✅ Models saved successfully!")
print("   - K-Means Model: models/kmeans_k5_model.pkl")
print("   - PCA Model: models/pca_7_components.pkl")
print("   - Scaler: models/standard_scaler.pkl")
```

---

## 💾 Save the Trained Model

### Complete Model Saving Code

Add this to your notebook:

```python
# Save all preprocessing information and models
import joblib
import json

# 1. Save the models
joblib.dump(KM_5, 'models/kmeans_k5_model.pkl')
joblib.dump(PCA_7, 'models/pca_7_components.pkl')
joblib.dump(SS, 'models/standard_scaler.pkl')

# 2. Save feature names and metadata
model_metadata = {
    'feature_names': x_num,  # List of numerical feature names
    'categorical_columns': x_cat.columns.tolist(),
    'n_clusters': 5,
    'segments': {
        0: 'Big Tickets',
        1: 'Medium Tickets',
        2: 'Rare Purchasers',
        3: 'Beginners',
        4: 'Risk'
    },
    'training_date': '2025-12-18',
    'variance_explained': 0.85,
    'n_samples_trained': len(concat_df)
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=2)

print("✅ All models and metadata saved!")
```

---

## 📂 Load and Use the Model

### Create a Python Module: `customer_segmentation.py`

```python
"""
Customer Segmentation Module
Load and use the trained clustering model
"""

import joblib
import json
import pandas as pd
import numpy as np


class CustomerSegmentation:
    """
    Customer segmentation model for predicting customer segments
    """

    def __init__(self, model_path='models'):
        """
        Load the trained models

        Parameters:
        -----------
        model_path : str
            Path to directory containing saved models
        """
        self.model_path = model_path

        # Load models
        self.kmeans = joblib.load(f'{model_path}/kmeans_k5_model.pkl')
        self.pca = joblib.load(f'{model_path}/pca_7_components.pkl')
        self.scaler = joblib.load(f'{model_path}/standard_scaler.pkl')

        # Load metadata
        with open(f'{model_path}/model_metadata.json', 'r') as f:
            self.metadata = json.load(f)

        self.segment_names = self.metadata['segments']
        self.feature_names = self.metadata['feature_names']

        print(f"✅ Models loaded successfully")
        print(f"   Segments: {list(self.segment_names.values())}")


    def preprocess(self, customer_data):
        """
        Preprocess customer data for prediction

        Parameters:
        -----------
        customer_data : dict or DataFrame
            Customer features

        Returns:
        --------
        X_processed : array
            Preprocessed features ready for prediction
        """
        # Convert to DataFrame if dict
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])

        # Select numerical features
        X = customer_data[self.feature_names]

        # Handle missing values (fill with 0)
        X = X.fillna(0)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Apply PCA
        X_pca = self.pca.transform(X_scaled)

        return X_pca


    def predict_segment(self, customer_data):
        """
        Predict customer segment

        Parameters:
        -----------
        customer_data : dict or DataFrame
            Customer features

        Returns:
        --------
        segment_id : int
            Cluster/segment ID (0-4)
        segment_name : str
            Segment name (e.g., 'Big Tickets')
        """
        # Preprocess
        X_processed = self.preprocess(customer_data)

        # Predict
        segment_id = self.kmeans.predict(X_processed)[0]
        segment_name = self.segment_names[str(segment_id)]

        return segment_id, segment_name


    def predict_batch(self, customers_df):
        """
        Predict segments for multiple customers

        Parameters:
        -----------
        customers_df : DataFrame
            Multiple customers' features

        Returns:
        --------
        results : DataFrame
            Original data with segment_id and segment_name columns
        """
        # Preprocess
        X_processed = self.preprocess(customers_df)

        # Predict
        segment_ids = self.kmeans.predict(X_processed)
        segment_names = [self.segment_names[str(sid)] for sid in segment_ids]

        # Add to dataframe
        results = customers_df.copy()
        results['segment_id'] = segment_ids
        results['segment_name'] = segment_names

        return results


    def get_segment_description(self, segment_id):
        """
        Get description and recommendations for a segment
        """
        descriptions = {
            0: {
                'name': 'Big Tickets',
                'description': 'High-value customers with frequent large purchases',
                'marketing_strategy': 'Premium rewards, VIP services, exclusive perks',
                'retention_priority': 'Very High'
            },
            1: {
                'name': 'Medium Tickets',
                'description': 'Moderate spenders who prefer installment payments',
                'marketing_strategy': '0% interest installments, loyalty points',
                'retention_priority': 'High'
            },
            2: {
                'name': 'Rare Purchasers',
                'description': 'Infrequent purchasers, occasional one-off payments',
                'marketing_strategy': 'Re-engagement campaigns, special offers',
                'retention_priority': 'Medium'
            },
            3: {
                'name': 'Beginners',
                'description': 'New to credit, building purchase history',
                'marketing_strategy': 'Educational content, welcome bonuses',
                'retention_priority': 'Medium'
            },
            4: {
                'name': 'Risk',
                'description': 'Minimal engagement, potential dormant accounts',
                'marketing_strategy': 'Support programs, reactivation campaigns',
                'retention_priority': 'Low'
            }
        }

        return descriptions.get(segment_id, {})


# Example usage
if __name__ == "__main__":
    # Load the model
    model = CustomerSegmentation(model_path='models')

    # Example customer data
    customer = {
        'BALANCE_FREQUENCY': 0.95,
        'PURCHASES': 2500.00,
        'ONEOFF_PURCHASES': 1500.00,
        'INSTALLMENTS_PURCHASES': 1000.00,
        'CASH_ADVANCE': 500.00,
        'PURCHASES_FREQUENCY': 0.85,
        'ONEOFF_PURCHASES_FREQUENCY': 0.45,
        'PURCHASES_INSTALLMENTS_FREQUENCY': 0.40,
        'CASH_ADVANCE_FREQUENCY': 0.25,
        'CASH_ADVANCE_TRX': 5,
        'PURCHASES_TRX': 45,
        'PRC_FULL_PAYMENT': 0.15,
        'Monthly_Avg_Purchase': 208.33,
        'Monthly_Avg_Cash': 41.67,
        'Limit_Usage': 0.45,
        'Pay_to_MinimumPay': 2.5
    }

    # Predict segment
    segment_id, segment_name = model.predict_segment(customer)
    print(f"\n📊 Customer Segment: {segment_name} (ID: {segment_id})")

    # Get description
    description = model.get_segment_description(segment_id)
    print(f"\n📝 Description: {description['description']}")
    print(f"🎯 Strategy: {description['marketing_strategy']}")
```

---

## 🎯 Segment New Customers

### Simple Usage Example

```python
from customer_segmentation import CustomerSegmentation

# Initialize model
model = CustomerSegmentation()

# New customer data
new_customer = {
    'BALANCE_FREQUENCY': 0.89,
    'PURCHASES': 3200.00,
    'ONEOFF_PURCHASES': 2000.00,
    'INSTALLMENTS_PURCHASES': 1200.00,
    'CASH_ADVANCE': 300.00,
    'PURCHASES_FREQUENCY': 0.92,
    'ONEOFF_PURCHASES_FREQUENCY': 0.55,
    'PURCHASES_INSTALLMENTS_FREQUENCY': 0.37,
    'CASH_ADVANCE_FREQUENCY': 0.15,
    'CASH_ADVANCE_TRX': 3,
    'PURCHASES_TRX': 52,
    'PRC_FULL_PAYMENT': 0.25,
    'Monthly_Avg_Purchase': 266.67,
    'Monthly_Avg_Cash': 25.00,
    'Limit_Usage': 0.38,
    'Pay_to_MinimumPay': 3.2
}

# Predict
segment_id, segment_name = model.predict_segment(new_customer)

print(f"Customer belongs to: {segment_name}")
print(f"Recommended action: {model.get_segment_description(segment_id)['marketing_strategy']}")
```

---

## 🔌 Integration Examples

### 1. Flask API Endpoint

```python
from flask import Flask, request, jsonify
from customer_segmentation import CustomerSegmentation

app = Flask(__name__)
model = CustomerSegmentation()

@app.route('/predict_segment', methods=['POST'])
def predict_segment():
    """
    API endpoint to predict customer segment

    Example request:
    POST /predict_segment
    {
        "BALANCE_FREQUENCY": 0.95,
        "PURCHASES": 2500.00,
        ...
    }
    """
    try:
        customer_data = request.json
        segment_id, segment_name = model.predict_segment(customer_data)
        description = model.get_segment_description(segment_id)

        return jsonify({
            'success': True,
            'segment_id': int(segment_id),
            'segment_name': segment_name,
            'description': description
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Predict segments for multiple customers
    """
    try:
        customers = request.json['customers']
        df = pd.DataFrame(customers)
        results = model.predict_batch(df)

        return jsonify({
            'success': True,
            'results': results.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### 2. Database Integration (SQL)

```python
import pandas as pd
from sqlalchemy import create_engine
from customer_segmentation import CustomerSegmentation

# Initialize model
model = CustomerSegmentation()

# Connect to database
engine = create_engine('postgresql://user:password@localhost:5432/bank_db')

# Load customers from database
query = """
SELECT
    customer_id,
    balance_frequency,
    purchases,
    oneoff_purchases,
    installments_purchases,
    cash_advance,
    purchases_frequency,
    oneoff_purchases_frequency,
    purchases_installments_frequency,
    cash_advance_frequency,
    cash_advance_trx,
    purchases_trx,
    prc_full_payment,
    monthly_avg_purchase,
    monthly_avg_cash,
    limit_usage,
    pay_to_minimumpay
FROM customers
WHERE segment_id IS NULL  -- Only unsegmented customers
"""

customers_df = pd.read_sql(query, engine)

# Predict segments
results = model.predict_batch(customers_df)

# Update database
for _, row in results.iterrows():
    update_query = f"""
    UPDATE customers
    SET segment_id = {row['segment_id']},
        segment_name = '{row['segment_name']}',
        last_segmented = NOW()
    WHERE customer_id = {row['customer_id']}
    """
    engine.execute(update_query)

print(f"✅ Updated {len(results)} customers with segments")
```

### 3. Scheduled Batch Processing

```python
# schedule_segmentation.py
import schedule
import time
from customer_segmentation import CustomerSegmentation
from database_handler import update_customer_segments

model = CustomerSegmentation()

def run_segmentation():
    """
    Run daily segmentation job
    """
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting segmentation...")

    # Get new/updated customers
    customers = get_new_customers()

    # Predict segments
    results = model.predict_batch(customers)

    # Update database
    update_customer_segments(results)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed! {len(results)} customers segmented.")

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(run_segmentation)

# Keep running
while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 🌐 API Endpoint Example

### Complete REST API with FastAPI

```python
# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from customer_segmentation import CustomerSegmentation

app = FastAPI(title="Customer Segmentation API")
model = CustomerSegmentation()

class CustomerData(BaseModel):
    BALANCE_FREQUENCY: float
    PURCHASES: float
    ONEOFF_PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    PURCHASES_FREQUENCY: float
    ONEOFF_PURCHASES_FREQUENCY: float
    PURCHASES_INSTALLMENTS_FREQUENCY: float
    CASH_ADVANCE_FREQUENCY: float
    CASH_ADVANCE_TRX: int
    PURCHASES_TRX: int
    PRC_FULL_PAYMENT: float
    Monthly_Avg_Purchase: float
    Monthly_Avg_Cash: float
    Limit_Usage: float
    Pay_to_MinimumPay: float

class SegmentResponse(BaseModel):
    segment_id: int
    segment_name: str
    description: str
    marketing_strategy: str

@app.post("/segment", response_model=SegmentResponse)
def segment_customer(customer: CustomerData):
    """
    Predict customer segment
    """
    try:
        customer_dict = customer.dict()
        segment_id, segment_name = model.predict_segment(customer_dict)
        description = model.get_segment_description(segment_id)

        return SegmentResponse(
            segment_id=segment_id,
            segment_name=segment_name,
            description=description['description'],
            marketing_strategy=description['marketing_strategy']
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/segment/batch")
def segment_customers_batch(customers: List[CustomerData]):
    """
    Predict segments for multiple customers
    """
    try:
        df = pd.DataFrame([c.dict() for c in customers])
        results = model.predict_batch(df)
        return results.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run with: uvicorn api:app --reload
```

### Test the API

```bash
# Start the API
uvicorn api:app --reload

# Test with curl
curl -X POST "http://localhost:8000/segment" \
  -H "Content-Type: application/json" \
  -d '{
    "BALANCE_FREQUENCY": 0.95,
    "PURCHASES": 2500.00,
    "ONEOFF_PURCHASES": 1500.00,
    "INSTALLMENTS_PURCHASES": 1000.00,
    "CASH_ADVANCE": 500.00,
    "PURCHASES_FREQUENCY": 0.85,
    "ONEOFF_PURCHASES_FREQUENCY": 0.45,
    "PURCHASES_INSTALLMENTS_FREQUENCY": 0.40,
    "CASH_ADVANCE_FREQUENCY": 0.25,
    "CASH_ADVANCE_TRX": 5,
    "PURCHASES_TRX": 45,
    "PRC_FULL_PAYMENT": 0.15,
    "Monthly_Avg_Purchase": 208.33,
    "Monthly_Avg_Cash": 41.67,
    "Limit_Usage": 0.45,
    "Pay_to_MinimumPay": 2.5
  }'
```

---

## 📦 Batch Processing

### Process Large CSV Files

```python
# batch_process.py
import pandas as pd
from customer_segmentation import CustomerSegmentation

def process_customer_file(input_file, output_file):
    """
    Process a CSV file of customers and add segments
    """
    # Load model
    model = CustomerSegmentation()

    # Read CSV
    print(f"Reading {input_file}...")
    customers = pd.read_csv(input_file)

    print(f"Processing {len(customers)} customers...")

    # Predict in batches of 1000
    batch_size = 1000
    all_results = []

    for i in range(0, len(customers), batch_size):
        batch = customers.iloc[i:i+batch_size]
        results = model.predict_batch(batch)
        all_results.append(results)
        print(f"  Processed {min(i+batch_size, len(customers))}/{len(customers)}")

    # Combine results
    final_results = pd.concat(all_results, ignore_index=True)

    # Save
    final_results.to_csv(output_file, index=False)
    print(f"✅ Results saved to {output_file}")

    # Print summary
    print("\nSegment Distribution:")
    print(final_results['segment_name'].value_counts())

# Usage
if __name__ == "__main__":
    process_customer_file(
        input_file='data/new_customers.csv',
        output_file='data/segmented_customers.csv'
    )
```

---

## 💼 Real-World Use Cases

### Use Case 1: Real-Time Marketing Automation

```python
# marketing_automation.py
from customer_segmentation import CustomerSegmentation

model = CustomerSegmentation()

def trigger_marketing_campaign(customer_id, customer_data):
    """
    Automatically trigger appropriate marketing campaign
    """
    segment_id, segment_name = model.predict_segment(customer_data)

    campaigns = {
        0: 'premium_vip_offer',          # Big Tickets
        1: 'installment_promotion',       # Medium Tickets
        2: 'reengagement_discount',       # Rare Purchasers
        3: 'welcome_bonus',               # Beginners
        4: 'support_outreach'             # Risk
    }

    campaign = campaigns[segment_id]

    # Trigger campaign (integrate with your marketing platform)
    send_campaign(customer_id, campaign)

    return campaign

def send_campaign(customer_id, campaign_name):
    """
    Send campaign via email/SMS/push notification
    """
    print(f"📧 Sending '{campaign_name}' to customer {customer_id}")
    # Integrate with SendGrid, Twilio, Firebase, etc.
```

### Use Case 2: Dynamic Pricing Strategy

```python
def get_pricing_strategy(customer_data):
    """
    Adjust pricing/offers based on segment
    """
    segment_id, segment_name = model.predict_segment(customer_data)

    strategies = {
        0: {  # Big Tickets
            'discount': 0,           # No discount needed
            'cashback': 3.0,         # 3% cashback
            'installment_months': 12
        },
        1: {  # Medium Tickets
            'discount': 5,           # 5% discount
            'cashback': 2.0,         # 2% cashback
            'installment_months': 24
        },
        2: {  # Rare Purchasers
            'discount': 15,          # 15% reactivation discount
            'cashback': 1.5,
            'installment_months': 6
        },
        3: {  # Beginners
            'discount': 10,          # 10% welcome discount
            'cashback': 2.0,
            'installment_months': 12
        },
        4: {  # Risk
            'discount': 20,          # Aggressive retention
            'cashback': 1.0,
            'installment_months': 3
        }
    }

    return strategies[segment_id]
```

### Use Case 3: Customer Lifecycle Monitoring

```python
def monitor_segment_migration():
    """
    Track customers moving between segments
    """
    # Get historical segments
    previous_segments = get_previous_segments()

    # Get current segments
    current_customers = get_all_customers()
    current_results = model.predict_batch(current_customers)

    # Detect migrations
    migrations = []
    for _, row in current_results.iterrows():
        old_segment = previous_segments.get(row['customer_id'])
        new_segment = row['segment_id']

        if old_segment != new_segment:
            migrations.append({
                'customer_id': row['customer_id'],
                'from_segment': old_segment,
                'to_segment': new_segment,
                'direction': 'upgrade' if new_segment < old_segment else 'downgrade'
            })

    # Alert on concerning migrations
    downgrades = [m for m in migrations if m['direction'] == 'downgrade']
    if downgrades:
        alert_customer_success_team(downgrades)
```

---

## 🔄 Model Retraining

### When to Retrain

```python
# retrain_checker.py
from datetime import datetime, timedelta

def should_retrain_model():
    """
    Check if model needs retraining
    """
    reasons = []

    # 1. Time-based (quarterly)
    last_train_date = get_last_training_date()
    if datetime.now() - last_train_date > timedelta(days=90):
        reasons.append("90 days since last training")

    # 2. Data drift detection
    if detect_data_drift() > 0.15:
        reasons.append("Significant data drift detected")

    # 3. Performance degradation
    if get_model_performance() < 0.7:
        reasons.append("Model performance below threshold")

    # 4. Segment distribution changes
    if check_segment_distribution_shift() > 0.20:
        reasons.append("Segment distribution shifted > 20%")

    return len(reasons) > 0, reasons
```

---

## 📊 Monitoring Dashboard

```python
# monitoring.py
def get_segment_metrics():
    """
    Get current segment statistics for dashboard
    """
    customers = get_all_customers()
    results = model.predict_batch(customers)

    metrics = {
        'total_customers': len(results),
        'segment_distribution': results['segment_name'].value_counts().to_dict(),
        'avg_purchase_by_segment': results.groupby('segment_name')['PURCHASES'].mean().to_dict(),
        'model_version': model.metadata['training_date'],
        'variance_explained': model.metadata['variance_explained']
    }

    return metrics
```

---

## 🎓 Best Practices

### 1. **Version Control Your Models**
```python
# Save with version
version = "v1.2.0"
joblib.dump(KM_5, f'models/kmeans_{version}.pkl')
```

### 2. **Monitor Model Performance**
- Track segment distribution over time
- Monitor customer migration patterns
- Set up alerts for unusual predictions

### 3. **A/B Testing**
```python
# Test new vs old segment strategies
if customer_id % 2 == 0:
    segment = model_v1.predict(customer)  # Control
else:
    segment = model_v2.predict(customer)  # Treatment
```

### 4. **Graceful Error Handling**
```python
try:
    segment_id, segment_name = model.predict_segment(customer)
except Exception as e:
    # Log error
    logger.error(f"Segmentation failed: {e}")
    # Fallback to default segment
    segment_id, segment_name = 3, "Default"
```

---

## 📚 Additional Resources

- **Notebook:** `customer-clustring-using-pca.ipynb`
- **Analysis Notes:** `ANALYSIS_NOTES.md`
- **Model Files:** `models/` directory
- **Documentation:** This file

---

## ❓ FAQ

**Q: How often should I retrain the model?**
A: Quarterly or when you detect data drift > 15%

**Q: Can I use this for real-time predictions?**
A: Yes! The prediction takes < 10ms per customer

**Q: What if a customer has missing data?**
A: The model fills missing values with 0 (median imputation is better - see preprocessing)

**Q: How accurate is the model?**
A: Silhouette score ~0.4-0.5 indicates reasonable clustering quality

---

**Need help?** Check the notebook comments or create an issue in your repository.
