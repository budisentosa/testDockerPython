# Quick Start Guide - Customer Segmentation

## ⚠️ Important: Feature Requirements

The model requires **17 features** total:
- **14 numerical features** (spending metrics)
- **3 categorical dummy features** (from Purchase_Type)

---

## 📋 Required Features

### Numerical Features (14):
```python
{
    'BALANCE_FREQUENCY': float,           # 0-1 (how often balance is updated)
    'PURCHASES': float,                   # Total purchase amount
    'ONEOFF_PURCHASES': float,           # One-time purchase amount
    'INSTALLMENTS_PURCHASES': float,     # Installment purchase amount
    'CASH_ADVANCE': float,               # Cash advance amount
    'PURCHASES_FREQUENCY': float,        # 0-1 (purchase frequency)
    'ONEOFF_PURCHASES_FREQUENCY': float, # 0-1 (one-off frequency)
    'PURCHASES_INSTALLMENTS_FREQUENCY': float,  # 0-1 (installment frequency)
    'CASH_ADVANCE_FREQUENCY': float,     # 0-1 (cash advance frequency)
    'CASH_ADVANCE_TRX': int,             # Number of cash advance transactions
    'PURCHASES_TRX': int,                # Number of purchase transactions
    'PRC_FULL_PAYMENT': float,           # 0-1 (percent of full payment)
    'Monthly_Avg_Purchase': float,       # Average monthly purchase
    'Monthly_Avg_Cash': float,           # Average monthly cash advance
    'Limit_Usage': float,                # Credit utilization (0-1)
    'Pay_to_MinimumPay': float,          # Payment ratio
}
```

### Categorical Feature (1):
```python
'Purchase_Type': str  # One of:
                      # - 'Both_the_Purchases'
                      # - 'Installment_Purchases'
                      # - 'None_Of_the_Purchases'
                      # - 'One_Of_Purchase'
```

---

## 🚀 Usage Examples

### Example 1: Predict Single Customer

```python
from customer_segmentation import CustomerSegmentation

# Load model
model = CustomerSegmentation()

# Customer data (all 14 numerical + 1 categorical = 17 features after encoding)
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
    'Pay_to_MinimumPay': 2.5,
    'Purchase_Type': 'Both_the_Purchases'  # IMPORTANT: Include this!
}

# Predict
segment_id, segment_name = model.predict_segment(customer)
print(f"Customer segment: {segment_name}")
# Output: Customer segment: Big Tickets
```

---

### Example 2: Batch Processing

```python
import pandas as pd

# Load customer data
customers_df = pd.read_csv('customers.csv')

# Make sure Purchase_Type column exists
# If you need to create it:
customers_df['Purchase_Type'] = np.where(
    (customers_df['ONEOFF_PURCHASES'] == 0) & (customers_df['INSTALLMENTS_PURCHASES'] == 0),
    'None_Of_the_Purchases',
    np.where(
        (customers_df['ONEOFF_PURCHASES'] > 0) & (customers_df['INSTALLMENTS_PURCHASES'] == 0),
        'One_Of_Purchase',
        np.where(
            (customers_df['ONEOFF_PURCHASES'] == 0) & (customers_df['INSTALLMENTS_PURCHASES'] > 0),
            'Installment_Purchases',
            'Both_the_Purchases'
        )
    )
)

# Predict segments
model = CustomerSegmentation()
results = model.predict_batch(customers_df)

# View results
print(results[['segment_name', 'Purchase_Type']].head())
```

---

### Example 3: Without Purchase_Type (Auto-fills with zeros)

```python
# If you don't have Purchase_Type, the model will auto-fill categorical features with 0
customer_minimal = {
    'BALANCE_FREQUENCY': 0.95,
    'PURCHASES': 2500.00,
    # ... all 14 numerical features ...
    'Pay_to_MinimumPay': 2.5
    # Purchase_Type omitted - will be filled with zeros
}

segment_id, segment_name = model.predict_segment(customer_minimal)
```

**Note:** Predictions without Purchase_Type may be less accurate!

---

## 🔍 Troubleshooting

### Error: "X has 14 features, but PCA is expecting 17 features"

**Cause:** Missing the categorical Purchase_Type feature

**Solution:** Add the Purchase_Type feature to your data:

```python
# Calculate Purchase_Type from ONEOFF_PURCHASES and INSTALLMENTS_PURCHASES
def calculate_purchase_type(oneoff, installments):
    if oneoff == 0 and installments == 0:
        return 'None_Of_the_Purchases'
    elif oneoff > 0 and installments == 0:
        return 'One_Of_Purchase'
    elif oneoff == 0 and installments > 0:
        return 'Installment_Purchases'
    else:
        return 'Both_the_Purchases'

customer['Purchase_Type'] = calculate_purchase_type(
    customer['ONEOFF_PURCHASES'],
    customer['INSTALLMENTS_PURCHASES']
)
```

---

### Error: "Missing required features"

**Cause:** One or more of the 14 numerical features is missing

**Solution:** Ensure all 14 numerical features are present:

```python
# Check required features
model = CustomerSegmentation()
required_features = model.get_required_features()
print("Required features:", required_features)

# Check your data
missing = [f for f in required_features if f not in customer.keys()]
print("Missing features:", missing)
```

---

## 📊 Feature Calculation Guide

If you have raw data, here's how to calculate the derived features:

### Monthly_Avg_Purchase
```python
Monthly_Avg_Purchase = PURCHASES / TENURE
```

### Monthly_Avg_Cash
```python
Monthly_Avg_Cash = CASH_ADVANCE / TENURE
```

### Limit_Usage (Credit Utilization)
```python
Limit_Usage = BALANCE / CREDIT_LIMIT
```

### Pay_to_MinimumPay
```python
Pay_to_MinimumPay = PAYMENTS / MINIMUM_PAYMENTS
```

### Purchase_Type
```python
if (ONEOFF_PURCHASES == 0) and (INSTALLMENTS_PURCHASES == 0):
    Purchase_Type = 'None_Of_the_Purchases'
elif (ONEOFF_PURCHASES > 0) and (INSTALLMENTS_PURCHASES == 0):
    Purchase_Type = 'One_Of_Purchase'
elif (ONEOFF_PURCHASES == 0) and (INSTALLMENTS_PURCHASES > 0):
    Purchase_Type = 'Installment_Purchases'
else:
    Purchase_Type = 'Both_the_Purchases'
```

---

## 🎯 Complete Working Example

```python
from customer_segmentation import CustomerSegmentation
import pandas as pd

# Initialize model
model = CustomerSegmentation()

# Example customer with all required features
customer = {
    # Basic metrics
    'BALANCE_FREQUENCY': 0.95,
    'PURCHASES': 2500.00,
    'ONEOFF_PURCHASES': 1500.00,
    'INSTALLMENTS_PURCHASES': 1000.00,
    'CASH_ADVANCE': 500.00,

    # Frequency metrics (0-1 range)
    'PURCHASES_FREQUENCY': 0.85,
    'ONEOFF_PURCHASES_FREQUENCY': 0.45,
    'PURCHASES_INSTALLMENTS_FREQUENCY': 0.40,
    'CASH_ADVANCE_FREQUENCY': 0.25,

    # Transaction counts
    'CASH_ADVANCE_TRX': 5,
    'PURCHASES_TRX': 45,

    # Payment metrics
    'PRC_FULL_PAYMENT': 0.15,

    # Derived features
    'Monthly_Avg_Purchase': 208.33,
    'Monthly_Avg_Cash': 41.67,
    'Limit_Usage': 0.45,
    'Pay_to_MinimumPay': 2.5,

    # Categorical feature (IMPORTANT!)
    'Purchase_Type': 'Both_the_Purchases'
}

# Predict segment
segment_id, segment_name = model.predict_segment(customer)
print(f"\n✅ Customer segment: {segment_name}")

# Get detailed recommendations
info = model.predict_segment(customer, include_details=True)
print(f"\n📝 Description: {info['description']}")
print(f"\n🎯 Marketing Strategy: {info['marketing_strategy']}")
print(f"\n✅ Recommended Actions:")
for action in info['recommended_actions']:
    print(f"   • {action}")
```

---

## 📚 Next Steps

1. **Run the model**: `python customer_segmentation.py`
2. **Read full guide**: `HOW_TO_USE_MODEL.md`
3. **Check technical details**: `ANALYSIS_NOTES.md`

---

## ✅ Checklist Before Running

- [ ] All 14 numerical features are present
- [ ] Purchase_Type feature is included
- [ ] Feature values are in correct ranges (frequencies: 0-1, etc.)
- [ ] No NaN or null values (model fills with 0)
- [ ] Models are saved in `models/` directory

---

**Having issues?** Make sure you:
1. Ran the notebook and saved models
2. Have all 17 features (14 numerical + Purchase_Type → 3 dummy variables)
3. Purchase_Type values match expected categories
