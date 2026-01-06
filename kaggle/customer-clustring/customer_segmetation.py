import joblib
import json
import pandas as pd
import numpy as np


class CustomerSegmentation:
    """
    Customer segmentation model for predicting customer segments
    """

    def __init__(self, model_path="models"):
        """
        Load the trained models

        Parameters:
        -----------
        model_path : str
            Path to directory containing saved models
        """
        self.model_path = model_path

        # Load models
        self.kmeans = joblib.load(f"{model_path}/kmeans_k5_model.pkl")
        self.pca = joblib.load(f"{model_path}/pca_7_components.pkl")
        self.scaler = joblib.load(f"{model_path}/standard_scaler.pkl")

        # Load metadata
        with open(f"{model_path}/model_metadata.json", "r") as f:
            self.metadata = json.load(f)

        self.segment_names = self.metadata["segments"]
        self.feature_names = self.metadata["feature_names"]

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
        results["segment_id"] = segment_ids
        results["segment_name"] = segment_names

        return results

    def get_segment_description(self, segment_id):
        """
        Get description and recommendations for a segment
        """
        descriptions = {
            0: {
                "name": "Big Tickets",
                "description": "High-value customers with frequent large purchases",
                "marketing_strategy": "Premium rewards, VIP services, exclusive perks",
                "retention_priority": "Very High",
            },
            1: {
                "name": "Medium Tickets",
                "description": "Moderate spenders who prefer installment payments",
                "marketing_strategy": "0% interest installments, loyalty points",
                "retention_priority": "High",
            },
            2: {
                "name": "Rare Purchasers",
                "description": "Infrequent purchasers, occasional one-off payments",
                "marketing_strategy": "Re-engagement campaigns, special offers",
                "retention_priority": "Medium",
            },
            3: {
                "name": "Beginners",
                "description": "New to credit, building purchase history",
                "marketing_strategy": "Educational content, welcome bonuses",
                "retention_priority": "Medium",
            },
            4: {
                "name": "Risk",
                "description": "Minimal engagement, potential dormant accounts",
                "marketing_strategy": "Support programs, reactivation campaigns",
                "retention_priority": "Low",
            },
        }

        return descriptions.get(segment_id, {})


# Example usage
if __name__ == "__main__":
    # Load the model
    model = CustomerSegmentation(model_path="models")

    # Example customer data
    customer = {
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
        "Pay_to_MinimumPay": 2.5,
    }

    # Predict segment
    segment_id, segment_name = model.predict_segment(customer)
    print(f"\n📊 Customer Segment: {segment_name} (ID: {segment_id})")

    # Get description
    description = model.get_segment_description(segment_id)
    print(f"\n📝 Description: {description['description']}")
    print(f"🎯 Strategy: {description['marketing_strategy']}")
