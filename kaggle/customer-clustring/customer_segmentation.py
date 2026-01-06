"""
Customer Segmentation Module
============================

This module provides easy-to-use functions for customer segmentation using
the trained K-Means clustering model.

Usage:
------
    from customer_segmentation import CustomerSegmentation

    # Load model
    model = CustomerSegmentation()

    # Predict segment for one customer
    segment_id, segment_name = model.predict_segment(customer_data)

    # Predict for multiple customers
    results = model.predict_batch(customers_df)

Author: Generated from customer clustering analysis
Date: 2025-12-18
"""

import joblib
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, List, Union


class CustomerSegmentation:
    """
    Customer segmentation model for predicting customer segments
    based on credit card usage patterns
    """

    # Segment definitions
    SEGMENTS = {
        0: {
            'name': 'Big Tickets',
            'description': 'High-value customers with frequent large purchases',
            'characteristics': [
                'Very high purchase amounts',
                'Frequent purchase activity',
                'Large one-off purchases',
                'Low credit risk'
            ],
            'marketing_strategy': 'Premium rewards, VIP services, exclusive perks',
            'retention_priority': 'Very High',
            'recommended_actions': [
                'Offer premium rewards program',
                'Provide VIP customer service',
                'Invite to exclusive events',
                'Cross-sell premium products'
            ]
        },
        1: {
            'name': 'Medium Tickets',
            'description': 'Moderate spenders who prefer installment payments',
            'characteristics': [
                'Moderate purchase amounts',
                'Prefer installment payments',
                'Regular purchase frequency',
                'Medium credit utilization'
            ],
            'marketing_strategy': '0% interest installments, loyalty points',
            'retention_priority': 'High',
            'recommended_actions': [
                'Promote installment plans',
                'Offer loyalty points multipliers',
                'Provide financial education',
                'Encourage credit limit increases'
            ]
        },
        2: {
            'name': 'Rare Purchasers',
            'description': 'Infrequent purchasers, occasional one-off payments',
            'characteristics': [
                'Infrequent purchases',
                'Prefer one-off payments',
                'Lower purchase amounts',
                'Low engagement'
            ],
            'marketing_strategy': 'Re-engagement campaigns, special offers',
            'retention_priority': 'Medium',
            'recommended_actions': [
                'Send re-engagement emails',
                'Offer special discounts',
                'Run seasonal campaigns',
                'Survey to understand barriers'
            ]
        },
        3: {
            'name': 'Beginners',
            'description': 'New to credit, building purchase history',
            'characteristics': [
                'Starting to make purchases',
                'Low transaction amounts',
                'Building credit behavior',
                'High growth potential'
            ],
            'marketing_strategy': 'Educational content, welcome bonuses',
            'retention_priority': 'Medium',
            'recommended_actions': [
                'Provide welcome bonuses',
                'Share educational content',
                'Gamify credit card usage',
                'Gradual credit limit increases'
            ]
        },
        4: {
            'name': 'Risk',
            'description': 'Minimal engagement, potential dormant accounts',
            'characteristics': [
                'Very rare purchases',
                'Low purchase amounts',
                'Minimal credit card usage',
                'High risk or inactive'
            ],
            'marketing_strategy': 'Support programs, reactivation campaigns',
            'retention_priority': 'Low',
            'recommended_actions': [
                'Conduct usage surveys',
                'Offer credit counseling',
                'Send payment reminders',
                'Consider account review'
            ]
        }
    }

    def __init__(self, model_path: str = 'models'):
        """
        Initialize the segmentation model

        Parameters:
        -----------
        model_path : str
            Path to directory containing saved models
            Default: 'models'

        Raises:
        -------
        FileNotFoundError : If model files are not found
        """
        self.model_path = model_path

        # Check if models exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model directory '{model_path}' not found. "
                "Please train and save the models first."
            )

        try:
            # Load models
            self.kmeans = joblib.load(f'{model_path}/kmeans_k5_model.pkl')
            self.pca = joblib.load(f'{model_path}/pca_7_components.pkl')
            self.scaler = joblib.load(f'{model_path}/standard_scaler.pkl')

            # Load metadata
            with open(f'{model_path}/model_metadata.json', 'r') as f:
                self.metadata = json.load(f)

            self.feature_names = self.metadata['feature_names']

            print("✅ Customer Segmentation Model Loaded")
            print(f"   Training Date: {self.metadata.get('training_date', 'Unknown')}")
            print(f"   Variance Explained: {self.metadata.get('variance_explained', 'N/A')}")
            print(f"   Segments: {list(self.SEGMENTS.keys())}")

        except FileNotFoundError as e:
            print("❌ Error: Model files not found!")
            print("   Please run the notebook and save models first.")
            print(f"   Missing file: {e}")
            raise


    def _validate_features(self, data: pd.DataFrame) -> None:
        """
        Validate that all required features are present

        Parameters:
        -----------
        data : DataFrame
            Customer data to validate

        Raises:
        -------
        ValueError : If required features are missing
        """
        missing_features = [f for f in self.feature_names if f not in data.columns]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                f"Required features: {self.feature_names}"
            )


    def preprocess(self, customer_data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Preprocess customer data for prediction

        This applies the same preprocessing pipeline used during training:
        1. Feature selection
        2. Handle missing values
        3. Create dummy variables for categorical features
        4. Standardization (scaling)
        5. PCA transformation

        Parameters:
        -----------
        customer_data : dict or DataFrame
            Customer features

        Returns:
        --------
        X_processed : ndarray
            Preprocessed features ready for prediction (7 PCA components)
        """
        # Convert to DataFrame if dict
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])

        # Validate features
        self._validate_features(customer_data)

        # Select numerical features in correct order
        X_numeric = customer_data[self.feature_names].copy()

        # Handle missing values (fill with 0 - consider median imputation in production)
        X_numeric = X_numeric.fillna(0)

        # IMPORTANT: Scale ONLY numerical features (StandardScaler was trained on 14 features, not 17)
        # Convert to numpy to avoid feature name checking
        X_numeric_array = X_numeric.values
        X_numeric_scaled = self.scaler.transform(X_numeric_array)

        # Create dummy variables for categorical features if present
        # The model expects Purchase_Type to be encoded
        categorical_cols = self.metadata.get('categorical_columns', [])

        if categorical_cols and 'Purchase_Type' in customer_data.columns:
            # Create dummy variables with drop_first=True (same as training)
            X_cat = pd.get_dummies(customer_data['Purchase_Type'], drop_first=True)

            # Ensure all expected categorical columns are present (in correct order)
            X_cat_final = pd.DataFrame(index=customer_data.index)
            for col in categorical_cols:
                if col in X_cat.columns:
                    X_cat_final[col] = X_cat[col].values
                else:
                    X_cat_final[col] = 0

            X_cat_array = X_cat_final.values
        else:
            # If no Purchase_Type provided, create zero columns
            X_cat_array = np.zeros((len(X_numeric), len(categorical_cols)))

        # Combine categorical and scaled numerical features
        # Categorical columns come first, then scaled numerical (same order as training)
        X_combined = np.concatenate([X_cat_array, X_numeric_scaled], axis=1)

        # Apply PCA transformation
        X_pca = self.pca.transform(X_combined)

        return X_pca


    def predict_segment(
        self,
        customer_data: Union[Dict, pd.DataFrame],
        include_details: bool = False
    ) -> Union[Tuple[int, str], Dict]:
        """
        Predict customer segment

        Parameters:
        -----------
        customer_data : dict or DataFrame
            Customer features (must include all required features)
        include_details : bool
            If True, return full segment details instead of just ID and name

        Returns:
        --------
        If include_details=False:
            segment_id : int
                Cluster/segment ID (0-4)
            segment_name : str
                Segment name (e.g., 'Big Tickets')

        If include_details=True:
            segment_info : dict
                Complete segment information including description and recommendations

        Examples:
        ---------
        >>> model = CustomerSegmentation()
        >>> customer = {'BALANCE_FREQUENCY': 0.95, 'PURCHASES': 2500, ...}
        >>> segment_id, segment_name = model.predict_segment(customer)
        >>> print(f"Customer is in: {segment_name}")

        >>> # With details
        >>> info = model.predict_segment(customer, include_details=True)
        >>> print(info['marketing_strategy'])
        """
        # Preprocess
        X_processed = self.preprocess(customer_data)

        # Predict
        segment_id = int(self.kmeans.predict(X_processed)[0])
        segment_info = self.SEGMENTS[segment_id].copy()
        segment_info['segment_id'] = segment_id

        if include_details:
            return segment_info
        else:
            return segment_id, segment_info['name']


    def predict_batch(
        self,
        customers_df: pd.DataFrame,
        include_details: bool = False
    ) -> pd.DataFrame:
        """
        Predict segments for multiple customers at once

        Parameters:
        -----------
        customers_df : DataFrame
            Multiple customers' features
        include_details : bool
            If True, add detailed segment information to results

        Returns:
        --------
        results : DataFrame
            Original data with added columns:
            - segment_id: Numeric segment identifier (0-4)
            - segment_name: Human-readable segment name
            - (optional) description, marketing_strategy, etc.

        Examples:
        ---------
        >>> model = CustomerSegmentation()
        >>> customers = pd.read_csv('customers.csv')
        >>> results = model.predict_batch(customers)
        >>> print(results[['customer_id', 'segment_name']])
        """
        # Preprocess
        X_processed = self.preprocess(customers_df)

        # Predict
        segment_ids = self.kmeans.predict(X_processed)

        # Add to dataframe
        results = customers_df.copy()
        results['segment_id'] = segment_ids
        results['segment_name'] = [self.SEGMENTS[sid]['name'] for sid in segment_ids]

        # Add detailed information if requested
        if include_details:
            results['segment_description'] = [
                self.SEGMENTS[sid]['description'] for sid in segment_ids
            ]
            results['marketing_strategy'] = [
                self.SEGMENTS[sid]['marketing_strategy'] for sid in segment_ids
            ]
            results['retention_priority'] = [
                self.SEGMENTS[sid]['retention_priority'] for sid in segment_ids
            ]

        return results


    def get_segment_info(self, segment_id: int) -> Dict:
        """
        Get complete information about a specific segment

        Parameters:
        -----------
        segment_id : int
            Segment ID (0-4)

        Returns:
        --------
        info : dict
            Complete segment information

        Examples:
        ---------
        >>> model = CustomerSegmentation()
        >>> info = model.get_segment_info(0)
        >>> print(info['name'])
        'Big Tickets'
        >>> for action in info['recommended_actions']:
        ...     print(f"- {action}")
        """
        if segment_id not in self.SEGMENTS:
            raise ValueError(f"Invalid segment_id: {segment_id}. Must be 0-4.")

        return self.SEGMENTS[segment_id]


    def get_segment_statistics(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each segment in the dataset

        Parameters:
        -----------
        customers_df : DataFrame
            Customer data

        Returns:
        --------
        stats : DataFrame
            Statistics by segment including count, percentages, and averages

        Examples:
        ---------
        >>> model = CustomerSegmentation()
        >>> stats = model.get_segment_statistics(customers_df)
        >>> print(stats)
        """
        # Predict segments
        results = self.predict_batch(customers_df)

        # Calculate statistics
        stats = pd.DataFrame({
            'segment_name': [self.SEGMENTS[i]['name'] for i in range(5)],
            'count': [sum(results['segment_id'] == i) for i in range(5)],
        })

        stats['percentage'] = (stats['count'] / len(results) * 100).round(2)

        # Add average metrics per segment
        for col in ['PURCHASES', 'CASH_ADVANCE', 'PURCHASES_FREQUENCY']:
            if col in results.columns:
                avg_values = []
                for i in range(5):
                    segment_data = results[results['segment_id'] == i]
                    avg_values.append(segment_data[col].mean())
                stats[f'avg_{col.lower()}'] = avg_values

        return stats


    def print_segment_summary(self, segment_id: int) -> None:
        """
        Print a formatted summary of a segment

        Parameters:
        -----------
        segment_id : int
            Segment ID (0-4)
        """
        info = self.get_segment_info(segment_id)

        print(f"\n{'='*60}")
        print(f"Segment {segment_id}: {info['name']}")
        print(f"{'='*60}")
        print(f"\n📝 Description:")
        print(f"   {info['description']}")
        print(f"\n🎯 Marketing Strategy:")
        print(f"   {info['marketing_strategy']}")
        print(f"\n⭐ Retention Priority: {info['retention_priority']}")
        print(f"\n✅ Recommended Actions:")
        for action in info['recommended_actions']:
            print(f"   • {action}")
        print(f"\n{'='*60}\n")


    def get_required_features(self) -> List[str]:
        """
        Get list of required features for prediction

        Returns:
        --------
        features : list
            List of required feature names
        """
        return self.feature_names.copy()


# Example usage
if __name__ == "__main__":
    print("Customer Segmentation Model - Example Usage")
    print("=" * 60)

    try:
        # Initialize model
        model = CustomerSegmentation()

        # Example customer data
        # NOTE: Include all numerical features + Purchase_Type categorical feature
        example_customer = {
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
            'Purchase_Type': 'Both_the_Purchases'  # One of: Both_the_Purchases, Installment_Purchases, None_Of_the_Purchases, One_Of_Purchase
        }

        # Predict segment
        print("\n1. Predicting segment for example customer...")
        segment_id, segment_name = model.predict_segment(example_customer)
        print(f"   → Customer belongs to: {segment_name} (ID: {segment_id})")

        # Get detailed information
        print("\n2. Getting detailed segment information...")
        segment_info = model.predict_segment(example_customer, include_details=True)
        print(f"   → Description: {segment_info['description']}")
        print(f"   → Strategy: {segment_info['marketing_strategy']}")

        # Print full summary
        print("\n3. Full segment summary:")
        model.print_segment_summary(segment_id)

        # Show required features
        print("\n4. Required features for prediction:")
        features = model.get_required_features()
        for i, feat in enumerate(features, 1):
            print(f"   {i:2d}. {feat}")

        print("\n✅ Example completed successfully!")
        print("\nNext steps:")
        print("  - Load your customer data: pd.read_csv('customers.csv')")
        print("  - Use model.predict_batch(customers_df)")
        print("  - Integrate with your marketing automation")

    except FileNotFoundError:
        print("\n❌ Model files not found!")
        print("\nTo use this module, you need to:")
        print("  1. Run the Jupyter notebook: customer-clustring-using-pca.ipynb")
        print("  2. Train the models (run all cells)")
        print("  3. Save the models using the code in the notebook")
        print("  4. Then run this script again")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
