"""
Test script for customer segmentation model
Run this after training and saving the models
"""

def test_model():
    print("=" * 60)
    print("Testing Customer Segmentation Model")
    print("=" * 60)

    try:
        from customer_segmentation import CustomerSegmentation
        import pandas as pd

        # Initialize model
        print("\n1. Loading model...")
        model = CustomerSegmentation()
        print("   ✅ Model loaded successfully!")

        # Test data with all required features
        print("\n2. Testing with complete customer data...")
        test_customer = {
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
            'Purchase_Type': 'Both_the_Purchases'
        }

        segment_id, segment_name = model.predict_segment(test_customer)
        print(f"   ✅ Prediction successful: {segment_name} (ID: {segment_id})")

        # Test with details
        print("\n3. Testing detailed prediction...")
        info = model.predict_segment(test_customer, include_details=True)
        print(f"   ✅ Got detailed info:")
        print(f"      Description: {info['description'][:50]}...")
        print(f"      Strategy: {info['marketing_strategy'][:50]}...")

        # Test batch prediction
        print("\n4. Testing batch prediction...")
        test_batch = pd.DataFrame([test_customer] * 3)
        results = model.predict_batch(test_batch)
        print(f"   ✅ Batch prediction successful: {len(results)} customers processed")

        # Test segment statistics
        print("\n5. Testing segment statistics...")
        stats = model.get_segment_statistics(test_batch)
        print(f"   ✅ Statistics calculated for {len(stats)} segments")

        # Test without Purchase_Type (should still work)
        print("\n6. Testing without Purchase_Type (auto-fill)...")
        test_minimal = test_customer.copy()
        del test_minimal['Purchase_Type']
        segment_id2, segment_name2 = model.predict_segment(test_minimal)
        print(f"   ✅ Prediction without Purchase_Type: {segment_name2}")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour model is ready to use in production!")
        print("\nNext steps:")
        print("  • Integrate with your application")
        print("  • Load your real customer data")
        print("  • Start segmenting customers!")

        return True

    except FileNotFoundError:
        print("\n❌ ERROR: Model files not found!")
        print("\nTo fix this:")
        print("  1. Open: customer-clustring-using-pca.ipynb")
        print("  2. Run ALL cells")
        print("  3. Run the last cell to save models")
        print("  4. Then run this test again")
        return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nDebug info:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model()
    exit(0 if success else 1)
