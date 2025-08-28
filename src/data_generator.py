import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

def generate_synthetic_data(n_customers=1000, n_transactions=5000):
    """
    Generate synthetic consumer goods sales data for analytics demo
    """
    
    # Product categories and brands (P&G-like)
    categories = ['Personal Care', 'Beauty', 'Health Care', 'Fabric Care', 'Home Care']
    brands = {
        'Personal Care': ['CleanFresh', 'SoftTouch', 'PureEssence'],
        'Beauty': ['GlowBeauty', 'RadiantLook', 'NaturalGlow'],
        'Health Care': ['HealthPlus', 'WellCare', 'VitalLife'],
        'Fabric Care': ['FreshClean', 'SoftFabric', 'PowerWash'],
        'Home Care': ['HomeShine', 'CleanSpace', 'FreshHome']
    }
    
    # Generate customers
    customers = []
    for i in range(n_customers):
        customer = {
            'customer_id': f'CUST_{i+1:04d}',
            'age': np.random.normal(40, 15),
            'gender': np.random.choice(['M', 'F'], p=[0.45, 0.55]),
            'income': np.random.lognormal(10.5, 0.5),
            'city': fake.city(),
            'registration_date': fake.date_between(start_date='-2y', end_date='-6m')
        }
        customers.append(customer)
    
    customers_df = pd.DataFrame(customers)
    
    # Generate transactions
    transactions = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(n_transactions):
        # Select random customer
        customer_id = np.random.choice(customers_df['customer_id'])
        customer_data = customers_df[customers_df['customer_id'] == customer_id].iloc[0]
        
        # Generate transaction date with seasonal patterns
        days_offset = np.random.randint(0, 365)
        transaction_date = start_date + timedelta(days=days_offset)
        
        # Seasonal multiplier (higher sales in Nov-Dec)
        seasonal_multiplier = 1.0
        if transaction_date.month in [11, 12]:
            seasonal_multiplier = 1.5
        elif transaction_date.month in [6, 7, 8]:
            seasonal_multiplier = 1.2
        
        # Select category and brand
        category = np.random.choice(categories)
        brand = np.random.choice(brands[category])
        
        # Generate price based on category and customer income
        base_prices = {
            'Personal Care': 15,
            'Beauty': 25,
            'Health Care': 20,
            'Fabric Care': 12,
            'Home Care': 18
        }
        
        base_price = base_prices[category]
        income_factor = min(customer_data['income'] / 50000, 2.0)  # Cap at 2x
        price = base_price * income_factor * np.random.uniform(0.8, 1.3)
        
        # Quantity (influenced by seasonal patterns)
        quantity = max(1, int(np.random.poisson(2) * seasonal_multiplier))
        
        transaction = {
            'transaction_id': f'TXN_{i+1:06d}',
            'customer_id': customer_id,
            'transaction_date': transaction_date,
            'category': category,
            'brand': brand,
            'product_name': f'{brand} {category} Product',
            'quantity': quantity,
            'unit_price': round(price, 2),
            'total_amount': round(price * quantity, 2),
            'channel': np.random.choice(['Online', 'Retail', 'Pharmacy'], p=[0.4, 0.5, 0.1])
        }
        transactions.append(transaction)
    
    transactions_df = pd.DataFrame(transactions)
    
    # Merge customer and transaction data
    full_data = transactions_df.merge(customers_df, on='customer_id', how='left')
    
    return full_data

def calculate_rfm_features(data):
    """
    Calculate RFM (Recency, Frequency, Monetary) features for customer segmentation
    """
    current_date = data['transaction_date'].max()
    
    rfm = data.groupby('customer_id').agg({
        'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
        'transaction_id': 'count',  # Frequency
        'total_amount': 'sum'  # Monetary
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Add customer demographics
    customer_info = data.groupby('customer_id').agg({
        'age': 'first',
        'gender': 'first',
        'income': 'first',
        'city': 'first'
    }).reset_index()
    
    rfm_final = rfm.merge(customer_info, on='customer_id')
    
    return rfm_final

def main():
    print("Generating synthetic consumer goods data...")
    
    # Generate data
    data = generate_synthetic_data(n_customers=1000, n_transactions=5000)
    
    # Calculate RFM features
    rfm_data = calculate_rfm_features(data)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save datasets
    data.to_csv('data/synthetic_data.csv', index=False)
    rfm_data.to_csv('data/processed_data.csv', index=False)
    
    print(f"✅ Generated {len(data)} transactions for {len(rfm_data)} customers")
    print(f"📁 Data saved to:")
    print(f"   - data/synthetic_data.csv ({len(data)} rows)")
    print(f"   - data/processed_data.csv ({len(rfm_data)} rows)")
    
    # Display sample data
    print("\n📊 Sample transaction data:")
    print(data[['customer_id', 'transaction_date', 'category', 'brand', 'total_amount']].head())
    
    print("\n📈 Sample RFM data:")
    print(rfm_data[['customer_id', 'recency', 'frequency', 'monetary', 'age', 'gender']].head())

if __name__ == "__main__":
    main()