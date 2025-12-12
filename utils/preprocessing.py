"""
Feature engineering for real-time inference.
Mirrors the transformations in RetailBuyerSegmentation.ipynb
to ensure consistency between training and prediction.

Column names match the original data.csv format (snake_case).
"""

import pandas as pd
import numpy as np
from datetime import datetime

def create_features(data):
    """
    Create all engineered features from the input data.
    Matches the feature engineering from RetailBuyerSegmentation.ipynb
    
    Args:
        data (pd.DataFrame): Raw customer data with original columns
        
    Returns:
        pd.DataFrame: Data with all engineered features added
    """
    df = data.copy()
    
    # ===========================================
    # 1. TIME-BASED FEATURES
    # ===========================================
    df['customer_age'] = 2014 - df['birth_year']
    
    # Parse signup_date if it exists (for batch upload)
    # For single predictions, we'll use a default value
    if 'signup_date' in df.columns:
        # Handle mixed date formats from the original data
        df['signup_date_parsed'] = pd.to_datetime(
            df['signup_date'], 
            format='mixed', 
            dayfirst=False,
            errors='coerce'
        )
        # Calculate tenure days from signup to reference date (2014-12-31)
        reference_date = pd.Timestamp('2014-12-31')
        df['customer_tenure_days'] = (reference_date - df['signup_date_parsed']).dt.days
    else:
        # Default value for single customer prediction (median from training data)
        df['customer_tenure_days'] = 537  # Median from original data
    
    # ===========================================
    # 2. SPENDING AGGREGATIONS
    # ===========================================
    spending_cols = ['spend_wine', 'spend_fruits', 'spend_meat', 
                     'spend_fish', 'spend_sweets', 'spend_gold']
    
    df['total_spend'] = df[spending_cols].sum(axis=1)
    df['avg_spend_per_category'] = df[spending_cols].mean(axis=1)
    
    # Ratios (add 1 to avoid division by zero)
    df['spend_wine_ratio'] = df['spend_wine'] / (df['total_spend'] + 1)
    df['spend_meat_ratio'] = df['spend_meat'] / (df['total_spend'] + 1)
    
    # ===========================================
    # 3. CHANNEL PREFERENCES
    # ===========================================
    channel_cols = ['num_web_purchases', 'num_catalog_purchases', 'num_store_purchases']
    
    df['total_purchases'] = df[channel_cols].sum(axis=1)
    df['web_purchase_ratio'] = df['num_web_purchases'] / (df['total_purchases'] + 1)
    df['store_purchase_ratio'] = df['num_store_purchases'] / (df['total_purchases'] + 1)
    
    # ===========================================
    # 4. CAMPAIGN ENGAGEMENT
    # ===========================================
    campaign_cols = [col for col in df.columns if 'campaign' in col.lower()]
    
    if campaign_cols:
        df['total_campaigns_accepted'] = df[campaign_cols].sum(axis=1)
        df['campaign_acceptance_rate'] = df['total_campaigns_accepted'] / len(campaign_cols)
    else:
        df['total_campaigns_accepted'] = 0
        df['campaign_acceptance_rate'] = 0.0
    
    # ===========================================
    # 5. FAMILY COMPOSITION
    # ===========================================
    df['family_size'] = df['num_children'] + df['num_teenagers'].fillna(0)
    df['has_dependents'] = (df['family_size'] > 0).astype(int)
    
    return df


def prepare_input_features(df, numeric_cols):
    """
    Select and order features for model prediction.
    Returns DataFrame to preserve feature names for sklearn.
    
    Args:
        df (pd.DataFrame): DataFrame with all engineered features
        numeric_cols (list): List of column names in the correct order for the model
        
    Returns:
        pd.DataFrame: Feature matrix ready for scaling/prediction (preserves column names)
    """
    # Ensure all required columns exist
    missing_cols = [col for col in numeric_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        print(f"✅ Available columns: {df.columns.tolist()}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Return DataFrame instead of .values to preserve feature names
    return df[numeric_cols]


def encode_education(education_level):
    """
    Encode education level to numeric value.
    Matches the encoding from the notebook.
    
    Args:
        education_level (str): Education level string
        
    Returns:
        float: Encoded education value
    """
    education_mapping = {
        'Basic': 1.0,
        '2n Cycle': 2.0,
        'Graduation': 3.0,
        'Master': 4.0,
        'PhD': 5.0,
        'Unknown': 0.0
    }
    return education_mapping.get(education_level, 0.0)


def create_marital_status_dummies(df):
    """
    Create one-hot encoded columns for marital status.
    
    Args:
        df (pd.DataFrame): DataFrame with 'marital_status' column
        
    Returns:
        pd.DataFrame: DataFrame with marital status dummy columns added
    """
    # Define expected categories (matching notebook)
    marital_categories = ['Divorced', 'Married', 'Other', 'Single', 'Together', 'Widow']
    
    for category in marital_categories:
        col_name = f'marital_status_{category}'
        df[col_name] = (df['marital_status'] == category).astype(float)
    
    return df