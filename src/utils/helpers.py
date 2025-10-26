"""
Utility Helper Functions - EdTech Token Economy

This module provides utility functions used across the EdTech platform.

Functions:
- Data validation
- Metric calculations
- Formatting utilities
- Configuration helpers

Author: EdTech Token Economy Pipeline
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate a DataFrame for common issues

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        Dictionary with validation results
    """
    validation = {
        'is_valid': True,
        'issues': [],
        'warnings': []
    }

    # Check if DataFrame is empty
    if df.empty:
        validation['is_valid'] = False
        validation['issues'].append("DataFrame is empty")
        return validation

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation['is_valid'] = False
            validation['issues'].append(f"Missing required columns: {missing_cols}")

    # Check for missing values
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 20:
        validation['warnings'].append(f"High missing data: {missing_pct:.1f}%")

    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        validation['warnings'].append(f"Found {duplicates} duplicate rows")

    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            validation['is_valid'] = False
            validation['issues'].append(f"Found {inf_count} infinite values")

    logger.info(f"DataFrame validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")
    return validation


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values

    Args:
        old_value: Original value
        new_value: New value

    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    
    return ((new_value - old_value) / old_value) * 100


def format_currency(amount: float, currency: str = 'USD') -> str:
    """
    Format amount as currency string

    Args:
        amount: Amount to format
        currency: Currency code

    Returns:
        Formatted currency string
    """
    if currency == 'USD':
        return f"${amount:,.2f}"
    elif currency == 'tokens':
        return f"{amount:,.0f} tokens"
    else:
        return f"{amount:,.2f} {currency}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage string

    Args:
        value: Value to format (as decimal, e.g., 0.15 for 15%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def calculate_roi(investment: float, returns: float) -> float:
    """
    Calculate Return on Investment

    Args:
        investment: Investment amount
        returns: Returns amount

    Returns:
        ROI percentage
    """
    if investment == 0:
        return 0.0
    
    return ((returns - investment) / investment) * 100


def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
    """
    Calculate Compound Annual Growth Rate

    Args:
        start_value: Starting value
        end_value: Ending value
        years: Number of years

    Returns:
        CAGR percentage
    """
    if start_value == 0 or years == 0:
        return 0.0
    
    return (((end_value / start_value) ** (1 / years)) - 1) * 100


def save_json(data: Dict, filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file

    Args:
        data: Dictionary to save
        filepath: Path to save file
        indent: JSON indentation
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        logger.info(f"Saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")


def load_json(filepath: str) -> Dict:
    """
    Load dictionary from JSON file

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON: {e}")
        return {}


def create_summary_statistics(df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    """
    Create summary statistics for DataFrame

    Args:
        df: DataFrame to summarize
        numeric_only: Whether to include only numeric columns

    Returns:
        DataFrame with summary statistics
    """
    if numeric_only:
        df = df.select_dtypes(include=[np.number])
    
    summary = df.describe().T
    summary['missing'] = df.isnull().sum()
    summary['missing_pct'] = (df.isnull().sum() / len(df)) * 100
    
    return summary


def bin_numeric_column(series: pd.Series, n_bins: int = 5, labels: List[str] = None) -> pd.Series:
    """
    Bin a numeric column into categories

    Args:
        series: Numeric series to bin
        n_bins: Number of bins
        labels: Labels for bins

    Returns:
        Categorical series
    """
    if labels is None:
        labels = [f"Bin_{i+1}" for i in range(n_bins)]
    
    return pd.cut(series, bins=n_bins, labels=labels)


def detect_outliers_iqr(series: pd.Series, threshold: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method

    Args:
        series: Numeric series
        threshold: IQR threshold multiplier

    Returns:
        Boolean series indicating outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def normalize_column(series: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize a numeric column

    Args:
        series: Numeric series
        method: Normalization method ('minmax', 'zscore')

    Returns:
        Normalized series
    """
    if method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'zscore':
        return (series - series.mean()) / series.std()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_elasticity(price_change_pct: float, demand_change_pct: float) -> float:
    """
    Calculate price elasticity of demand

    Args:
        price_change_pct: Percentage change in price
        demand_change_pct: Percentage change in demand

    Returns:
        Elasticity coefficient
    """
    if price_change_pct == 0:
        return 0.0
    
    return demand_change_pct / price_change_pct


def generate_timestamp() -> str:
    """
    Generate ISO format timestamp

    Returns:
        Timestamp string
    """
    return datetime.now().isoformat()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero

    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    
    return numerator / denominator


def calculate_ltv_cac_ratio(ltv: float, cac: float) -> float:
    """
    Calculate LTV:CAC ratio

    Args:
        ltv: Lifetime value
        cac: Customer acquisition cost

    Returns:
        LTV:CAC ratio
    """
    return safe_divide(ltv, cac, default=0.0)


def format_large_number(num: float) -> str:
    """
    Format large numbers with K, M, B suffixes

    Args:
        num: Number to format

    Returns:
        Formatted string
    """
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return f"{num:.0f}"


# Example usage
if __name__ == "__main__":
    print("Testing EdTech Utility Functions...")

    # Test DataFrame validation
    test_df = pd.DataFrame({
        'price': [100, 150, 200],
        'enrollments': [50, 75, 100]
    })
    
    validation = validate_dataframe(test_df, required_columns=['price', 'enrollments'])
    print(f"✅ DataFrame validation: {validation['is_valid']}")

    # Test percentage change
    pct_change = calculate_percentage_change(100, 120)
    print(f"✅ Percentage change: {pct_change:.1f}%")

    # Test currency formatting
    formatted = format_currency(1234.56)
    print(f"✅ Formatted currency: {formatted}")

    # Test ROI calculation
    roi = calculate_roi(1000, 1500)
    print(f"✅ ROI: {roi:.1f}%")

    # Test large number formatting
    large_num = format_large_number(1_234_567)
    print(f"✅ Large number: {large_num}")

    print("\n✅ All utility functions tested successfully!")

