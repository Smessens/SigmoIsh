"""
Utility functions for market analysis
"""

import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import os
import re
from datetime import datetime

def read_json_file(file_path):
    """Reads a JSON file and returns its content."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_date_from_filename(filename):
    """Extract date from filename in format bitcoin-up-or-down-on-may-30.json"""
    base_filename = os.path.basename(filename)
    
    # Pattern for: crypto-up-or-down-on-month-day.json
    pattern = r'(\w+)-up-or-down-on-(\w+)-(\d+)\.json'
    match = re.search(pattern, base_filename)
    
    if match:
        crypto, month, day = match.groups()
        try:
            # Convert month name to number
            month_map = {
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
                'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            month_num = month_map.get(month.lower())
            if month_num:
                # Assume current year (2024) since it's not in filename
                return datetime(2024, month_num, int(day))
        except ValueError:
            pass
    
    return None

def sigmoid(x, L, k, x0, b):
    """Sigmoid function for fitting hourly data"""
    return L / (1 + np.exp(-k * (x - x0))) + b

def fit_hourly_sigmoid(hourly_data):
    """Fit sigmoid to hourly bucket data"""
    if len(hourly_data) < 4:
        return None
    
    hours = np.array(list(hourly_data.keys()))
    probs = np.array(list(hourly_data.values()))
    
    try:
        # Initial parameter guess
        L_init = 1.0
        k_init = 0.1
        x0_init = np.mean(hours)
        b_init = 0.0
        
        popt, _ = curve_fit(sigmoid, hours, probs, 
                           p0=[L_init, k_init, x0_init, b_init],
                           maxfev=5000)
        return popt
    except:
        return None

def calculate_hours_remaining(timestamp, end_timestamp):
    """Calculate hours remaining from timestamp to end_timestamp"""
    return max(0, (end_timestamp - timestamp) / 3600.0)

def get_market_midpoint_probability(orderbook):
    """Calculate market midpoint probability from orderbook"""
    if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
        return None
    
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        return None
    
    try:
        valid_bids = [b.get('price') for b in bids if isinstance(b, dict) and 'price' in b]
        valid_asks = [a.get('price') for a in asks if isinstance(a, dict) and 'price' in a]
        
        if not valid_bids or not valid_asks:
            return None
        
        max_bid = max(valid_bids)
        min_ask = min(valid_asks)
        
        if 0 <= max_bid <= 1 and 0 <= min_ask <= 1 and max_bid <= min_ask:
            return (max_bid + min_ask) / 2.0
    except:
        pass
    
    return None

def get_btc_price_from_point(data_point):
    """Extract BTC price from data point"""
    if "btc_price" in data_point:
        return data_point["btc_price"]
    elif "price" in data_point:
        return data_point["price"]
    return None

def collect_market_files(data_dir="collected_data"):
    """Collect and sort Bitcoin market files chronologically"""
    btc_files = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return btc_files
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and "bitcoin-up-or-down" in filename.lower():
            file_path = os.path.join(data_dir, filename)
            date = extract_date_from_filename(filename)
            if date:
                btc_files.append((date, file_path, filename))
    
    # Sort by date
    btc_files.sort(key=lambda x: x[0])
    return btc_files

def get_next_market_starting_price(current_file_index, sorted_files):
    """Get the starting price of the next market"""
    if current_file_index + 1 >= len(sorted_files):
        return None
    
    next_file_path = sorted_files[current_file_index + 1][1]
    next_data = read_json_file(next_file_path)
    
    if not next_data or "collection_metadata" not in next_data:
        return None
    
    return next_data["collection_metadata"].get("starting_price")

def calculate_multiple_distances(current_prob, expected_prob, market_residuals_std=None):
    """Calculate multiple distance metrics between current and expected probability"""
    
    # 1. Linear distance (current)
    linear_distance = current_prob - expected_prob
    
    # 2. Absolute distance
    absolute_distance = abs(current_prob - expected_prob)
    
    # 3. Squared distance
    squared_distance = (current_prob - expected_prob) ** 2
    
    # 4. Relative distance (percentage)
    relative_distance = (current_prob - expected_prob) / max(expected_prob, 0.01)  # Avoid division by zero
    
    # 5. Log-odds distance
    try:
        current_logit = np.log(current_prob / (1 - current_prob + 1e-10))
        expected_logit = np.log(expected_prob / (1 - expected_prob + 1e-10))
        logit_distance = current_logit - expected_logit
    except:
        logit_distance = None
    
    # 6. Standardized distance (if we have market std)
    standardized_distance = None
    if market_residuals_std and market_residuals_std > 0:
        standardized_distance = linear_distance / market_residuals_std
    
    return {
        'linear': linear_distance,
        'absolute': absolute_distance,
        'squared': squared_distance,
        'relative': relative_distance,
        'logit': logit_distance,
        'standardized': standardized_distance
    }

def calculate_dispersion_metrics(data):
    """Calculate various dispersion metrics for a dataset"""
    if len(data) == 0:
        return {}
    
    return {
        'std': np.std(data),
        'var': np.var(data),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25),
        'range': np.max(data) - np.min(data),
        'mad': np.median(np.abs(data - np.median(data))),  # Median Absolute Deviation
        'cv': np.std(data) / np.mean(data) if np.mean(data) != 0 else 0  # Coefficient of Variation
    }

def calculate_time_bucket_stats(df, time_bucket_size=2):
    """Calculate statistics for time buckets"""
    buckets = []
    
    max_hours = int(df['hours_remaining'].max())
    
    for start_hour in range(0, max_hours + 1, time_bucket_size):
        end_hour = start_hour + time_bucket_size
        
        bucket_data = df[
            (df['hours_remaining'] >= start_hour) & 
            (df['hours_remaining'] < end_hour)
        ]
        
        if len(bucket_data) < 10:  # Skip buckets with too few points
            continue
        
        # Calculate correlation
        corr, p_val = pearsonr(bucket_data['distance_linear'], bucket_data['price_evolution'])
        
        # Calculate dispersion metrics
        linear_dispersion = calculate_dispersion_metrics(bucket_data['distance_linear'])
        price_dispersion = calculate_dispersion_metrics(bucket_data['price_evolution'])
        
        buckets.append({
            'time_bucket': f"{start_hour}-{end_hour}h",
            'start_hour': start_hour,
            'end_hour': end_hour,
            'mid_hour': (start_hour + end_hour) / 2,
            'n_points': len(bucket_data),
            'correlation': corr,
            'p_value': p_val,
            'linear_std': linear_dispersion['std'],
            'linear_iqr': linear_dispersion['iqr'],
            'linear_range': linear_dispersion['range'],
            'price_std': price_dispersion['std'],
            'price_iqr': price_dispersion['iqr'],
            'price_range': price_dispersion['range'],
            'linear_cv': linear_dispersion['cv'],
            'price_cv': price_dispersion['cv']
        })
    
    return pd.DataFrame(buckets) 