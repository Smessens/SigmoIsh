import json
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import re
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import pandas as pd

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

def analyze_sigmoid_price_correlation_multi_distance(data_dir="collected_data"):
    """Enhanced analysis with multiple distance metrics"""
    print("=== Multi-Distance Sigmoid-Price Correlation Analysis ===\n")
    
    # Collect and sort market files
    sorted_files = collect_market_files(data_dir)
    print(f"Found {len(sorted_files)} Bitcoin market files")
    
    correlation_data = []
    valid_markets = []
    invalid_markets = []
    
    # Process each market file
    for file_index, (date, file_path, filename) in enumerate(sorted_files):
        json_data = read_json_file(file_path)
        
        if not json_data:
            invalid_markets.append((filename, "Failed to read JSON"))
            continue
        
        # Get market metadata
        metadata = json_data.get("collection_metadata", {})
        data_points = json_data.get("data_points", [])
        
        if not data_points:
            invalid_markets.append((filename, "No data points"))
            continue
        
        end_timestamp = metadata.get("collection_period", {}).get("end")
        starting_price = metadata.get("starting_price")
        
        if not end_timestamp or not starting_price:
            invalid_markets.append((filename, "Missing end timestamp or starting price"))
            continue
        
        # Get next market's starting price (as proxy for end price)
        next_starting_price = get_next_market_starting_price(file_index, sorted_files)
        if not next_starting_price:
            invalid_markets.append((filename, "No next market for end price"))
            continue
        
        # Collect hourly bucket data for sigmoid fitting
        hourly_data = {}
        
        # First pass: collect data for sigmoid fitting
        for point in data_points:
            if not isinstance(point, dict):
                continue
            
            timestamp = point.get("timestamp")
            orderbook = point.get("orderbook")
            
            if not timestamp or not orderbook:
                continue
            
            hours_remaining = calculate_hours_remaining(timestamp, end_timestamp)
            if hours_remaining <= 0 or hours_remaining > 24:
                continue
            
            prob = get_market_midpoint_probability(orderbook)
            if prob is None:
                continue
            
            hour_bucket = int(hours_remaining)
            if hour_bucket not in hourly_data:
                hourly_data[hour_bucket] = []
            hourly_data[hour_bucket].append(prob)
        
        # Average probabilities per hour bucket
        hourly_averages = {}
        for hour, probs in hourly_data.items():
            if probs:
                hourly_averages[hour] = np.mean(probs)
        
        if len(hourly_averages) < 4:
            invalid_markets.append((filename, f"Insufficient hourly data ({len(hourly_averages)} hours)"))
            continue
        
        # Fit sigmoid to hourly data
        sigmoid_params = fit_hourly_sigmoid(hourly_averages)
        if sigmoid_params is None:
            invalid_markets.append((filename, "Failed to fit sigmoid"))
            continue
        
        # Calculate residuals for standardization
        residuals = []
        for hour, avg_prob in hourly_averages.items():
            expected = sigmoid(hour, *sigmoid_params)
            residuals.append(avg_prob - expected)
        market_residuals_std = np.std(residuals) if residuals else None
        
        market_valid_points = 0
        
        # Second pass: calculate multiple distances
        for point in data_points:
            if not isinstance(point, dict):
                continue
            
            timestamp = point.get("timestamp")
            orderbook = point.get("orderbook")
            
            if not timestamp or not orderbook:
                continue
            
            hours_remaining = calculate_hours_remaining(timestamp, end_timestamp)
            if hours_remaining <= 0 or hours_remaining > 24:
                continue
            
            current_prob = get_market_midpoint_probability(orderbook)
            current_btc_price = get_btc_price_from_point(point)
            
            if current_prob is None or current_btc_price is None:
                continue
            
            expected_prob = sigmoid(hours_remaining, *sigmoid_params)
            price_evolution = (next_starting_price - current_btc_price) / current_btc_price
            
            # Calculate multiple distance metrics
            distances = calculate_multiple_distances(current_prob, expected_prob, market_residuals_std)
            
            # Store correlation data
            correlation_data.append({
                'market': filename,
                'timestamp': timestamp,
                'hours_remaining': hours_remaining,
                'current_prob': current_prob,
                'expected_prob': expected_prob,
                'price_evolution': price_evolution,
                'current_btc_price': current_btc_price,
                'end_btc_price': next_starting_price,
                **{f'distance_{k}': v for k, v in distances.items()}
            })
            
            market_valid_points += 1
        
        if market_valid_points > 0:
            valid_markets.append((filename, market_valid_points))
        else:
            invalid_markets.append((filename, "No valid correlation points"))
    
    if not correlation_data:
        print("\nNo valid correlation data found!")
        return
    
    df = pd.DataFrame(correlation_data)
    
    # Calculate correlations for all distance metrics
    distance_columns = [col for col in df.columns if col.startswith('distance_')]
    correlations = {}
    
    print(f"\n=== Multi-Distance Correlation Results ===")
    print(f"Total datapoints: {len(df):,}")
    print(f"Valid markets: {len(valid_markets)}")
    print(f"Invalid markets: {len(invalid_markets)}")
    
    for dist_col in distance_columns:
        if df[dist_col].notna().any():
            valid_data = df.dropna(subset=[dist_col, 'price_evolution'])
            if len(valid_data) > 100:
                corr, p_val = pearsonr(valid_data[dist_col], valid_data['price_evolution'])
                correlations[dist_col] = {'correlation': corr, 'p_value': p_val, 'n_points': len(valid_data)}
                
                distance_name = dist_col.replace('distance_', '').title()
                significance = "Significant" if p_val < 0.05 else "Not significant"
                print(f"{distance_name:12}: r = {corr:7.4f}, p = {p_val:.4f}, n = {len(valid_data):5,} ({significance})")
    
    # Create enhanced visualization
    create_multi_distance_plots(df, correlations)
    
    return df, correlations

def create_multi_distance_plots(df, correlations):
    """Create plots comparing different distance metrics"""
    valid_distances = [k for k, v in correlations.items() if v is not None]
    n_distances = len(valid_distances)
    
    if n_distances == 0:
        print("No valid distance metrics to plot")
        return
    
    # Create subplot grid
    n_cols = min(3, n_distances)
    n_rows = (n_distances + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_distances == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, dist_col in enumerate(valid_distances):
        ax = axes[i]
        
        # Get valid data for this distance metric
        valid_data = df.dropna(subset=[dist_col, 'price_evolution'])
        
        if len(valid_data) == 0:
            continue
            
        # Scatter plot
        ax.scatter(valid_data[dist_col], valid_data['price_evolution'], alpha=0.5, s=10)
        
        # Trend line
        z = np.polyfit(valid_data[dist_col], valid_data['price_evolution'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data[dist_col].min(), valid_data[dist_col].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        
        # Labels and title
        distance_name = dist_col.replace('distance_', '').title()
        corr_info = correlations[dist_col]
        ax.set_xlabel(f'{distance_name} Distance')
        ax.set_ylabel('Price Evolution')
        ax.set_title(f'{distance_name}\nr = {corr_info["correlation"]:.4f}, p = {corr_info["p_value"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_distances, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = "analysis_results"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "multi_distance_correlation_analysis.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nMulti-distance plot saved: {plot_dir}/multi_distance_correlation_analysis.png")

def main():
    """Main execution function"""
    df, correlations = analyze_sigmoid_price_correlation_multi_distance()

if __name__ == "__main__":
    main() 