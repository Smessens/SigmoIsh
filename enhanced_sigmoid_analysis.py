import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
from scipy.optimize import curve_fit
import os
import re
from datetime import datetime

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

def collect_market_files(data_dir="collected_data", crypto="bitcoin"):
    """Collect and sort market files chronologically for specified cryptocurrency"""
    crypto_files = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return crypto_files
    
    search_pattern = f"{crypto.lower()}-up-or-down"
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and search_pattern in filename.lower():
            file_path = os.path.join(data_dir, filename)
            date = extract_date_from_filename(filename)
            if date:
                crypto_files.append((date, file_path, filename))
    
    # Sort by date
    crypto_files.sort(key=lambda x: x[0])
    return crypto_files

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
    
    # 7. NEW: Exponential distance (emphasizes larger deviations)
    exponential_distance = np.sign(linear_distance) * (np.exp(abs(linear_distance)) - 1)
    
    # 8. NEW: Power distance (adjustable exponent)
    power_distance = np.sign(linear_distance) * (abs(linear_distance) ** 1.5)
    
    # 9. NEW: Hyperbolic distance (bounded but sensitive)
    hyperbolic_distance = np.tanh(linear_distance * 5)  # Scale factor of 5
    
    # 10. NEW: Asymmetric distance (different treatment for positive/negative)
    if linear_distance > 0:
        asymmetric_distance = linear_distance ** 1.2  # Penalize overoptimism more
    else:
        asymmetric_distance = linear_distance ** 1.0  # Standard penalty for pessimism
    
    # 11. NEW: Winsorized distance (cap extreme values)
    winsorized_distance = np.clip(linear_distance, -0.3, 0.3)
    
    # 12. NEW: Huber-like distance (robust to outliers)
    delta = 0.1
    if abs(linear_distance) <= delta:
        huber_distance = linear_distance ** 2 / (2 * delta)
    else:
        huber_distance = np.sign(linear_distance) * (abs(linear_distance) - delta/2)
    
    return {
        'linear': linear_distance,
        'absolute': absolute_distance,
        'squared': squared_distance,
        'relative': relative_distance,
        'logit': logit_distance,
        'standardized': standardized_distance,
        'exponential': exponential_distance,
        'power': power_distance,
        'hyperbolic': hyperbolic_distance,
        'asymmetric': asymmetric_distance,
        'winsorized': winsorized_distance,
        'huber': huber_distance
    }

def analyze_sigmoid_price_correlation_multi_distance(data_dir="collected_data", crypto="bitcoin", min_hours=0, max_hours=24):
    """Enhanced analysis with multiple distance metrics for specific cryptocurrency and time window"""
    print(f"=== Multi-Distance Sigmoid-Price Correlation Analysis ({crypto.upper()}, {min_hours}-{max_hours}h) ===\n")
    
    # Collect and sort market files
    sorted_files = collect_market_files(data_dir, crypto)
    print(f"Found {len(sorted_files)} {crypto} market files")
    
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
        
        # Collect hourly bucket data for sigmoid fitting (using full time range for fitting)
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
        
        # Second pass: calculate multiple distances (applying time window filter here)
        for point in data_points:
            if not isinstance(point, dict):
                continue
            
            timestamp = point.get("timestamp")
            orderbook = point.get("orderbook")
            
            if not timestamp or not orderbook:
                continue
            
            hours_remaining = calculate_hours_remaining(timestamp, end_timestamp)
            
            # Apply time window filter
            if hours_remaining <= min_hours or hours_remaining > max_hours:
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
                'crypto': crypto,
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
        print(f"\nNo valid correlation data found for {crypto}!")
        return None, None
    
    df = pd.DataFrame(correlation_data)
    
    # Calculate correlations for all distance metrics
    distance_columns = [col for col in df.columns if col.startswith('distance_')]
    correlations = {}
    
    print(f"\n=== Multi-Distance Correlation Results ({crypto.upper()}, {min_hours}-{max_hours}h) ===")
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
    create_multi_distance_plots(df, correlations, crypto, min_hours, max_hours)
    
    return df, correlations

def create_multi_distance_plots(df, correlations, crypto="bitcoin", min_hours=0, max_hours=24):
    """Create plots comparing different distance metrics"""
    valid_distances = [k for k, v in correlations.items() if v is not None]
    n_distances = len(valid_distances)
    
    if n_distances == 0:
        print("No valid distance metrics to plot")
        return
    
    # Create subplot grid - add one more subplot for summary
    n_cols = min(3, n_distances)
    n_rows = (n_distances + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows + 1, n_cols, figsize=(5*n_cols, 4*(n_rows + 1)))
    if n_distances == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    N_total = len(df)
    weighted_metric_totals = {}

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
        
        # Calculate average Y per X bin
        num_bins = 15
        bins = pd.cut(valid_data[dist_col], num_bins)
        avg_by_bin = valid_data.groupby(bins, observed=False)['price_evolution'].mean()
        bin_counts = valid_data.groupby(bins, observed=False)['price_evolution'].count()
        bin_centers = [(interval.left + interval.right)/2 for interval in avg_by_bin.index]
        
        # Plot average values
        ax.plot(bin_centers, avg_by_bin.values, 'o-', color='black', linewidth=2, markersize=6, label='Avg Y')
        
        # Calculate and plot the requested metric
        weighted_metric = (avg_by_bin.abs() * bin_counts) / N_total
        weighted_metric_total = weighted_metric.sum()
        weighted_metric_totals[dist_col] = weighted_metric_total
        
        # Secondary y-axis for the metric
        ax2 = ax.twinx()
        ax2.plot(bin_centers, weighted_metric.values, 's--', color='orange', markersize=5, label='Weighted |mean|')
        ax2.set_ylabel('Weighted |mean|', color='orange', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Labels and title
        distance_name = dist_col.replace('distance_', '').title()
        corr_info = correlations[dist_col]
        ax.set_xlabel(f'{distance_name} Distance')
        ax.set_ylabel('Price Evolution')
        ax.set_title(f'{distance_name}\nr = {corr_info["correlation"]:.4f}, p = {corr_info["p_value"]:.4f}')
        ax.grid(True, alpha=0.3)
        
        # Legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8, loc='upper right')
    
    # Hide unused subplots in main grid
    for i in range(n_distances, n_cols * n_rows):
        if i < len(axes):
            axes[i].set_visible(False)
    
    # Create summary bar plot in the last row
    summary_ax = plt.subplot(n_rows + 1, n_cols, n_cols * n_rows + 1)
    
    # Prepare data for summary plot
    distance_names = [col.replace('distance_', '').title() for col in valid_distances]
    r_values = [abs(correlations[col]['correlation']) for col in valid_distances]  # Use absolute value
    weighted_totals = [weighted_metric_totals[col] for col in valid_distances]
    
    # Normalize values for better visualization
    r_normalized = np.array(r_values) / max(r_values)
    weighted_normalized = np.array(weighted_totals) / max(weighted_totals)
    
    x_pos = np.arange(len(distance_names))
    width = 0.35
    
    # Create bars
    bars1 = summary_ax.bar(x_pos - width/2, r_normalized, width, label='|Correlation|', alpha=0.8, color='steelblue')
    bars2 = summary_ax.bar(x_pos + width/2, weighted_normalized, width, label='Weighted |mean|', alpha=0.8, color='orange')
    
    # Add value labels on bars
    for i, (bar1, bar2, r_val, w_val) in enumerate(zip(bars1, bars2, r_values, weighted_totals)):
        summary_ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01, 
                       f'{r_val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        summary_ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01, 
                       f'{w_val:.4f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    summary_ax.set_xlabel('Distance Metrics')
    summary_ax.set_ylabel('Normalized Values')
    summary_ax.set_title(f'Summary: |Correlation| vs Weighted |mean| ({crypto.upper()}, {min_hours}-{max_hours}h)')
    summary_ax.set_xticks(x_pos)
    summary_ax.set_xticklabels(distance_names, rotation=45, ha='right')
    summary_ax.legend()
    summary_ax.grid(True, alpha=0.3)
    summary_ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = "analysis_results"
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"multi_distance_correlation_{crypto}_{min_hours}h_{max_hours}h.png"
    plt.savefig(os.path.join(plot_dir, filename), 
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nMulti-distance plot saved: {plot_dir}/{filename}")
    print("\nWeighted |mean| totals per metric:")
    for dist_col, total in weighted_metric_totals.items():
        print(f"{dist_col}: {total:.6f}")

def create_time_colored_linear_plot(df, correlations, crypto="bitcoin", min_hours=0, max_hours=24):
    """Create a large linear distance plot with time-based color coding"""
    
    # Get valid data for linear distance
    valid_data = df.dropna(subset=['distance_linear', 'price_evolution'])
    
    if len(valid_data) == 0:
        print("No valid linear distance data to plot")
        return
    
    # Create figure with larger size
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with color based on hours remaining
    scatter = plt.scatter(valid_data['distance_linear'], 
                         valid_data['price_evolution'], 
                         c=valid_data['hours_remaining'], 
                         cmap='viridis', 
                         alpha=0.6, 
                         s=15)
    
    # Add trend line
    z = np.polyfit(valid_data['distance_linear'], valid_data['price_evolution'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(valid_data['distance_linear'].min(), valid_data['distance_linear'].max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    
    # Calculate average Y per X bin
    num_bins = 20
    bins = pd.cut(valid_data['distance_linear'], num_bins)
    avg_by_bin = valid_data.groupby(bins, observed=False)['price_evolution'].mean()
    bin_counts = valid_data.groupby(bins, observed=False)['price_evolution'].count()
    bin_centers = [(interval.left + interval.right)/2 for interval in avg_by_bin.index]
    
    # Plot average values
    plt.plot(bin_centers, avg_by_bin.values, 'o-', color='black', linewidth=2, 
             markersize=8, label='Avg Price Evolution')
    
    # Calculate and plot the requested metric
    N_total = len(valid_data)
    weighted_metric = (avg_by_bin.abs() * bin_counts) / N_total
    weighted_metric_total = weighted_metric.sum()
    
    ax2 = plt.gca().twinx()
    ax2.plot(bin_centers, weighted_metric.values, 's--', color='orange', markersize=6, label='Weighted |mean|')
    ax2.set_ylabel('Weighted |mean|', color='orange', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='orange')
    
    # Get correlation information
    linear_corr_info = correlations.get('distance_linear', {})
    corr = linear_corr_info.get('correlation', 0)
    p_val = linear_corr_info.get('p_value', 1)
    n_points = linear_corr_info.get('n_points', len(valid_data))
    
    # Labels and title with larger fonts
    plt.xlabel('Linear Distance', fontsize=14)
    plt.ylabel('Price Evolution', fontsize=14)
    plt.title(f'Linear Distance vs Price Evolution ({crypto.upper()}, {min_hours}-{max_hours}h)\nr = {corr:.4f}, p = {p_val:.4f}, n = {n_points:,}', 
              fontsize=16, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Hours Remaining', fontsize=12)
    
    # Grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Adjust tick sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_dir = "analysis_results"
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"linear_distance_time_colored_{crypto}_{min_hours}h_{max_hours}h.png"
    plt.savefig(os.path.join(plot_dir, filename), 
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nTime-colored linear distance plot saved: {plot_dir}/{filename}")
    print(f"Weighted |mean| total: {weighted_metric_total:.6f}")
    print(f"\nTime distribution:")
    print(f"Hours remaining range: {valid_data['hours_remaining'].min():.1f} - {valid_data['hours_remaining'].max():.1f}")
    print(f"Average hours remaining: {valid_data['hours_remaining'].mean():.1f}")
    print(f"Median hours remaining: {valid_data['hours_remaining'].median():.1f}")

def analyze_asymmetric_effects(df):
    """Analyze if positive and negative deviations have different effects"""
    print("\n=== Asymmetric Effects Analysis ===")
    
    # Split into positive and negative deviations
    positive_deviations = df[df['distance_linear'] > 0]
    negative_deviations = df[df['distance_linear'] < 0]
    
    if len(positive_deviations) > 100 and len(negative_deviations) > 100:
        corr_pos, p_pos = pearsonr(positive_deviations['distance_linear'], positive_deviations['price_evolution'])
        corr_neg, p_neg = pearsonr(negative_deviations['distance_linear'], negative_deviations['price_evolution'])
        
        print(f"Positive deviations (market too optimistic): r = {corr_pos:.4f}, p = {p_pos:.4f}, n = {len(positive_deviations):,}")
        print(f"Negative deviations (market too pessimistic): r = {corr_neg:.4f}, p = {p_neg:.4f}, n = {len(negative_deviations):,}")
        
        # Calculate means
        mean_evolution_pos = positive_deviations['price_evolution'].mean()
        mean_evolution_neg = negative_deviations['price_evolution'].mean()
        
        print(f"Average price evolution when market too optimistic: {mean_evolution_pos:.4f} ({mean_evolution_pos*100:.2f}%)")
        print(f"Average price evolution when market too pessimistic: {mean_evolution_neg:.4f} ({mean_evolution_neg*100:.2f}%)")
        
        return {
            'positive': {'correlation': corr_pos, 'p_value': p_pos, 'n': len(positive_deviations), 'mean_evolution': mean_evolution_pos},
            'negative': {'correlation': corr_neg, 'p_value': p_neg, 'n': len(negative_deviations), 'mean_evolution': mean_evolution_neg}
        }
    
    return None

def analyze_time_dependent_effects(df):
    """Analyze how correlation changes by hours remaining"""
    print("\n=== Time-Dependent Effects Analysis ===")
    
    time_correlations = {}
    for hour in range(24):
        hour_data = df[df['hours_remaining'].between(hour, hour + 1)]
        if len(hour_data) > 100:
            corr, p_val = pearsonr(hour_data['distance_linear'], hour_data['price_evolution'])
            time_correlations[hour] = {
                'correlation': corr, 
                'p_value': p_val, 
                'n_points': len(hour_data),
                'mean_distance': hour_data['distance_linear'].mean(),
                'std_distance': hour_data['distance_linear'].std()
            }
            print(f"Hour {hour:2d}-{hour+1:2d}: r = {corr:7.4f}, p = {p_val:.4f}, n = {len(hour_data):4,}")
    
    return time_correlations

def analyze_magnitude_thresholds(df):
    """Analyze correlation by magnitude of deviation"""
    print("\n=== Magnitude Threshold Analysis ===")
    
    # Create quantiles based on absolute distance
    df['abs_distance_quantile'] = pd.qcut(df['distance_absolute'], q=10, labels=False)
    
    quantile_results = {}
    for q in range(10):
        quantile_data = df[df['abs_distance_quantile'] == q]
        if len(quantile_data) > 100:
            corr, p_val = pearsonr(quantile_data['distance_linear'], quantile_data['price_evolution'])
            
            quantile_results[q] = {
                'correlation': corr,
                'p_value': p_val,
                'n_points': len(quantile_data),
                'distance_range': (quantile_data['distance_absolute'].min(), quantile_data['distance_absolute'].max()),
                'mean_evolution': quantile_data['price_evolution'].mean()
            }
            
            print(f"Quantile {q} (|distance| {quantile_data['distance_absolute'].min():.3f}-{quantile_data['distance_absolute'].max():.3f}): "
                  f"r = {corr:.4f}, p = {p_val:.4f}, n = {len(quantile_data):,}")
    
    return quantile_results

def create_enhanced_visualizations(df, asymmetric_results, time_correlations, quantile_results):
    """Create comprehensive visualization suite"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Asymmetric effects
    ax1 = plt.subplot(3, 4, 1)
    positive_data = df[df['distance_linear'] > 0]
    negative_data = df[df['distance_linear'] < 0]
    
    ax1.scatter(positive_data['distance_linear'], positive_data['price_evolution'], 
               alpha=0.5, s=5, color='red', label=f'Overoptimistic (n={len(positive_data):,})')
    ax1.scatter(negative_data['distance_linear'], negative_data['price_evolution'], 
               alpha=0.5, s=5, color='blue', label=f'Overpessimistic (n={len(negative_data):,})')
    ax1.set_xlabel('Linear Distance')
    ax1.set_ylabel('Price Evolution')
    ax1.set_title('Asymmetric Effects')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time-dependent correlation
    ax2 = plt.subplot(3, 4, 2)
    if time_correlations:
        hours = list(time_correlations.keys())
        corrs = [time_correlations[h]['correlation'] for h in hours]
        ax2.plot(hours, corrs, 'o-', linewidth=2, markersize=6)
        ax2.set_xlabel('Hours Remaining')
        ax2.set_ylabel('Correlation Coefficient')
        ax2.set_title('Correlation by Time Remaining')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 3. Magnitude threshold effects
    ax3 = plt.subplot(3, 4, 3)
    if quantile_results:
        quantiles = list(quantile_results.keys())
        q_corrs = [quantile_results[q]['correlation'] for q in quantiles]
        ax3.plot(quantiles, q_corrs, 'o-', linewidth=2, markersize=6)
        ax3.set_xlabel('Distance Magnitude Quantile')
        ax3.set_ylabel('Correlation Coefficient')
        ax3.set_title('Correlation by Distance Magnitude')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # 4. Distribution of deviations by outcome
    ax4 = plt.subplot(3, 4, 4)
    high_performance = df[df['price_evolution'] > df['price_evolution'].quantile(0.8)]
    low_performance = df[df['price_evolution'] < df['price_evolution'].quantile(0.2)]
    
    ax4.hist(high_performance['distance_linear'], bins=50, alpha=0.6, label='Top 20% Performance', color='green')
    ax4.hist(low_performance['distance_linear'], bins=50, alpha=0.6, label='Bottom 20% Performance', color='red')
    ax4.set_xlabel('Linear Distance')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distance Distribution by Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Market evolution vs sigmoid prediction accuracy
    ax5 = plt.subplot(3, 4, 5)
    df['prediction_accuracy'] = 1 - df['distance_absolute']  # Higher = more accurate
    accuracy_bins = pd.qcut(df['prediction_accuracy'], q=10, labels=False)
    
    accuracy_evolution = []
    accuracy_levels = []
    for i in range(10):
        bin_data = df[accuracy_bins == i]
        if len(bin_data) > 100:
            accuracy_evolution.append(bin_data['price_evolution'].mean())
            accuracy_levels.append(bin_data['prediction_accuracy'].mean())
    
    if accuracy_evolution:
        ax5.plot(accuracy_levels, accuracy_evolution, 'o-', linewidth=2, markersize=8)
        ax5.set_xlabel('Sigmoid Prediction Accuracy')
        ax5.set_ylabel('Average Price Evolution')
        ax5.set_title('Performance vs Prediction Accuracy')
        ax5.grid(True, alpha=0.3)
    
    # 6. Heatmap of correlation by time and magnitude
    ax6 = plt.subplot(3, 4, 6)
    correlation_matrix = np.zeros((5, 24))  # 5 magnitude bins, 24 hours
    
    for hour in range(24):
        hour_data = df[df['hours_remaining'].between(hour, hour + 1)]
        if len(hour_data) > 50:
            magnitude_bins = pd.qcut(hour_data['distance_absolute'], q=5, labels=False)
            for mag_bin in range(5):
                bin_data = hour_data[magnitude_bins == mag_bin]
                if len(bin_data) > 20:
                    corr, _ = pearsonr(bin_data['distance_linear'], bin_data['price_evolution'])
                    correlation_matrix[mag_bin, hour] = corr
    
    im = ax6.imshow(correlation_matrix, aspect='auto', cmap='RdBu_r', vmin=-0.3, vmax=0.3)
    ax6.set_xlabel('Hours Remaining')
    ax6.set_ylabel('Distance Magnitude Bin')
    ax6.set_title('Correlation Heatmap (Time × Magnitude)')
    plt.colorbar(im, ax=ax6)
    
    # 7-12. Additional analysis plots
    # Add more sophisticated analysis here...
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = "analysis_results"
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "enhanced_sigmoid_analysis.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nEnhanced analysis plot saved: {plot_dir}/enhanced_sigmoid_analysis.png")

def analyze_eth_comparison(data_dir="collected_data"):
    """Compare results with Ethereum markets"""
    print("\n=== ETH vs BTC Comparison ===")
    
    # Modify collect_market_files to handle ETH
    eth_files = []
    
    if not os.path.exists(data_dir):
        return None
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and "ethereum-up-or-down" in filename.lower():
            file_path = os.path.join(data_dir, filename)
            date = extract_date_from_filename(filename)
            if date:
                eth_files.append((date, file_path, filename))
    
    eth_files.sort(key=lambda x: x[0])
    print(f"Found {len(eth_files)} Ethereum market files")
    
    if len(eth_files) < 5:
        print("Insufficient ETH data for comparison")
        return None
    
    # Analyze ETH using same methodology...
    # (Implementation similar to Bitcoin analysis)
    
    return None

def comprehensive_crypto_analysis():
    """Run comprehensive analysis for both BTC and ETH with different time windows"""
    print("=== COMPREHENSIVE CRYPTO ANALYSIS (BTC & ETH, Multiple Time Windows) ===")
    
    cryptocurrencies = ["bitcoin", "ethereum"]
    time_windows = [
        (0, 24, "full"),
        (4, 20, "mid-range")
    ]
    
    all_results = {}
    
    for crypto in cryptocurrencies:
        all_results[crypto] = {}
        print(f"\n{'='*60}")
        print(f"ANALYZING {crypto.upper()}")
        print(f"{'='*60}")
        
        for min_hours, max_hours, window_name in time_windows:
            print(f"\n--- {window_name.upper()} TIME WINDOW ({min_hours}-{max_hours}h) ---")
            
            # Run multi-distance analysis
            df, correlations = analyze_sigmoid_price_correlation_multi_distance(
                crypto=crypto, 
                min_hours=min_hours, 
                max_hours=max_hours
            )
            
            if df is not None and correlations is not None:
                # Create time-colored linear plot
                create_time_colored_linear_plot(df, correlations, crypto, min_hours, max_hours)
                
                # Store results
                all_results[crypto][window_name] = {
                    'df': df,
                    'correlations': correlations,
                    'time_window': (min_hours, max_hours)
                }
                
                # Quick asymmetric analysis
                print(f"\n--- Asymmetric Effects ({crypto.upper()}, {window_name}) ---")
                analyze_asymmetric_effects(df)
            else:
                print(f"No data available for {crypto} in {window_name} time window")
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for crypto in cryptocurrencies:
        for window_name in ["full", "mid-range"]:
            if window_name in all_results[crypto]:
                correlations = all_results[crypto][window_name]['correlations']
                linear_corr = correlations.get('distance_linear', {})
                if linear_corr:
                    print(f"{crypto.upper()} {window_name}: Linear correlation = {linear_corr.get('correlation', 0):.4f}, "
                          f"p = {linear_corr.get('p_value', 1):.4f}, n = {linear_corr.get('n_points', 0):,}")
    
    return all_results

def create_enhanced_linear_plot():
    """Quick function to generate the enhanced linear plot for both cryptos and time windows"""
    print("=== Creating Enhanced Linear Distance Plots (All Cryptos & Time Windows) ===")
    
    # Run the comprehensive analysis
    results = comprehensive_crypto_analysis()
    
    return results

if __name__ == "__main__":
    # Run the enhanced linear plot creation
    create_enhanced_linear_plot() 