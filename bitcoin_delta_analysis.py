import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import glob
import os
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap
import scipy.optimize as optimize

def create_analysis_folder():
    """Create the sinarb_analysis folder if it doesn't exist"""
    folder_path = Path("sinarb_analysis")
    folder_path.mkdir(exist_ok=True)
    return folder_path

def read_json_file(file_path):
    """Read JSON file with error handling"""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_crypto_data(file_path):
    """Extract delta price vs market price data from a crypto market JSON file"""
    data = read_json_file(file_path)
    if not data:
        return None
    
    # Get metadata
    metadata = data.get("collection_metadata", {})
    starting_price = metadata.get("starting_price")
    collection_period = metadata.get("collection_period", {})
    end_time_ts = collection_period.get("end")
    
    if starting_price is None or end_time_ts is None:
        print(f"Warning: Missing metadata in {file_path}")
        return None
    
    # Extract data points
    extracted_data = []
    
    for i, point in enumerate(data.get("data_points", [])):
        if not isinstance(point, dict):
            continue
            
        current_timestamp_ts = point.get('timestamp')
        current_actual_price = point.get('price')
        orderbook = point.get('orderbook')
        
        if current_timestamp_ts is None or current_actual_price is None or orderbook is None:
            continue
            
        # Calculate price delta as percentage
        price_delta_pct = ((current_actual_price - starting_price) / starting_price) * 100
        
        # Calculate hours remaining
        hours_remaining = (end_time_ts - current_timestamp_ts) / 3600.0
        hours_remaining = max(0, hours_remaining)
        
        # Get market price (midpoint of best bid and ask)
        bids_list = orderbook.get('bids', [])
        asks_list = orderbook.get('asks', [])
        
        if (bids_list and isinstance(bids_list, list) and 
            asks_list and isinstance(asks_list, list) and
            all(isinstance(b, dict) and 'price' in b for b in bids_list) and
            all(isinstance(a, dict) and 'price' in a for a in asks_list)):
            
            max_bid = max(bid['price'] for bid in bids_list)
            min_ask = min(ask['price'] for ask in asks_list)
            market_price = (max_bid + min_ask) / 2
            
            extracted_data.append({
                'timestamp': current_timestamp_ts,
                'datetime': datetime.fromtimestamp(current_timestamp_ts),
                'starting_price': starting_price,
                'current_price': current_actual_price,
                'price_delta_pct': price_delta_pct,
                'hours_remaining': hours_remaining,
                'market_price': market_price,
                'file_name': os.path.basename(file_path)
            })
    
    # Add market outcome based on last data point
    if extracted_data:
        final_price_delta = extracted_data[-1]['price_delta_pct']
        market_outcome = 1 if final_price_delta > 0 else 0
        for data_point in extracted_data:
            data_point['market_outcome'] = market_outcome
    
    return extracted_data

def extract_date_from_filename(filename):
    """Extract date from filename and convert to datetime object"""
    match = re.search(r'(?:bitcoin|ethereum)-up-or-down-on-(\w+)-(\d+)', filename)
    if match:
        month, day = match.groups()
        month_map = {'may': 5, 'june': 6}
        month_num = month_map.get(month.lower())
        if month_num:
            return datetime(2025, month_num, int(day))
    return None

def get_color_for_date(date, min_date, max_date):
    """Get color for a specific date using a gradient"""
    if min_date == max_date:
        return '#1f77b4'
    
    normalized = (date - min_date).total_seconds() / (max_date - min_date).total_seconds()
    colors = [(0, '#000066'), (0.5, '#0000FF'), (1, '#66B2FF')]
    cmap = LinearSegmentedColormap.from_list('custom_blues', colors)
    return cmap(normalized)

def categorize_hours_remaining_exact(hours):
    """Categorize hours remaining into exact hourly buckets (1-24h)"""
    if hours < 1:
        return "0-1h"
    elif hours >= 24:
        return "24h+"
    else:
        hour_bucket = int(np.ceil(hours))  # Round up to next hour
        return f"{hour_bucket-1}-{hour_bucket}h"

def collect_all_crypto_data(data_dir="collected_data"):
    """Collect data from all crypto market files"""
    btc_pattern = os.path.join(data_dir, "bitcoin-up-or-down-*.json")
    eth_pattern = os.path.join(data_dir, "ethereum-up-or-down-*.json")
    
    btc_files = glob.glob(btc_pattern)
    eth_files = glob.glob(eth_pattern)
    
    all_btc_data = []
    all_eth_data = []
    
    print(f"Found {len(btc_files)} Bitcoin and {len(eth_files)} Ethereum market files")
    
    for file_path in btc_files:
        print(f"Processing {os.path.basename(file_path)}...")
        file_data = extract_crypto_data(file_path)
        if file_data:
            for point in file_data:
                point['crypto'] = 'BTC'
            all_btc_data.extend(file_data)
    
    for file_path in eth_files:
        print(f"Processing {os.path.basename(file_path)}...")
        file_data = extract_crypto_data(file_path)
        if file_data:
            for point in file_data:
                point['crypto'] = 'ETH'
            all_eth_data.extend(file_data)
    
    btc_df = pd.DataFrame(all_btc_data)
    eth_df = pd.DataFrame(all_eth_data)
    
    return btc_df, eth_df

def create_triple_subplot_analysis(btc_df, eth_df, output_folder):
    """Create three subplots with different colorization schemes"""
    if btc_df.empty or eth_df.empty:
        print("No data to plot")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 24))
    
    # Subplot 1: Blue BTC, Green ETH
    ax1.scatter(btc_df['price_delta_pct'], btc_df['market_price'], 
               c='blue', alpha=0.6, s=20, label='BTC')
    ax1.scatter(eth_df['price_delta_pct'], eth_df['market_price'], 
               c='green', alpha=0.6, s=20, label='ETH')
    ax1.set_title('Crypto Comparison: Simple Color Scheme', fontsize=14)
    ax1.legend()
    
    # Subplot 2: Color by market date
    all_dates = pd.concat([
        btc_df['file_name'].apply(extract_date_from_filename),
        eth_df['file_name'].apply(extract_date_from_filename)
    ])
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    for crypto_df, marker in [(btc_df, 'o'), (eth_df, 's')]:
        for filename in sorted(crypto_df['file_name'].unique()):
            mask = crypto_df['file_name'] == filename
            data_subset = crypto_df[mask]
            date = extract_date_from_filename(filename)
            color = get_color_for_date(date, min_date, max_date)
            
            ax2.scatter(data_subset['price_delta_pct'], 
                       data_subset['market_price'],
                       c=[color],
                       marker=marker,
                       alpha=0.6,
                       s=20,
                       label=f"{'BTC' if marker=='o' else 'ETH'} {date.strftime('%b %d')}")
    
    ax2.set_title('Crypto Comparison: Colored by Market Date\nCircles=BTC, Squares=ETH', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    # Subplot 3: Color by market outcome
    outcome_colors = {
        ('BTC', 1): '#0000FF',  # Blue for BTC up
        ('BTC', 0): '#FF0000',  # Red for BTC down
        ('ETH', 1): '#00FF00',  # Green for ETH up
        ('ETH', 0): '#FF00FF'   # Purple for ETH down
    }
    
    for crypto_df in [btc_df, eth_df]:
        for outcome in [0, 1]:
            mask = (crypto_df['market_outcome'] == outcome)
            data_subset = crypto_df[mask]
            color = outcome_colors[(data_subset['crypto'].iloc[0], outcome)]
            
            ax3.scatter(data_subset['price_delta_pct'], 
                       data_subset['market_price'],
                       c=color,
                       alpha=0.6,
                       s=20,
                       label=f"{data_subset['crypto'].iloc[0]} {'Up' if outcome else 'Down'}")
    
    ax3.set_title('Crypto Comparison: Colored by Market Outcome', fontsize=14)
    ax3.legend()
    
    # Common settings for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Price Delta (% change from initial price)', fontsize=12)
        ax.set_ylabel('Market Price (Midpoint of Best Bid/Ask)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_folder / "crypto_comparison_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved as {plot_path}")

def create_hourly_bucket_analysis(btc_df, eth_df, output_folder):
    """Create hourly bucket analysis plots comparing BTC and ETH"""
    if btc_df.empty or eth_df.empty:
        print("No data to analyze")
        return
    
    # Add hour bucket category to both dataframes
    btc_df['hour_bucket'] = btc_df['hours_remaining'].apply(categorize_hours_remaining_exact)
    eth_df['hour_bucket'] = eth_df['hours_remaining'].apply(categorize_hours_remaining_exact)
    
    # Get all unique hour buckets and sort them
    hour_buckets = sorted(set(btc_df['hour_bucket'].unique()) | set(eth_df['hour_bucket'].unique()),
                         key=lambda x: 0 if x == "0-1h" else 25 if x == "24h+" else int(x.split('-')[1].replace('h', '')))
    
    print(f"\nCreating hourly analysis plots for {len(hour_buckets)} hour buckets...")
    
    for bucket in hour_buckets:
        btc_bucket_data = btc_df[btc_df['hour_bucket'] == bucket]
        eth_bucket_data = eth_df[eth_df['hour_bucket'] == bucket]
        
        if len(btc_bucket_data) > 0 or len(eth_bucket_data) > 0:
            create_hourly_triple_subplot(btc_bucket_data, eth_bucket_data, bucket, output_folder)
            print(f"  {bucket}: BTC: {len(btc_bucket_data)} points, ETH: {len(eth_bucket_data)} points")

def create_hourly_triple_subplot(btc_bucket_data, eth_bucket_data, bucket_name, output_folder):
    """Create three subplots for a specific hour bucket"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 24))
    
    # Subplot 1: Blue BTC, Green ETH
    if not btc_bucket_data.empty:
        ax1.scatter(btc_bucket_data['price_delta_pct'], btc_bucket_data['market_price'], 
                   c='blue', alpha=0.6, s=20, label=f'BTC ({len(btc_bucket_data)} points)')
    if not eth_bucket_data.empty:
        ax1.scatter(eth_bucket_data['price_delta_pct'], eth_bucket_data['market_price'], 
                   c='green', alpha=0.6, s=20, label=f'ETH ({len(eth_bucket_data)} points)')
    ax1.set_title(f'Hour Bucket {bucket_name}: Simple Color Scheme', fontsize=14)
    ax1.legend()
    
    # Subplot 2: Color by market date
    all_dates = pd.concat([
        btc_bucket_data['file_name'].apply(extract_date_from_filename),
        eth_bucket_data['file_name'].apply(extract_date_from_filename)
    ])
    min_date = min(all_dates)
    max_date = max(all_dates)
    
    for crypto_df, marker in [(btc_bucket_data, 'o'), (eth_bucket_data, 's')]:
        if not crypto_df.empty:
            for filename in sorted(crypto_df['file_name'].unique()):
                mask = crypto_df['file_name'] == filename
                data_subset = crypto_df[mask]
                date = extract_date_from_filename(filename)
                color = get_color_for_date(date, min_date, max_date)
                
                ax2.scatter(data_subset['price_delta_pct'], 
                           data_subset['market_price'],
                           c=[color],
                           marker=marker,
                           alpha=0.6,
                           s=20,
                           label=f"{'BTC' if marker=='o' else 'ETH'} {date.strftime('%b %d')}")
    
    ax2.set_title(f'Hour Bucket {bucket_name}: Colored by Market Date\nCircles=BTC, Squares=ETH', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    
    # Subplot 3: Color by market outcome
    outcome_colors = {
        ('BTC', 1): '#0000FF',  # Blue for BTC up
        ('BTC', 0): '#FF0000',  # Red for BTC down
        ('ETH', 1): '#00FF00',  # Green for ETH up
        ('ETH', 0): '#FF00FF'   # Purple for ETH down
    }
    
    for crypto_df in [btc_bucket_data, eth_bucket_data]:
        if not crypto_df.empty:
            for outcome in [0, 1]:
                mask = (crypto_df['market_outcome'] == outcome)
                data_subset = crypto_df[mask]
                if not data_subset.empty:
                    color = outcome_colors[(data_subset['crypto'].iloc[0], outcome)]
                    
                    ax3.scatter(data_subset['price_delta_pct'], 
                               data_subset['market_price'],
                               c=color,
                               alpha=0.6,
                               s=20,
                               label=f"{data_subset['crypto'].iloc[0]} {'Up' if outcome else 'Down'}")
    
    ax3.set_title(f'Hour Bucket {bucket_name}: Colored by Market Outcome', fontsize=14)
    ax3.legend()
    
    # Common settings for all subplots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('Price Delta (% change from initial price)', fontsize=12)
        ax.set_ylabel('Market Price (Midpoint of Best Bid/Ask)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add correlation info if data exists
        if not btc_bucket_data.empty:
            btc_corr = btc_bucket_data['price_delta_pct'].corr(btc_bucket_data['market_price'])
            ax.text(0.02, 0.98, f'BTC Correlation: {btc_corr:.4f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   color='blue',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if not eth_bucket_data.empty:
            eth_corr = eth_bucket_data['price_delta_pct'].corr(eth_bucket_data['market_price'])
            ax.text(0.02, 0.93, f'ETH Correlation: {eth_corr:.4f}', 
                   transform=ax.transAxes, 
                   verticalalignment='top',
                   color='green',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"crypto_hour_bucket_{bucket_name.replace('-', '_').replace('+', '_plus')}_analysis.png"
    plot_filepath = output_folder / plot_filename
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

def sigmoid(x, L, k, x0, b):
    """
    Generalized sigmoid function:
    L: maximum value
    k: steepness
    x0: x-value of sigmoid midpoint
    b: y offset
    """
    return L / (1 + np.exp(-k * (x - x0))) + b

def fit_market_data(df, crypto_type):
    """
    Fit sigmoid function to market data for a specific crypto type
    Returns the fitted parameters and R-squared value
    """
    if df.empty:
        return None, None, 0
    
    try:
        # Initial parameter guesses
        p0 = [1.0, 1.0, 0.0, 0.0]  # [L, k, x0, b]
        
        # Fit the sigmoid function
        popt, _ = optimize.curve_fit(sigmoid, 
                                   df['price_delta_pct'], 
                                   df['market_price'],
                                   p0=p0,
                                   maxfev=10000)
        
        # Calculate R-squared
        y_pred = sigmoid(df['price_delta_pct'], *popt)
        ss_res = np.sum((df['market_price'] - y_pred) ** 2)
        ss_tot = np.sum((df['market_price'] - df['market_price'].mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return popt, y_pred, r_squared
        
    except Exception as e:
        print(f"Fitting failed for {crypto_type}: {str(e)}")
        return None, None, 0

def create_hourly_fit_analysis(btc_df, eth_df, output_folder):
    """Create fitted analysis plots for each hour bucket"""
    # Add hour bucket category to both dataframes
    btc_df['hour_bucket'] = btc_df['hours_remaining'].apply(categorize_hours_remaining_exact)
    eth_df['hour_bucket'] = eth_df['hours_remaining'].apply(categorize_hours_remaining_exact)
    
    # Get all unique hour buckets and sort them
    hour_buckets = sorted(set(btc_df['hour_bucket'].unique()) | set(eth_df['hour_bucket'].unique()),
                         key=lambda x: 0 if x == "0-1h" else 25 if x == "24h+" else int(x.split('-')[1].replace('h', '')))
    
    # Store fitting results
    fitting_results = []
    
    for bucket in hour_buckets:
        btc_bucket_data = btc_df[btc_df['hour_bucket'] == bucket]
        eth_bucket_data = eth_df[eth_df['hour_bucket'] == bucket]
        
        # Fit data
        btc_params, btc_pred, btc_r2 = fit_market_data(btc_bucket_data, 'BTC')
        eth_params, eth_pred, eth_r2 = fit_market_data(eth_bucket_data, 'ETH')
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot original data
        if not btc_bucket_data.empty:
            plt.scatter(btc_bucket_data['price_delta_pct'], btc_bucket_data['market_price'],
                       c='blue', alpha=0.3, label=f'BTC Data (n={len(btc_bucket_data)})')
        if not eth_bucket_data.empty:
            plt.scatter(eth_bucket_data['price_delta_pct'], eth_bucket_data['market_price'],
                       c='green', alpha=0.3, label=f'ETH Data (n={len(eth_bucket_data)})')
        
        # Plot fitted curves
        if btc_params is not None:
            x_fit = np.linspace(btc_bucket_data['price_delta_pct'].min(),
                              btc_bucket_data['price_delta_pct'].max(), 1000)
            y_fit = sigmoid(x_fit, *btc_params)
            plt.plot(x_fit, y_fit, 'b-', linewidth=2,
                    label=f'BTC Fit (R²={btc_r2:.4f})')
            
            # Store results
            fitting_results.append({
                'hour_bucket': bucket,
                'crypto': 'BTC',
                'L': btc_params[0],
                'k': btc_params[1],
                'x0': btc_params[2],
                'b': btc_params[3],
                'r_squared': btc_r2,
                'n_points': len(btc_bucket_data)
            })
        
        if eth_params is not None:
            x_fit = np.linspace(eth_bucket_data['price_delta_pct'].min(),
                              eth_bucket_data['price_delta_pct'].max(), 1000)
            y_fit = sigmoid(x_fit, *eth_params)
            plt.plot(x_fit, y_fit, 'g-', linewidth=2,
                    label=f'ETH Fit (R²={eth_r2:.4f})')
            
            # Store results
            fitting_results.append({
                'hour_bucket': bucket,
                'crypto': 'ETH',
                'L': eth_params[0],
                'k': eth_params[1],
                'x0': eth_params[2],
                'b': eth_params[3],
                'r_squared': eth_r2,
                'n_points': len(eth_bucket_data)
            })
        
        plt.title(f'Hour Bucket {bucket}: Fitted Sigmoid Functions')
        plt.xlabel('Price Delta (% change from initial price)')
        plt.ylabel('Market Price (Midpoint of Best Bid/Ask)')
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.legend()
        
        # Save plot
        plot_filename = f"crypto_hour_bucket_{bucket.replace('-', '_').replace('+', '_plus')}_fitted.png"
        plt.savefig(output_folder / plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save fitting results to CSV
    results_df = pd.DataFrame(fitting_results)
    results_df.to_csv(output_folder / "fitting_results.csv", index=False)
    
    return results_df

def create_analysis_folders():
    """Create the analysis folders structure"""
    base_folder = Path("sinarb_analysis")
    btc_folder = base_folder / "bitcoin"
    eth_folder = base_folder / "ethereum"
    
    # Create folders
    base_folder.mkdir(exist_ok=True)
    btc_folder.mkdir(exist_ok=True)
    eth_folder.mkdir(exist_ok=True)
    
    return btc_folder, eth_folder

def create_crypto_analysis(df, crypto_type, output_folder, hour_bucket, outlier_counts):
    """Create analysis for a specific cryptocurrency"""
    plt.figure(figsize=(15, 10))
    
    # Fit sigmoid function
    popt, y_pred, r_squared = fit_market_data(df, crypto_type)
    
    # Calculate residuals and find outliers
    if not df.empty:
        residuals = df['market_price'] - y_pred
        outliers = df[abs(residuals) > 2 * residuals.std()]
        
        # Update outlier counts
        for filename in outliers['file_name'].unique():
            count = len(outliers[outliers['file_name'] == filename])
            if filename not in outlier_counts:
                outlier_counts[filename] = {
                    'total_outliers': 0,
                    'total_points': 0,
                    'hour_buckets': set()
                }
            outlier_counts[filename]['total_outliers'] += count
            outlier_counts[filename]['total_points'] += len(df[df['file_name'] == filename])
            outlier_counts[filename]['hour_buckets'].add(hour_bucket)
        
        # Plot data points colored by source file
        unique_files = df['file_name'].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_files)))
        
        for filename, color in zip(unique_files, colors):
            mask = df['file_name'] == filename
            date = extract_date_from_filename(filename)
            plt.scatter(df[mask]['price_delta_pct'], 
                       df[mask]['market_price'],
                       c=[color],
                       alpha=0.6, 
                       s=20,
                       label=f"{date.strftime('%b %d')}")
        
        # Plot fitted curve
        x_fit = np.linspace(df['price_delta_pct'].min(),
                           df['price_delta_pct'].max(), 1000)
        y_fit = sigmoid(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'k-', linewidth=2,
                label=f'Fitted Curve (R²={r_squared:.4f})')
        
        # Plot outliers with different marker
        plt.scatter(outliers['price_delta_pct'],
                   outliers['market_price'],
                   c='red',
                   marker='x',
                   s=100,
                   label='Outliers')
        
        # Print outlier summary for this hour bucket
        print(f"\n{crypto_type} Hour Bucket {hour_bucket} Outlier Summary:")
        print(f"Total points: {len(df)}")
        print(f"Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
        print("\nOutliers by date:")
        print(outliers['file_name'].value_counts())
        
    plt.title(f'{crypto_type} Hour Bucket {hour_bucket}')
    plt.xlabel('Price Delta (% change from initial price)')
    plt.ylabel('Market Price (Midpoint of Best Bid/Ask)')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot
    plt.savefig(output_folder / f"{crypto_type.lower()}_hour_{hour_bucket}_analysis.png",
                dpi=300, bbox_inches='tight')
    plt.close()

def print_outlier_summary(outlier_counts, crypto_type):
    """Print summary of outliers across all hour buckets"""
    print(f"\n{'='*50}")
    print(f"{crypto_type} OVERALL OUTLIER SUMMARY")
    print(f"{'='*50}")
    
    # Sort by total outliers percentage
    sorted_files = sorted(
        outlier_counts.items(),
        key=lambda x: x[1]['total_outliers'] / x[1]['total_points'],
        reverse=True
    )
    
    for filename, stats in sorted_files:
        date = extract_date_from_filename(filename)
        percentage = (stats['total_outliers'] / stats['total_points']) * 100
        print(f"\n{date.strftime('%b %d')}:")
        print(f"  Total outliers: {stats['total_outliers']}")
        print(f"  Total points: {stats['total_points']}")
        print(f"  Percentage: {percentage:.2f}%")
        print(f"  Affected hour buckets: {len(stats['hour_buckets'])}")

def create_outlier_comparison_graph(btc_outlier_counts, eth_outlier_counts):
    """Create a bar graph comparing BTC and ETH outliers by date"""
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    # Collect all unique dates
    all_dates = set()
    for counts in [btc_outlier_counts, eth_outlier_counts]:
        for filename in counts.keys():
            date = extract_date_from_filename(filename)
            if date:
                all_dates.add(date)
    
    # Sort dates chronologically
    dates = sorted(list(all_dates))
    
    # Prepare data for plotting
    btc_outliers = []
    eth_outliers = []
    date_labels = [d.strftime('%b %d') for d in dates]
    
    for date in dates:
        # Get BTC outliers for this date
        btc_count = 0
        for filename, stats in btc_outlier_counts.items():
            if extract_date_from_filename(filename) == date:
                btc_count = stats['total_outliers']
                break
        btc_outliers.append(btc_count)
        
        # Get ETH outliers for this date
        eth_count = 0
        for filename, stats in eth_outlier_counts.items():
            if extract_date_from_filename(filename) == date:
                eth_count = stats['total_outliers']
                break
        eth_outliers.append(eth_count)
    
    # Create the bar graph
    plt.figure(figsize=(15, 8))
    x = np.arange(len(dates))
    width = 0.35
    
    plt.bar(x - width/2, btc_outliers, width, label='BTC', color='blue', alpha=0.6)
    plt.bar(x + width/2, eth_outliers, width, label='ETH', color='green', alpha=0.6)
    
    plt.xlabel('Date')
    plt.ylabel('Number of Outliers')
    plt.title('Outliers by Date: BTC vs ETH')
    plt.xticks(x, date_labels, rotation=45, ha='right')
    plt.legend()
    
    # Add value labels on top of bars
    for i, v in enumerate(btc_outliers):
        if v > 0:
            plt.text(i - width/2, v, str(v), ha='center', va='bottom')
    for i, v in enumerate(eth_outliers):
        if v > 0:
            plt.text(i + width/2, v, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('outliers_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_separated_hourly_interpolations():
    """Create two graphs - one for BTC and one for ETH hourly interpolations"""
    # Collect all data
    btc_df, eth_df = collect_all_crypto_data()
    
    # Add hour bucket categorization
    btc_df['hour_bucket'] = btc_df['hours_remaining'].apply(categorize_hours_remaining_exact)
    eth_df['hour_bucket'] = eth_df['hours_remaining'].apply(categorize_hours_remaining_exact)
    
    # Sort hour buckets
    hour_buckets = sorted(set(btc_df['hour_bucket'].unique()) | set(eth_df['hour_bucket'].unique()),
                         key=lambda x: 0 if x == "0-1h" else 25 if x == "24h+" else int(x.split('-')[1].replace('h', '')))
    
    # Define colors for market outcomes
    up_color = '#00FF00'    # Green for up
    down_color = '#FF0000'  # Red for down
    
    # Common x-range for all curves
    x_range = np.linspace(-10, 10, 1000)
    
    # Plot BTC
    plt.figure(figsize=(40, 20))
    for hour_bucket in hour_buckets:
        btc_bucket_data = btc_df[btc_df['hour_bucket'] == hour_bucket]
        if not btc_bucket_data.empty:
            # Get market outcome for this bucket
            market_outcome = btc_bucket_data['market_outcome'].iloc[0]
            color = up_color if market_outcome == 1 else down_color
            
            btc_params, _, r2 = fit_market_data(btc_bucket_data, 'BTC')
            if btc_params is not None:
                y_fit = sigmoid(x_range, *btc_params)
                plt.plot(x_range, y_fit, '-', color=color, alpha=0.8, linewidth=2,
                        label=f'{hour_bucket} (R²={r2:.3f}) {"↑" if market_outcome == 1 else "↓"}')
    
    plt.title('BTC Hourly Sigmoid Fits (Green=Up, Red=Down)')
    plt.xlabel('Price Delta (%)')
    plt.ylabel('Market Price')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.ylim(-0.0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('btc_hourly_fits.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot ETH
    plt.figure(figsize=(40, 20))
    for hour_bucket in hour_buckets:
        eth_bucket_data = eth_df[eth_df['hour_bucket'] == hour_bucket]
        if not eth_bucket_data.empty:
            # Get market outcome for this bucket
            market_outcome = eth_bucket_data['market_outcome'].iloc[0]
            color = up_color if market_outcome == 1 else down_color
            
            eth_params, _, r2 = fit_market_data(eth_bucket_data, 'ETH')
            if eth_params is not None:
                y_fit = sigmoid(x_range, *eth_params)
                plt.plot(x_range, y_fit, '-', color=color, alpha=0.8, linewidth=2,
                        label=f'{hour_bucket} (R²={r2:.3f}) {"↑" if market_outcome == 1 else "↓"}')
    
    plt.title('ETH Hourly Sigmoid Fits (Green=Up, Red=Down)')
    plt.xlabel('Price Delta (%)')
    plt.ylabel('Market Price')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.ylim(-0.0, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('eth_hourly_fits.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_sigmoid_residuals(crypto_type):
    """Analyze points above and below sigmoid curve and perform calibration"""
    print(f"\nAnalyzing {crypto_type} residuals...")
    
    # Collect data
    btc_df, eth_df = collect_all_crypto_data()
    df = btc_df if crypto_type == 'BTC' else eth_df
    
    if df.empty:
        print(f"No data found for {crypto_type}")
        return
    
    # Fit sigmoid to all data
    params, y_pred, r2 = fit_market_data(df, crypto_type)
    
    if params is None:
        print(f"Could not fit sigmoid for {crypto_type}")
        return
    
    # Calculate residuals
    residuals = df['market_price'] - y_pred
    
    # Split data into above and below sigmoid
    above_mask = residuals > 0
    below_mask = residuals <= 0
    
    df_above = df[above_mask].copy()
    df_below = df[below_mask].copy()
    
    print(f"\n{crypto_type} Data Split:")
    print(f"Total points: {len(df)}")
    print(f"Points above sigmoid: {len(df_above)} ({len(df_above)/len(df)*100:.1f}%)")
    print(f"Points below sigmoid: {len(df_below)} ({len(df_below)/len(df)*100:.1f}%)")
    
    # Create calibration data for each subset
    def prepare_calibration_data(data):
        calibration_data = []
        for _, row in data.iterrows():
            calibration_data.append((row['market_price'], row['market_outcome']))
        return calibration_data
    
    all_data = prepare_calibration_data(df)
    above_data = prepare_calibration_data(df_above)
    below_data = prepare_calibration_data(df_below)
    
    # Plot calibration curves
    plt.figure(figsize=(15, 10))
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration", linewidth=2)
    
    # Function to calculate calibration curve points
    def calculate_calibration_points(data, n_bins=10):
        if not data:
            return [], []
        
        y_prob = np.array([p[0] for p in data])
        y_true = np.array([p[1] for p in data])
        
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
        binids = np.digitize(y_prob, bins) - 1
        
        mean_predicted_value = []
        fraction_of_positives = []
        
        for i in range(n_bins):
            bin_mask = (binids == i)
            if np.sum(bin_mask) > 0:
                mean_predicted_value.append(np.mean(y_prob[bin_mask]))
                fraction_of_positives.append(np.mean(y_true[bin_mask]))
        
        return mean_predicted_value, fraction_of_positives
    
    # Plot calibration curves for each subset
    x_all, y_all = calculate_calibration_points(all_data)
    x_above, y_above = calculate_calibration_points(above_data)
    x_below, y_below = calculate_calibration_points(below_data)
    
    plt.plot(x_all, y_all, "b-", label=f"All points (n={len(df)})", linewidth=2)
    plt.plot(x_above, y_above, "g-", label=f"Above sigmoid (n={len(df_above)})", linewidth=2)
    plt.plot(x_below, y_below, "r-", label=f"Below sigmoid (n={len(df_below)})", linewidth=2)
    
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"{crypto_type} Calibration Analysis - Points Above/Below Sigmoid")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save plot
    plt.savefig(f'{crypto_type.lower()}_calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data points for further analysis
    output_dir = f"{crypto_type.lower()}_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    df_above.to_csv(f"{output_dir}/above_sigmoid.csv", index=False)
    df_below.to_csv(f"{output_dir}/below_sigmoid.csv", index=False)
    
    print(f"\nAnalysis complete for {crypto_type}. Files saved in {output_dir}/")

def create_normalized_market_analysis_with_correlation(crypto_type='BTC'):
    """
    Create normalized analysis for each market/JSON file with correlation analysis
    and a recap graph showing all markets together.
    
    Args:
        crypto_type (str): 'BTC' or 'ETH'
    """
    # Collect data
    btc_df, eth_df = collect_all_crypto_data()
    df = btc_df if crypto_type == 'BTC' else eth_df
    
    if df.empty:
        print(f"No data found for {crypto_type}")
        return
    
    # Add hour bucket categorization
    df['hour_bucket'] = df['hours_remaining'].apply(categorize_hours_remaining_exact)
    
    # Calculate sigmoid fits for each hour bucket
    hour_bucket_fits = {}
    for hour_bucket in sorted(df['hour_bucket'].unique()):
        bucket_data = df[df['hour_bucket'] == hour_bucket]
        params, _, r2 = fit_market_data(bucket_data, crypto_type)
        if params is not None:
            hour_bucket_fits[hour_bucket] = params
    
    # Store all normalized data for recap
    all_normalized_data = []
    market_correlations = []
    
    # Create a figure for each unique market
    for json_file in df['file_name'].unique():
        market_data = df[df['file_name'] == json_file].copy()
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        
        # Calculate normalized values for each point
        normalized_points = []
        
        for _, row in market_data.iterrows():
            hour_bucket = row['hour_bucket']
            if hour_bucket in hour_bucket_fits:
                params = hour_bucket_fits[hour_bucket]
                expected_value = sigmoid(row['price_delta_pct'], *params)
                normalized_value = row['market_price'] - expected_value
                
                market_date = extract_date_from_filename(json_file)
                point_data = {
                    'delta': row['price_delta_pct'],
                    'normalized_value': normalized_value,
                    'hours_remaining': row['hours_remaining'],
                    'hour_bucket': hour_bucket,
                    'market_date': market_date.strftime('%b %d') if market_date else 'Unknown'
                }
                normalized_points.append(point_data)
                all_normalized_data.append(point_data)
        
        if normalized_points:
            norm_df = pd.DataFrame(normalized_points)
            
            # Calculate correlation
            correlation = norm_df['delta'].corr(norm_df['normalized_value'])
            market_date = extract_date_from_filename(json_file)
            market_correlations.append({
                'market_date': market_date.strftime('%b %d') if market_date else 'Unknown',
                'correlation': correlation,
                'n_points': len(norm_df),
                'raw_date': market_date  # Keep raw date for sorting
            })
            
            # Create scatter plot
            scatter = plt.scatter(norm_df['delta'], 
                                norm_df['normalized_value'],
                                c=norm_df['hours_remaining'],
                                cmap='viridis',
                                alpha=0.6)
            
            plt.colorbar(scatter, label='Hours Remaining')
            
            # Add reference lines
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            
            # Add correlation line
            z = np.polyfit(norm_df['delta'], norm_df['normalized_value'], 1)
            p = np.poly1d(z)
            plt.plot(norm_df['delta'], p(norm_df['delta']), "r--", alpha=0.8)
            
            date_str = market_date.strftime('%b %d') if market_date else 'Unknown'
            
            plt.title(f'{crypto_type} Market Analysis - {date_str}\nNormalized Against Hourly Sigmoids')
            plt.xlabel('Price Delta (%)')
            plt.ylabel('Market Price - Expected Price (Normalized)')
            plt.grid(True, alpha=0.3)
            
            # Add statistics including correlation
            stats_text = (
                f'Total Points: {len(norm_df)}\n'
                f'Mean Deviation: {norm_df["normalized_value"].mean():.4f}\n'
                f'Std Deviation: {norm_df["normalized_value"].std():.4f}\n'
                f'Correlation: {correlation:.4f}'
            )
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save individual market plot
            output_dir = f'{crypto_type.lower()}_normalized_analysis'
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(
                f'{output_dir}/{crypto_type.lower()}_{date_str.replace(" ", "_")}_normalized.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()
    
    # Create recap graph
    if all_normalized_data:
        plt.figure(figsize=(20, 12))
        
        # Convert all data to DataFrame
        all_norm_df = pd.DataFrame(all_normalized_data)
        
        # Create scatter plot with different color for each market date
        unique_dates = sorted(all_norm_df['market_date'].unique())
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_dates)))
        
        for date, color in zip(unique_dates, colors):
            mask = all_norm_df['market_date'] == date
            date_data = all_norm_df[mask]
            plt.scatter(date_data['delta'], 
                       date_data['normalized_value'],
                       c=[color],
                       alpha=0.6,
                       label=date)
        
        # Calculate and plot overall correlation
        overall_corr = all_norm_df['delta'].corr(all_norm_df['normalized_value'])
        z = np.polyfit(all_norm_df['delta'], all_norm_df['normalized_value'], 1)
        p = np.poly1d(z)
        plt.plot(all_norm_df['delta'], p(all_norm_df['delta']), "k--", 
                alpha=0.8, label=f'Overall Correlation: {overall_corr:.4f}')
        
        plt.title(f'{crypto_type} All Markets Normalized Analysis')
        plt.xlabel('Price Delta (%)')
        plt.ylabel('Market Price - Expected Price (Normalized)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add correlation summary
        corr_df = pd.DataFrame(market_correlations)
        corr_summary = (
            f'Overall Correlation: {overall_corr:.4f}\n'
            f'Mean Market Correlation: {corr_df["correlation"].mean():.4f}\n'
            f'Std Market Correlation: {corr_df["correlation"].std():.4f}\n'
            f'Total Points: {len(all_norm_df)}'
        )
        plt.text(0.02, 0.98, corr_summary,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save recap plot
        plt.savefig(
            f'{crypto_type.lower()}_normalized_analysis/{crypto_type.lower()}_all_markets_recap.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        
        # Print correlation summary
        print(f"\n{crypto_type} Correlation Summary:")
        print("=" * 40)
        print(f"Overall correlation: {overall_corr:.4f}")
        print("\nCorrelations by market:")
        # Sort by raw date for chronological order
        for corr_data in sorted(market_correlations, key=lambda x: x['raw_date']):
            print(f"{corr_data['market_date']}: "
                  f"{corr_data['correlation']:.4f} "
                  f"(n={corr_data['n_points']})")

def main():
    """Main function to run the normalized market analysis with correlation"""
    print("Starting Normalized Market Analysis with Correlation...")
    
    # Run calibration analysis for both BTC and ETH
    print("\nRunning Calibration Analysis...")
    analyze_sigmoid_residuals('BTC')
    analyze_sigmoid_residuals('ETH')
    
    # Run normalized market analysis
    print("\nRunning Normalized Market Analysis...")
    create_normalized_market_analysis_with_correlation('BTC')
    create_normalized_market_analysis_with_correlation('ETH')
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()