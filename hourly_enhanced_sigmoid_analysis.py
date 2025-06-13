import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import os
from datetime import datetime

# Import functions from the existing enhanced_sigmoid_analysis.py
from enhanced_sigmoid_analysis import (
    collect_market_files,
    read_json_file,
    get_next_market_starting_price,
    calculate_hours_remaining,
    get_market_midpoint_probability,
    get_btc_price_from_point,
    sigmoid,
    fit_hourly_sigmoid
)

def analyze_asymmetric_effects_hourly(df):
    """Analyze if positive and negative deviations have different effects - adapted for hourly analysis"""
    print("\n=== Asymmetric Effects Analysis ===")
    
    # Split into positive and negative deviations using the correct column name
    positive_deviations = df[df['linear_distance'] > 0]
    negative_deviations = df[df['linear_distance'] < 0]
    
    if len(positive_deviations) > 100 and len(negative_deviations) > 100:
        corr_pos, p_pos = pearsonr(positive_deviations['linear_distance'], positive_deviations['price_evolution'])
        corr_neg, p_neg = pearsonr(negative_deviations['linear_distance'], negative_deviations['price_evolution'])
        
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

def analyze_hourly_linear_correlation(data_dir="collected_data", crypto="bitcoin", min_hours=0, max_hours=24):
    """Analyze linear correlation and weighted mean per hour for a specific cryptocurrency"""
    print(f"=== Hourly Linear Correlation Analysis ({crypto.upper()}, {min_hours}-{max_hours}h) ===\n")
    
    # Collect and sort market files
    sorted_files = collect_market_files(data_dir, crypto)
    print(f"Found {len(sorted_files)} {crypto} market files")
    
    correlation_data = []
    hourly_stats = {}
    valid_markets = []
    invalid_markets = []
    
    # Initialize hourly stats dictionary
    for hour in range(max_hours + 1):
        if hour >= min_hours:
            hourly_stats[hour] = {
                'data_points': [],
                'correlations': [],
                'weighted_means': [],
                'n_markets': 0
            }
    
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
        
        # Second pass: collect data for hourly analysis
        market_hourly_data = {}  # Store data by hour for this market
        
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
            current_crypto_price = get_btc_price_from_point(point)
            
            if current_prob is None or current_crypto_price is None:
                continue
            
            expected_prob = sigmoid(hours_remaining, *sigmoid_params)
            price_evolution = (next_starting_price - current_crypto_price) / current_crypto_price
            linear_distance = current_prob - expected_prob
            
            hour_bucket = int(hours_remaining)
            
            # Store for overall correlation data
            correlation_data.append({
                'market': filename,
                'crypto': crypto,
                'timestamp': timestamp,
                'hours_remaining': hours_remaining,
                'hour_bucket': hour_bucket,
                'current_prob': current_prob,
                'expected_prob': expected_prob,
                'price_evolution': price_evolution,
                'linear_distance': linear_distance,
                'current_crypto_price': current_crypto_price,
                'end_crypto_price': next_starting_price,
            })
            
            # Store for market-level hourly analysis
            if hour_bucket not in market_hourly_data:
                market_hourly_data[hour_bucket] = []
            market_hourly_data[hour_bucket].append({
                'linear_distance': linear_distance,
                'price_evolution': price_evolution
            })
        
        # Calculate correlations and weighted means for this market by hour
        for hour, hour_data in market_hourly_data.items():
            if len(hour_data) >= 10:  # Minimum points for correlation
                distances = [d['linear_distance'] for d in hour_data]
                evolutions = [d['price_evolution'] for d in hour_data]
                
                try:
                    corr, p_val = pearsonr(distances, evolutions)
                    if not np.isnan(corr):
                        hourly_stats[hour]['correlations'].append(corr)
                        
                        # Calculate weighted mean for this hour in this market
                        abs_evolutions = [abs(e) for e in evolutions]
                        weighted_mean = np.mean(abs_evolutions)
                        hourly_stats[hour]['weighted_means'].append(weighted_mean)
                        
                        hourly_stats[hour]['data_points'].extend(hour_data)
                except:
                    pass
        
        # Count this as a valid market for each hour that had data
        for hour in market_hourly_data.keys():
            if len(market_hourly_data[hour]) >= 10:
                hourly_stats[hour]['n_markets'] += 1
        
        valid_markets.append((filename, len([d for d in correlation_data if d['market'] == filename])))
    
    if not correlation_data:
        print(f"\nNo valid correlation data found for {crypto}!")
        return None, None
    
    # Calculate overall hourly statistics
    hourly_results = {}
    for hour in hourly_stats:
        stats = hourly_stats[hour]
        if len(stats['correlations']) >= 3:  # At least 3 markets
            avg_correlation = np.mean(stats['correlations'])
            std_correlation = np.std(stats['correlations'])
            avg_weighted_mean = np.mean(stats['weighted_means'])
            std_weighted_mean = np.std(stats['weighted_means'])
            
            # Overall correlation for this hour across all data
            hour_data = stats['data_points']
            if len(hour_data) >= 50:
                all_distances = [d['linear_distance'] for d in hour_data]
                all_evolutions = [d['price_evolution'] for d in hour_data]
                overall_corr, overall_p = pearsonr(all_distances, all_evolutions)
            else:
                overall_corr, overall_p = np.nan, np.nan
            
            hourly_results[hour] = {
                'avg_correlation': avg_correlation,
                'std_correlation': std_correlation,
                'avg_weighted_mean': avg_weighted_mean,
                'std_weighted_mean': std_weighted_mean,
                'overall_correlation': overall_corr,
                'overall_p_value': overall_p,
                'n_markets': stats['n_markets'],
                'n_total_points': len(hour_data)
            }
    
    print(f"\n=== Hourly Results ({crypto.upper()}, {min_hours}-{max_hours}h) ===")
    print(f"Total datapoints: {len(correlation_data):,}")
    print(f"Valid markets: {len(valid_markets)}")
    print(f"Invalid markets: {len(invalid_markets)}")
    print(f"\nHour | Avg Corr | Std Corr | Avg W.Mean | Std W.Mean | Overall Corr | P-Value | Markets | Points")
    print("-" * 100)
    
    for hour in sorted(hourly_results.keys()):
        result = hourly_results[hour]
        print(f"{hour:4d} | {result['avg_correlation']:8.4f} | {result['std_correlation']:8.4f} | "
              f"{result['avg_weighted_mean']:10.4f} | {result['std_weighted_mean']:10.4f} | "
              f"{result['overall_correlation']:12.4f} | {result['overall_p_value']:7.4f} | "
              f"{result['n_markets']:7d} | {result['n_total_points']:6d}")
    
    return pd.DataFrame(correlation_data), hourly_results

def create_hourly_analysis_plots(df, hourly_results, crypto="bitcoin", min_hours=0, max_hours=24):
    """Create comprehensive hourly analysis visualizations"""
    
    if df is None or not hourly_results:
        print("No data available for plotting")
        return
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Hourly Analysis for {crypto.upper()} ({min_hours}-{max_hours}h)', fontsize=16, y=0.98)
    
    hours = sorted(hourly_results.keys())
    
    # 1. Average correlation by hour
    ax1 = axes[0, 0]
    avg_corrs = [hourly_results[h]['avg_correlation'] for h in hours]
    std_corrs = [hourly_results[h]['std_correlation'] for h in hours]
    
    ax1.errorbar(hours, avg_corrs, yerr=std_corrs, marker='o', capsize=5, capthick=2, linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Hours Remaining')
    ax1.set_ylabel('Average Correlation')
    ax1.set_title('Average Linear Correlation by Hour')
    ax1.grid(True, alpha=0.3)
    
    # 2. Average weighted mean by hour
    ax2 = axes[0, 1]
    avg_wmeans = [hourly_results[h]['avg_weighted_mean'] for h in hours]
    std_wmeans = [hourly_results[h]['std_weighted_mean'] for h in hours]
    
    ax2.errorbar(hours, avg_wmeans, yerr=std_wmeans, marker='s', capsize=5, capthick=2, linewidth=2, color='orange')
    ax2.set_xlabel('Hours Remaining')
    ax2.set_ylabel('Average Weighted Mean')
    ax2.set_title('Average Weighted |Mean| by Hour')
    ax2.grid(True, alpha=0.3)
    
    # 3. Overall correlation by hour
    ax3 = axes[0, 2]
    overall_corrs = [hourly_results[h]['overall_correlation'] for h in hours]
    
    ax3.plot(hours, overall_corrs, marker='D', linewidth=2, markersize=8, color='green')
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Hours Remaining')
    ax3.set_ylabel('Overall Correlation')
    ax3.set_title('Overall Linear Correlation by Hour')
    ax3.grid(True, alpha=0.3)
    
    # 4. Number of markets and data points by hour
    ax4 = axes[1, 0]
    n_markets = [hourly_results[h]['n_markets'] for h in hours]
    n_points = [hourly_results[h]['n_total_points'] for h in hours]
    
    ax4_twin = ax4.twinx()
    ax4.bar([h - 0.2 for h in hours], n_markets, width=0.4, alpha=0.7, label='Markets', color='steelblue')
    ax4_twin.bar([h + 0.2 for h in hours], n_points, width=0.4, alpha=0.7, label='Data Points', color='lightcoral')
    
    ax4.set_xlabel('Hours Remaining')
    ax4.set_ylabel('Number of Markets', color='steelblue')
    ax4_twin.set_ylabel('Number of Data Points', color='lightcoral')
    ax4.set_title('Data Availability by Hour')
    ax4.grid(True, alpha=0.3)
    
    # 5. Scatter plot of correlation vs weighted mean
    ax5 = axes[1, 1]
    scatter_corrs = [hourly_results[h]['avg_correlation'] for h in hours]
    scatter_wmeans = [hourly_results[h]['avg_weighted_mean'] for h in hours]
    
    scatter = ax5.scatter(scatter_corrs, scatter_wmeans, c=hours, cmap='viridis', s=100, alpha=0.7)
    ax5.set_xlabel('Average Correlation')
    ax5.set_ylabel('Average Weighted Mean')
    ax5.set_title('Correlation vs Weighted Mean by Hour')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Hours Remaining')
    
    # 6. Heatmap-style visualization
    ax6 = axes[1, 2]
    
    # Create a 2D representation of the data
    metrics = ['Avg Correlation', 'Avg Weighted Mean', 'Overall Correlation']
    data_matrix = np.array([
        [hourly_results[h]['avg_correlation'] for h in hours],
        [hourly_results[h]['avg_weighted_mean'] for h in hours],
        [hourly_results[h]['overall_correlation'] for h in hours]
    ])
    
    # Normalize each row for better visualization
    for i in range(len(metrics)):
        row = data_matrix[i]
        if not np.all(np.isnan(row)):
            data_matrix[i] = (row - np.nanmin(row)) / (np.nanmax(row) - np.nanmin(row))
    
    im = ax6.imshow(data_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax6.set_xticks(range(len(hours)))
    ax6.set_xticklabels(hours)
    ax6.set_yticks(range(len(metrics)))
    ax6.set_yticklabels(metrics)
    ax6.set_xlabel('Hours Remaining')
    ax6.set_title('Normalized Metrics Heatmap')
    plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    
    # Save plot
    plot_dir = "analysis_results"
    os.makedirs(plot_dir, exist_ok=True)
    filename = f"hourly_analysis_{crypto}_{min_hours}h_{max_hours}h.png"
    plt.savefig(os.path.join(plot_dir, filename), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nHourly analysis plot saved: {plot_dir}/{filename}")

def create_crypto_recap_file(hourly_results, crypto, min_hours, max_hours):
    """Create a recap file for the hourly analysis of a specific crypto"""
    
    recap_dir = "analysis_results"
    os.makedirs(recap_dir, exist_ok=True)
    
    filename = f"hourly_recap_{crypto}_{min_hours}h_{max_hours}h.txt"
    filepath = os.path.join(recap_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"HOURLY ANALYSIS RECAP FOR {crypto.upper()}\n")
        f.write(f"Time Window: {min_hours}-{max_hours} hours\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        if hourly_results:
            # Overall statistics
            all_avg_corrs = [hourly_results[h]['avg_correlation'] for h in hourly_results.keys()]
            all_overall_corrs = [hourly_results[h]['overall_correlation'] for h in hourly_results.keys()]
            all_weighted_means = [hourly_results[h]['avg_weighted_mean'] for h in hourly_results.keys()]
            
            # Filter out NaN values
            all_avg_corrs = [x for x in all_avg_corrs if not np.isnan(x)]
            all_overall_corrs = [x for x in all_overall_corrs if not np.isnan(x)]
            all_weighted_means = [x for x in all_weighted_means if not np.isnan(x)]
            
            if all_avg_corrs:
                f.write(f"Average Correlation across hours: {np.mean(all_avg_corrs):.4f} ± {np.std(all_avg_corrs):.4f}\n")
                f.write(f"Min/Max Average Correlation: {np.min(all_avg_corrs):.4f} / {np.max(all_avg_corrs):.4f}\n")
            
            if all_overall_corrs:
                f.write(f"Overall Correlation across hours: {np.mean(all_overall_corrs):.4f} ± {np.std(all_overall_corrs):.4f}\n")
                f.write(f"Min/Max Overall Correlation: {np.min(all_overall_corrs):.4f} / {np.max(all_overall_corrs):.4f}\n")
            
            if all_weighted_means:
                f.write(f"Average Weighted Mean across hours: {np.mean(all_weighted_means):.4f} ± {np.std(all_weighted_means):.4f}\n")
                f.write(f"Min/Max Weighted Mean: {np.min(all_weighted_means):.4f} / {np.max(all_weighted_means):.4f}\n")
            
            # Best and worst performing hours
            if all_overall_corrs:
                best_hour = max(hourly_results.keys(), key=lambda h: hourly_results[h]['overall_correlation'] if not np.isnan(hourly_results[h]['overall_correlation']) else -999)
                worst_hour = min(hourly_results.keys(), key=lambda h: hourly_results[h]['overall_correlation'] if not np.isnan(hourly_results[h]['overall_correlation']) else 999)
                
                f.write(f"\nBest performing hour: {best_hour} (correlation: {hourly_results[best_hour]['overall_correlation']:.4f})\n")
                f.write(f"Worst performing hour: {worst_hour} (correlation: {hourly_results[worst_hour]['overall_correlation']:.4f})\n")
        
        f.write(f"\nNumber of hours analyzed: {len(hourly_results)}\n")
        f.write(f"Total hours in window: {max_hours - min_hours}\n")
        
        f.write("\n\nDETAILED HOURLY BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Hour':>4} | {'Avg Corr':>9} | {'Std Corr':>9} | {'Avg W.Mean':>11} | {'Std W.Mean':>11} | {'Overall Corr':>13} | {'P-Value':>8} | {'Markets':>8} | {'Points':>7}\n")
        f.write("-" * 120 + "\n")
        
        for hour in sorted(hourly_results.keys()):
            result = hourly_results[hour]
            f.write(f"{hour:4d} | {result['avg_correlation']:9.4f} | {result['std_correlation']:9.4f} | "
                   f"{result['avg_weighted_mean']:11.4f} | {result['std_weighted_mean']:11.4f} | "
                   f"{result['overall_correlation']:13.4f} | {result['overall_p_value']:8.4f} | "
                   f"{result['n_markets']:8d} | {result['n_total_points']:7d}\n")
        
        f.write("\n\nINTERPRETATION NOTES\n")
        f.write("-" * 40 + "\n")
        f.write("- Avg Corr: Average correlation across markets for this hour\n")
        f.write("- Std Corr: Standard deviation of correlations across markets\n")
        f.write("- Avg W.Mean: Average weighted mean of |price_evolution| for this hour\n")
        f.write("- Std W.Mean: Standard deviation of weighted means across markets\n")
        f.write("- Overall Corr: Correlation using all data points for this hour\n")
        f.write("- P-Value: Statistical significance of overall correlation\n")
        f.write("- Markets: Number of markets contributing data for this hour\n")
        f.write("- Points: Total number of data points for this hour\n")
        
        # Additional insights
        f.write("\n\nKEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        if hourly_results:
            # Find hours with strongest correlations
            significant_hours = []
            for hour, result in hourly_results.items():
                if (not np.isnan(result['overall_correlation']) and 
                    abs(result['overall_correlation']) > 0.05 and 
                    result['overall_p_value'] < 0.05):
                    significant_hours.append((hour, result['overall_correlation'], result['overall_p_value']))
            
            if significant_hours:
                f.write("Hours with statistically significant correlations (p < 0.05, |r| > 0.05):\n")
                for hour, corr, p_val in sorted(significant_hours, key=lambda x: abs(x[1]), reverse=True):
                    f.write(f"  Hour {hour}: r = {corr:.4f}, p = {p_val:.4f}\n")
            else:
                f.write("No hours show statistically significant correlations.\n")
            
            # Trend analysis
            hours_list = sorted(hourly_results.keys())
            if len(hours_list) > 3:
                early_hours = hours_list[:len(hours_list)//3]
                late_hours = hours_list[-len(hours_list)//3:]
                
                early_corr = np.mean([hourly_results[h]['overall_correlation'] for h in early_hours if not np.isnan(hourly_results[h]['overall_correlation'])])
                late_corr = np.mean([hourly_results[h]['overall_correlation'] for h in late_hours if not np.isnan(hourly_results[h]['overall_correlation'])])
                
                if not np.isnan(early_corr) and not np.isnan(late_corr):
                    f.write(f"\nTrend Analysis:\n")
                    f.write(f"  Early hours ({min(early_hours)}-{max(early_hours)}): avg correlation = {early_corr:.4f}\n")
                    f.write(f"  Late hours ({min(late_hours)}-{max(late_hours)}): avg correlation = {late_corr:.4f}\n")
                    f.write(f"  Difference: {late_corr - early_corr:.4f}\n")
    
    print(f"Hourly recap file saved: {filepath}")
    return filepath

def analyze_profitability_windows(df, crypto="bitcoin"):
    """Analyze profitability across different time windows for trading strategies"""
    
    print(f"\n=== PROFITABILITY WINDOW ANALYSIS ({crypto.upper()}) ===")
    
    if df is None:
        print("No data available for profitability analysis")
        return None
    
    # Define different time windows to analyze
    time_windows = [
        (1, 5, "Early (1-5h)"),
        (6, 10, "Mid-Early (6-10h)"),
        (11, 15, "Mid-Late (11-15h)"),
        (16, 20, "Late (16-20h)"),
        (21, 24, "Very Late (21-24h)"),
        (1, 15, "Extended Early (1-15h)"),
        (4, 20, "Core Trading (4-20h)"),
        (1, 24, "Full Window (1-24h)")
    ]
    
    profitability_results = {}
    
    print(f"\n{'Window':<20} | {'Mean Gain':<10} | {'Median':<8} | {'Std Dev':<8} | {'Sharpe':<7} | {'Win%':<6} | {'Count':<7} | {'CI±':<6}")
    print("-" * 90)
    
    for min_hour, max_hour, window_name in time_windows:
        # Filter data to this time window
        window_data = df[(df['hour_bucket'] >= min_hour) & (df['hour_bucket'] <= max_hour)]
        
        if len(window_data) < 50:
            continue
        
        gains = window_data['price_evolution']
        
        # Calculate key metrics
        mean_gain = gains.mean()
        median_gain = gains.median()
        std_gain = gains.std()
        win_rate = (gains > 0).mean() * 100
        count = len(gains)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = mean_gain / std_gain if std_gain > 0 else 0
        
        # 95% confidence interval
        from scipy import stats
        ci = stats.t.interval(0.95, count-1, loc=mean_gain, scale=std_gain/np.sqrt(count))
        ci_range = (ci[1] - ci[0]) / 2
        
        # Store results
        profitability_results[window_name] = {
            'time_range': (min_hour, max_hour),
            'mean_gain': mean_gain,
            'median_gain': median_gain,
            'std_gain': std_gain,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'count': count,
            'confidence_interval': ci,
            'confidence_range': ci_range
        }
        
        # Print results
        print(f"{window_name:<20} | {mean_gain*100:9.2f}% | {median_gain*100:7.2f}% | {std_gain*100:7.2f}% | {sharpe_ratio:6.2f} | {win_rate:5.1f}% | {count:6d} | ±{ci_range*100:.2f}%")
    
    return profitability_results

def compare_crypto_profitability(btc_results, eth_results):
    """Compare profitability between BTC and ETH across different windows"""
    
    print(f"\n{'='*80}")
    print("CRYPTO PROFITABILITY COMPARISON")
    print(f"{'='*80}")
    
    if not btc_results or not eth_results:
        print("Missing data for comparison")
        return
    
    # Find common windows
    common_windows = set(btc_results.keys()) & set(eth_results.keys())
    
    print(f"\n{'Window':<20} | {'BTC Gain':<10} | {'ETH Gain':<10} | {'Difference':<11} | {'BTC Sharpe':<10} | {'ETH Sharpe':<10}")
    print("-" * 95)
    
    comparison_results = {}
    
    for window in sorted(common_windows):
        btc = btc_results[window]
        eth = eth_results[window]
        
        btc_gain = btc['mean_gain']
        eth_gain = eth['mean_gain']
        gain_diff = btc_gain - eth_gain
        
        btc_sharpe = btc['sharpe_ratio']
        eth_sharpe = eth['sharpe_ratio']
        
        comparison_results[window] = {
            'btc_gain': btc_gain,
            'eth_gain': eth_gain,
            'gain_difference': gain_diff,
            'btc_sharpe': btc_sharpe,
            'eth_sharpe': eth_sharpe,
            'better_crypto': 'BTC' if btc_gain > eth_gain else 'ETH'
        }
        
        print(f"{window:<20} | {btc_gain*100:9.2f}% | {eth_gain*100:9.2f}% | {gain_diff*100:10.2f}% | {btc_sharpe:9.2f} | {eth_sharpe:9.2f}")
    
    # Summary insights
    print(f"\n=== COMPARISON INSIGHTS ===")
    
    # Count wins for each crypto
    btc_wins = sum(1 for r in comparison_results.values() if r['better_crypto'] == 'BTC')
    eth_wins = sum(1 for r in comparison_results.values() if r['better_crypto'] == 'ETH')
    
    print(f"BTC outperforms in {btc_wins}/{len(common_windows)} time windows")
    print(f"ETH outperforms in {eth_wins}/{len(common_windows)} time windows")
    
    # Best windows for each crypto
    if comparison_results:
        best_btc_window = max(btc_results.items(), key=lambda x: x[1]['mean_gain'])
        best_eth_window = max(eth_results.items(), key=lambda x: x[1]['mean_gain'])
        
        print(f"\nBest BTC window: {best_btc_window[0]} ({best_btc_window[1]['mean_gain']*100:.2f}% gain)")
        print(f"Best ETH window: {best_eth_window[0]} ({best_eth_window[1]['mean_gain']*100:.2f}% gain)")
        
        # Risk-adjusted performance
        best_btc_sharpe = max(btc_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        best_eth_sharpe = max(eth_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        
        print(f"\nBest BTC risk-adjusted: {best_btc_sharpe[0]} (Sharpe: {best_btc_sharpe[1]['sharpe_ratio']:.2f})")
        print(f"Best ETH risk-adjusted: {best_eth_sharpe[0]} (Sharpe: {best_eth_sharpe[1]['sharpe_ratio']:.2f})")
    
    return comparison_results

def create_profitability_visualization(btc_results, eth_results, crypto_comparison):
    """Create visualization comparing profitability across windows and cryptos"""
    
    if not btc_results or not eth_results:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cryptocurrency Profitability Analysis Across Time Windows', fontsize=16)
    
    common_windows = set(btc_results.keys()) & set(eth_results.keys())
    windows = sorted(common_windows)
    
    # 1. Mean gains comparison
    ax1 = axes[0, 0]
    btc_gains = [btc_results[w]['mean_gain']*100 for w in windows]
    eth_gains = [eth_results[w]['mean_gain']*100 for w in windows]
    
    x = np.arange(len(windows))
    width = 0.35
    
    ax1.bar(x - width/2, btc_gains, width, label='BTC', alpha=0.8, color='orange')
    ax1.bar(x + width/2, eth_gains, width, label='ETH', alpha=0.8, color='blue')
    ax1.set_xlabel('Time Windows')
    ax1.set_ylabel('Mean Gain (%)')
    ax1.set_title('Mean Gains by Time Window')
    ax1.set_xticks(x)
    ax1.set_xticklabels([w.replace(' ', '\n') for w in windows], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 2. Sharpe ratios comparison
    ax2 = axes[0, 1]
    btc_sharpe = [btc_results[w]['sharpe_ratio'] for w in windows]
    eth_sharpe = [eth_results[w]['sharpe_ratio'] for w in windows]
    
    ax2.bar(x - width/2, btc_sharpe, width, label='BTC', alpha=0.8, color='orange')
    ax2.bar(x + width/2, eth_sharpe, width, label='ETH', alpha=0.8, color='blue')
    ax2.set_xlabel('Time Windows')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Risk-Adjusted Returns (Sharpe Ratio)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([w.replace(' ', '\n') for w in windows], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Win rates comparison
    ax3 = axes[1, 0]
    btc_winrates = [btc_results[w]['win_rate'] for w in windows]
    eth_winrates = [eth_results[w]['win_rate'] for w in windows]
    
    ax3.bar(x - width/2, btc_winrates, width, label='BTC', alpha=0.8, color='orange')
    ax3.bar(x + width/2, eth_winrates, width, label='ETH', alpha=0.8, color='blue')
    ax3.set_xlabel('Time Windows')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_title('Win Rate by Time Window')
    ax3.set_xticks(x)
    ax3.set_xticklabels([w.replace(' ', '\n') for w in windows], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    
    # 4. Risk vs Return scatter
    ax4 = axes[1, 1]
    btc_std = [btc_results[w]['std_gain']*100 for w in windows]
    eth_std = [eth_results[w]['std_gain']*100 for w in windows]
    
    ax4.scatter(btc_std, btc_gains, alpha=0.7, s=100, color='orange', label='BTC')
    ax4.scatter(eth_std, eth_gains, alpha=0.7, s=100, color='blue', label='ETH')
    
    # Add window labels
    for i, window in enumerate(windows):
        ax4.annotate(window.split()[0], (btc_std[i], btc_gains[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        ax4.annotate(window.split()[0], (eth_std[i], eth_gains[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax4.set_xlabel('Standard Deviation (%)')
    ax4.set_ylabel('Mean Gain (%)')
    ax4.set_title('Risk vs Return Profile')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_dir = "analysis_results"
    os.makedirs(plot_dir, exist_ok=True)
    filename = "profitability_comparison_btc_eth.png"
    plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nProfitability comparison plot saved: {plot_dir}/{filename}")

def analyze_grouped_hours_expected_gains(df, hourly_results, crypto="bitcoin", min_hours=1, max_hours=15):
    """Analyze expected mean gain metrics for grouped hours 1-15"""
    
    print(f"\n=== EXPECTED MEAN GAIN ANALYSIS ({crypto.upper()}, Hours {min_hours}-{max_hours}) ===")
    
    if df is None or not hourly_results:
        print("No data available for gain analysis")
        return None
    
    # Filter data to the specified hour range
    grouped_data = df[(df['hour_bucket'] >= min_hours) & (df['hour_bucket'] <= max_hours)]
    
    if len(grouped_data) == 0:
        print(f"No data found for hours {min_hours}-{max_hours}")
        return None
    
    gain_metrics = {}
    
    print(f"\nHour | Mean Gain | Median Gain | Std Dev | Positive % | Count | Confidence")
    print("-" * 85)
    
    for hour in range(min_hours, max_hours + 1):
        if hour not in hourly_results:
            continue
            
        hour_data = grouped_data[grouped_data['hour_bucket'] == hour]
        
        if len(hour_data) < 10:
            continue
        
        # Calculate gain metrics
        price_evolutions = hour_data['price_evolution']
        
        mean_gain = price_evolutions.mean()
        median_gain = price_evolutions.median()
        std_gain = price_evolutions.std()
        positive_pct = (price_evolutions > 0).mean() * 100
        count = len(price_evolutions)
        
        # Calculate confidence interval for mean (95%)
        from scipy import stats
        confidence_interval = stats.t.interval(0.95, count-1, 
                                             loc=mean_gain, 
                                             scale=std_gain/np.sqrt(count))
        confidence_range = confidence_interval[1] - confidence_interval[0]
        
        # Store metrics
        gain_metrics[hour] = {
            'mean_gain': mean_gain,
            'median_gain': median_gain,
            'std_gain': std_gain,
            'positive_percentage': positive_pct,
            'count': count,
            'confidence_interval': confidence_interval,
            'confidence_range': confidence_range
        }
        
        # Print formatted results
        print(f"{hour:4d} | {mean_gain:9.4f} | {median_gain:11.4f} | {std_gain:7.4f} | {positive_pct:9.1f}% | {count:5d} | ±{confidence_range/2:.4f}")
    
    # Calculate overall statistics for the grouped hours
    all_gains = grouped_data['price_evolution']
    overall_mean = all_gains.mean()
    overall_median = all_gains.median()
    overall_std = all_gains.std()
    overall_positive_pct = (all_gains > 0).mean() * 100
    overall_count = len(all_gains)
    
    # Overall confidence interval
    overall_ci = stats.t.interval(0.95, overall_count-1, 
                                 loc=overall_mean, 
                                 scale=overall_std/np.sqrt(overall_count))
    
    print("-" * 85)
    print(f"{'ALL':>4} | {overall_mean:9.4f} | {overall_median:11.4f} | {overall_std:7.4f} | {overall_positive_pct:9.1f}% | {overall_count:5d} | ±{(overall_ci[1]-overall_ci[0])/2:.4f}")
    
    # Additional insights
    print(f"\n=== GAIN INSIGHTS FOR HOURS {min_hours}-{max_hours} ===")
    
    if gain_metrics:
        # Best and worst hours by mean gain
        best_hour = max(gain_metrics.keys(), key=lambda h: gain_metrics[h]['mean_gain'])
        worst_hour = min(gain_metrics.keys(), key=lambda h: gain_metrics[h]['mean_gain'])
        
        print(f"Best performing hour: {best_hour} (mean gain: {gain_metrics[best_hour]['mean_gain']:.4f} = {gain_metrics[best_hour]['mean_gain']*100:.2f}%)")
        print(f"Worst performing hour: {worst_hour} (mean gain: {gain_metrics[worst_hour]['mean_gain']:.4f} = {gain_metrics[worst_hour]['mean_gain']*100:.2f}%)")
        
        # Most consistent hours (lowest std dev relative to mean)
        consistent_hours = []
        for hour, metrics in gain_metrics.items():
            if abs(metrics['mean_gain']) > 0.001:  # Avoid division by very small numbers
                cv = abs(metrics['std_gain'] / metrics['mean_gain'])  # Coefficient of variation
                consistent_hours.append((hour, cv, metrics['std_gain']))
        
        if consistent_hours:
            consistent_hours.sort(key=lambda x: x[1])
            most_consistent = consistent_hours[0]
            least_consistent = consistent_hours[-1]
            
            print(f"Most consistent hour: {most_consistent[0]} (CV: {most_consistent[1]:.2f}, std: {most_consistent[2]:.4f})")
            print(f"Least consistent hour: {least_consistent[0]} (CV: {least_consistent[1]:.2f}, std: {least_consistent[2]:.4f})")
        
        # Hours with highest positive percentage
        positive_hours = sorted(gain_metrics.items(), key=lambda x: x[1]['positive_percentage'], reverse=True)
        print(f"Most positive hour: {positive_hours[0][0]} ({positive_hours[0][1]['positive_percentage']:.1f}% positive outcomes)")
        print(f"Least positive hour: {positive_hours[-1][0]} ({positive_hours[-1][1]['positive_percentage']:.1f}% positive outcomes)")
        
        # Statistical significance test (t-test against zero)
        significant_hours = []
        for hour, metrics in gain_metrics.items():
            hour_data = grouped_data[grouped_data['hour_bucket'] == hour]['price_evolution']
            t_stat, p_value = stats.ttest_1samp(hour_data, 0)
            if p_value < 0.05:
                significant_hours.append((hour, metrics['mean_gain'], p_value))
        
        if significant_hours:
            print(f"\nHours with statistically significant gains (p < 0.05):")
            for hour, gain, p_val in sorted(significant_hours, key=lambda x: abs(x[1]), reverse=True):
                direction = "positive" if gain > 0 else "negative"
                print(f"  Hour {hour}: {direction} gain of {gain:.4f} ({gain*100:.2f}%), p = {p_val:.4f}")
        else:
            print(f"\nNo hours show statistically significant gains (p < 0.05)")
    
    # Correlation between hour and expected gain
    if len(gain_metrics) > 3:
        hours_list = list(gain_metrics.keys())
        gains_list = [gain_metrics[h]['mean_gain'] for h in hours_list]
        
        hour_gain_corr, hour_gain_p = pearsonr(hours_list, gains_list)
        print(f"\nCorrelation between hour number and mean gain: r = {hour_gain_corr:.4f}, p = {hour_gain_p:.4f}")
        
        if abs(hour_gain_corr) > 0.3 and hour_gain_p < 0.05:
            trend = "increasing" if hour_gain_corr > 0 else "decreasing"
            print(f"  -> Significant {trend} trend in gains over time")
        else:
            print(f"  -> No significant trend in gains over time")
    
    return gain_metrics

def comprehensive_hourly_analysis():
    """Run comprehensive hourly analysis for both BTC and ETH with different time windows"""
    print("=== COMPREHENSIVE HOURLY ANALYSIS (BTC & ETH, Multiple Time Windows) ===")
    
    cryptocurrencies = ["bitcoin", "ethereum"]
    time_windows = [
        (0, 24, "full"),
        (4, 20, "mid-range")
    ]
    
    all_results = {}
    recap_files = []
    profitability_results = {}
    
    for crypto in cryptocurrencies:
        all_results[crypto] = {}
        print(f"\n{'='*60}")
        print(f"ANALYZING {crypto.upper()}")
        print(f"{'='*60}")
        
        for min_hours, max_hours, window_name in time_windows:
            print(f"\n--- {window_name.upper()} TIME WINDOW ({min_hours}-{max_hours}h) ---")
            
            # Run hourly analysis
            df, hourly_results = analyze_hourly_linear_correlation(
                crypto=crypto, 
                min_hours=min_hours, 
                max_hours=max_hours
            )
            
            if df is not None and hourly_results is not None:
                # Initialize the results dictionary for this crypto and window
                all_results[crypto][window_name] = {
                    'df': df,
                    'hourly_results': hourly_results,
                    'time_window': (min_hours, max_hours)
                }
                
                # Create summary plots (but not individual hour plots)
                create_hourly_analysis_plots(df, hourly_results, crypto, min_hours, max_hours)
                
                # Run profitability analysis for the full window
                if window_name == "full":
                    profitability_results[crypto] = analyze_profitability_windows(df, crypto)
                
                # Analyze expected gains for hours 1-15 (only for full window)
                if min_hours <= 1 and max_hours >= 15:
                    gain_metrics = analyze_grouped_hours_expected_gains(df, hourly_results, crypto, 1, 15)
                    all_results[crypto][window_name]['gain_metrics'] = gain_metrics
                
                # Create recap file
                recap_file = create_crypto_recap_file(hourly_results, crypto, min_hours, max_hours)
                recap_files.append(recap_file)
                
                # Quick asymmetric analysis
                print(f"\n--- Asymmetric Effects ({crypto.upper()}, {window_name}) ---")
                analyze_asymmetric_effects_hourly(df)
            else:
                print(f"No data available for {crypto} in {window_name} time window")
    
    # Cross-crypto profitability comparison
    if 'bitcoin' in profitability_results and 'ethereum' in profitability_results:
        crypto_comparison = compare_crypto_profitability(
            profitability_results['bitcoin'], 
            profitability_results['ethereum']
        )
        
        # Create profitability visualization
        create_profitability_visualization(
            profitability_results['bitcoin'],
            profitability_results['ethereum'],
            crypto_comparison
        )
    
    # Overall comparison summary
    print(f"\n{'='*60}")
    print("HOURLY ANALYSIS COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for crypto in cryptocurrencies:
        for window_name in ["full", "mid-range"]:
            if window_name in all_results[crypto]:
                hourly_results = all_results[crypto][window_name]['hourly_results']
                if hourly_results:
                    # Calculate overall average correlation
                    all_corrs = [hourly_results[h]['overall_correlation'] for h in hourly_results.keys()]
                    all_corrs = [x for x in all_corrs if not np.isnan(x)]
                    
                    if all_corrs:
                        avg_corr = np.mean(all_corrs)
                        print(f"{crypto.upper()} {window_name}: Average hourly correlation = {avg_corr:.4f}, "
                              f"Range = [{np.min(all_corrs):.4f}, {np.max(all_corrs):.4f}], "
                              f"Hours analyzed = {len(all_corrs)}")
                        
                        # Print gain summary if available
                        if 'gain_metrics' in all_results[crypto][window_name]:
                            gain_metrics = all_results[crypto][window_name]['gain_metrics']
                            if gain_metrics:
                                best_hour = max(gain_metrics.keys(), key=lambda h: gain_metrics[h]['mean_gain'])
                                best_gain = gain_metrics[best_hour]['mean_gain']
                                print(f"    Best gain hour: {best_hour} ({best_gain*100:.2f}%)")
    
    print(f"\nRecap files created:")
    for file in recap_files:
        print(f"  - {file}")
    
    return all_results, profitability_results

if __name__ == "__main__":
    # Run the comprehensive hourly analysis
    all_results, profitability_results = comprehensive_hourly_analysis()
