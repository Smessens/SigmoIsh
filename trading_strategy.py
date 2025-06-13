"""
Fee-Optimized Hourly Trading Strategy with Advanced Signal Filtering
Designed for per-minute data with aggressive fee reduction techniques
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from scipy.stats import pearsonr
import seaborn as sns
from itertools import product
from scipy import stats

# Import functions from existing analysis files
from hourly_enhanced_sigmoid_analysis import (
    analyze_hourly_linear_correlation,
    analyze_asymmetric_effects_hourly
)

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

class FeeOptimizedTradingStrategy:
    def __init__(self, initial_capital=10000, signal_threshold=0.05, 
                 maker_fee=0.001, taker_fee=0.001, max_position_size=0.3,
                 min_trade_interval=30, signal_smoothing_window=5,
                 min_expected_profit=0.005, position_scaling=True):
        """
        Fee-optimized trading strategy for per-minute data
        
        Parameters:
        - min_trade_interval: Minimum minutes between trades (default: 30min)
        - signal_smoothing_window: Number of periods to smooth signals (reduces noise)
        - min_expected_profit: Minimum expected profit % to cover fees (0.5% = 0.005)
        - position_scaling: Whether to scale positions based on signal persistence
        """
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.signal_threshold = signal_threshold
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.max_position_size = max_position_size
        self.min_hours = 1
        self.max_hours = 15
        
        # Fee optimization parameters
        self.min_trade_interval = min_trade_interval  # minutes
        self.signal_smoothing_window = signal_smoothing_window
        self.min_expected_profit = min_expected_profit
        self.position_scaling = position_scaling
        
        # Trading state
        self.trades = []
        self.current_position = None
        self.last_trade_timestamp = None
        
        # Signal tracking for smoothing and persistence
        self.signal_history = []
        self.price_history = []
        
        # Performance tracking
        self.win_rate = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.max_drawdown = 0
        self.max_portfolio_value = initial_capital
        self.total_fees_paid = 0
        self.start_timestamp = None
        self.end_timestamp = None
        self.signals_filtered_by_timing = 0
        self.signals_filtered_by_profit = 0
        self.signals_filtered_by_smoothing = 0
        
        print(f"🛡️  FEE-OPTIMIZED STRATEGY INITIALIZED")
        print(f"💰 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"📊 Signal Threshold: {self.signal_threshold}")
        print(f"⏱️  Min Trade Interval: {self.min_trade_interval} minutes")
        print(f"📈 Min Expected Profit: {self.min_expected_profit:.2%}")
        print(f"🔄 Signal Smoothing: {self.signal_smoothing_window} periods")

    def can_trade_now(self, current_timestamp):
        """Check if enough time has passed since last trade"""
        if self.last_trade_timestamp is None:
            return True
        
        time_diff = current_timestamp - self.last_trade_timestamp
        time_diff_minutes = time_diff / 60  # Convert seconds to minutes
        
        return time_diff_minutes >= self.min_trade_interval

    def smooth_signal(self, new_signal_data):
        """Smooth signals using moving average to reduce noise"""
        self.signal_history.append(new_signal_data)
        
        # Keep only the recent history
        if len(self.signal_history) > self.signal_smoothing_window * 2:
            self.signal_history = self.signal_history[-self.signal_smoothing_window * 2:]
        
        if len(self.signal_history) < self.signal_smoothing_window:
            return None  # Not enough data for smoothing
        
        # Calculate smoothed values
        recent_signals = self.signal_history[-self.signal_smoothing_window:]
        
        smoothed_data = {
            'linear_distance': np.mean([s['linear_distance'] for s in recent_signals]),
            'signal_strength': np.mean([s['signal_strength'] for s in recent_signals]),
            'signal_direction': new_signal_data['signal_direction'],  # Use latest direction
            'price': new_signal_data['price'],
            'hours_remaining': new_signal_data['hours_remaining']
        }
        
        # Check signal persistence (same direction for majority of recent signals)
        directions = [s['signal_direction'] for s in recent_signals if s['signal_direction'] != 'HOLD']
        if len(directions) < self.signal_smoothing_window * 0.6:  # Less than 60% actual signals
            return None
        
        direction_consistency = directions.count(smoothed_data['signal_direction']) / len(directions)
        if direction_consistency < 0.7:  # Less than 70% consistency
            smoothed_data['signal_direction'] = 'HOLD'
            smoothed_data['signal_strength'] = 0
        
        return smoothed_data

    def estimate_expected_profit(self, signal_strength, hours_remaining, current_price):
        """Estimate expected profit based on signal strength and historical performance"""
        # Base expected move (this would be calibrated from historical data)
        base_move_pct = 0.02  # 2% base expected move for strong signals
        
        # Scale by signal strength
        expected_move = base_move_pct * signal_strength
        
        # Time decay - closer to expiry = more volatile
        time_factor = 1.0 + (self.max_hours - hours_remaining) / self.max_hours * 0.5
        expected_move *= time_factor
        
        # Account for position size
        position_size = self.calculate_position_size(signal_strength)
        expected_profit_dollars = self.portfolio_value * position_size * expected_move
        
        # Total fees for round trip
        trade_value = self.portfolio_value * position_size
        total_fees = 2 * self.calculate_trading_fee(trade_value, is_maker=False)
        
        # Net expected profit
        net_expected_profit = expected_profit_dollars - total_fees
        net_expected_profit_pct = net_expected_profit / trade_value
        
        return net_expected_profit_pct, total_fees

    def calculate_signal_strength(self, linear_distance, hours_remaining):
        """Enhanced signal calculation with higher threshold for fee optimization"""
        abs_signal = abs(linear_distance)
        
        # Higher threshold for per-minute data
        if abs_signal < self.signal_threshold:
            return 'HOLD', 0.0
        
        # Time decay factor
        time_factor = 1.0 + (self.max_hours - hours_remaining) / self.max_hours * 0.5
        
        # Signal strength with more aggressive scaling
        max_signal = 0.1
        raw_strength = min(abs_signal / max_signal, 1.0)
        signal_strength = min(raw_strength * time_factor, 1.0)
        
        # Determine direction
        if linear_distance > 0:
            return 'SHORT', signal_strength
        else:
            return 'LONG', signal_strength

    def calculate_position_size(self, signal_strength):
        """Dynamic position sizing based on signal strength and fee optimization"""
        base_position = 0.15  # Larger base position to amortize fees
        
        if self.position_scaling:
            # Scale more aggressively for stronger signals
            position_multiplier = 1 + (signal_strength ** 2) * 2  # Quadratic scaling
        else:
        position_multiplier = 1 + signal_strength
        
        position_size_pct = min(base_position * position_multiplier, self.max_position_size)
        return position_size_pct

    def calculate_trading_fee(self, trade_value, is_maker=False):
        """Calculate trading fee with BNB discount consideration"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        base_fee = trade_value * fee_rate
        
        # Simulate 5% BNB discount (common on Binance)
        bnb_discount = 0.05
        return base_fee * (1 - bnb_discount)

    def should_execute_trade(self, signal_data, current_price, current_timestamp):
        """Comprehensive trade execution filter"""
        
        # 1. Check timing constraint
        if not self.can_trade_now(current_timestamp):
            self.signals_filtered_by_timing += 1
            return False, "Timing filter"
        
        # 2. Smooth the signal
        smoothed = self.smooth_signal(signal_data)
        if smoothed is None or smoothed['signal_direction'] == 'HOLD':
            self.signals_filtered_by_smoothing += 1
            return False, "Smoothing filter"
        
        # 3. Check expected profitability
        expected_profit_pct, total_fees = self.estimate_expected_profit(
            smoothed['signal_strength'], 
            smoothed['hours_remaining'], 
            current_price
        )
        
        if expected_profit_pct < self.min_expected_profit:
            self.signals_filtered_by_profit += 1
            return False, f"Profit filter ({expected_profit_pct:.3f} < {self.min_expected_profit:.3f})"
        
        return True, smoothed

    def execute_trade(self, signal_direction, signal_strength, crypto_price, market_prob, 
                     expected_prob, linear_distance, timestamp, hours_remaining, market_name):
        """Execute trade with all fee optimization checks"""
        
        # Package signal data
        signal_data = {
            'signal_direction': signal_direction,
            'signal_strength': signal_strength,
            'linear_distance': linear_distance,
            'price': crypto_price,
            'hours_remaining': hours_remaining
        }
        
        # Check if we should execute this trade
        should_trade, result = self.should_execute_trade(signal_data, crypto_price, timestamp)
        
        if not should_trade:
            return  # Trade filtered out
        
        # Use smoothed signal data
        smoothed_signal = result
        signal_direction = smoothed_signal['signal_direction']
        signal_strength = smoothed_signal['signal_strength']
        
        if signal_direction == 'HOLD':
            return
        
        # Close existing position if any
        if self.current_position is not None:
            self.close_position(crypto_price, timestamp, market_name)
        
        # Calculate position size
        position_size_pct = self.calculate_position_size(signal_strength)
        gross_trade_value = self.portfolio_value * position_size_pct
        
        # Calculate trading fee with BNB discount
        trading_fee = self.calculate_trading_fee(gross_trade_value, is_maker=False)
        net_trade_value = gross_trade_value - trading_fee
        position_shares = net_trade_value / crypto_price
        
        # Update portfolio for fee
        self.portfolio_value -= trading_fee
        self.total_fees_paid += trading_fee
        self.last_trade_timestamp = timestamp
        
        # Track timestamps
        if self.start_timestamp is None:
            self.start_timestamp = timestamp
        self.end_timestamp = timestamp
        
        # Open new position
        self.current_position = {
            'direction': signal_direction,
            'entry_price': crypto_price,
            'entry_timestamp': timestamp,
            'position_shares': position_shares,
            'trade_value': net_trade_value,
            'entry_fee': trading_fee,
            'market_name': market_name,
            'hours_remaining': hours_remaining,
            'signal_strength': signal_strength,
            'linear_distance': linear_distance,
            'market_prob': market_prob,
            'expected_prob': expected_prob
        }
        
        trade_record = {
            'timestamp': timestamp,
            'action': 'OPEN',
            'direction': signal_direction,
            'crypto_price': crypto_price,
            'position_shares': position_shares,
            'trade_value': net_trade_value,
            'fee': trading_fee,
            'signal_strength': signal_strength,
            'linear_distance': linear_distance,
            'hours_remaining': hours_remaining,
            'market_name': market_name,
            'portfolio_value': self.portfolio_value
        }
        
        self.trades.append(trade_record)
        
        #print(f"🎯 {signal_direction} | Price: ${crypto_price:,.0f} | Size: {position_size_pct:.1%} | "
              #f"Signal: {abs(linear_distance):.3f} | Hours: {hours_remaining:.1f} | "
              #f"Fee: ${trading_fee:.2f}")

    def close_position(self, exit_price, exit_timestamp, market_name):
        """Close position with fee optimization"""
        if self.current_position is None:
            return
        
        position = self.current_position
        
        # Calculate gross P&L
        if position['direction'] == 'LONG':
            gross_pnl = position['position_shares'] * (exit_price - position['entry_price'])
        else:  # SHORT
            gross_pnl = position['position_shares'] * (position['entry_price'] - exit_price)
        
        # Calculate exit fee with BNB discount
        exit_trade_value = position['position_shares'] * exit_price
        exit_fee = self.calculate_trading_fee(exit_trade_value, is_maker=False)
        
        # Net P&L after exit fee
        net_pnl = gross_pnl - exit_fee
        pnl_pct = net_pnl / position['trade_value']
        
        # Update portfolio
        self.portfolio_value += net_pnl
        self.total_fees_paid += exit_fee
        
        # Track performance metrics
        self.total_trades += 1
        if net_pnl > 0:
            self.profitable_trades += 1
        
        # Update max drawdown
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        else:
            drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Record trade
        trade_record = {
            'timestamp': exit_timestamp,
            'action': 'CLOSE',
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'gross_pnl': gross_pnl,
            'exit_fee': exit_fee,
            'net_pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'total_fees': position['entry_fee'] + exit_fee,
            'hold_duration': exit_timestamp - position['entry_timestamp'],
            'market_name': market_name,
            'portfolio_value': self.portfolio_value
        }
        
        self.trades.append(trade_record)
        
        direction_emoji = "📈" if position['direction'] == 'LONG' else "📉"
        pnl_emoji = "💚" if net_pnl > 0 else "❌"
        #print(f"{pnl_emoji} CLOSE {direction_emoji} | P&L: ${net_pnl:+,.0f} ({pnl_pct:+.2%}) | "
              #f"Fees: ${position['entry_fee'] + exit_fee:.2f} | Portfolio: ${self.portfolio_value:,.0f}")
        
        self.current_position = None

    def process_market_data(self, data_dir="collected_data", crypto="bitcoin"):
        """Process market data with per-minute sampling and fee optimization"""
        #print(f"\n🔍 PROCESSING {crypto.upper()} MARKET DATA (Fee-Optimized)")
        #print("=" * 70)
        
        # Collect market files
        sorted_files = collect_market_files(data_dir, crypto)
        #print(f"Found {len(sorted_files)} {crypto} market files")
        
        processed_markets = 0
        valid_signals = 0
        total_signals_generated = 0
        
        for file_index, (date, file_path, filename) in enumerate(sorted_files):
            json_data = read_json_file(file_path)
            
            if not json_data:
                continue
            
            # Get market metadata
            metadata = json_data.get("collection_metadata", {})
            data_points = json_data.get("data_points", [])
            
            if not data_points:
                continue
            
            end_timestamp = metadata.get("collection_period", {}).get("end")
            starting_price = metadata.get("starting_price")
            
            if not end_timestamp or not starting_price:
                continue
            
            # Get next market's starting price
            next_starting_price = get_next_market_starting_price(file_index, sorted_files)
            if not next_starting_price:
                continue
            
            # Fit sigmoid for this market
            hourly_data = {}
            
            # Collect data for sigmoid fitting
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
            
            # Average probabilities per hour
            hourly_averages = {}
            for hour, probs in hourly_data.items():
                if probs:
                    hourly_averages[hour] = np.mean(probs)
            
            if len(hourly_averages) < 4:
                continue
            
            # Fit sigmoid
            sigmoid_params = fit_hourly_sigmoid(hourly_averages)
            if sigmoid_params is None:
                continue
            
            # Process with per-minute data (every point, not sampled)
            market_signals = 0
            
            for point in data_points:
                timestamp = point.get("timestamp")
                orderbook = point.get("orderbook")
                crypto_price = get_btc_price_from_point(point)
                
                if not all([timestamp, orderbook, crypto_price]):
                    continue
                
                hours_remaining = calculate_hours_remaining(timestamp, end_timestamp)
                
                # Apply trading window filter
                if hours_remaining <= self.min_hours or hours_remaining > self.max_hours:
                    continue
                
                current_prob = get_market_midpoint_probability(orderbook)
                if current_prob is None:
                    continue
                
                # Calculate signal
                expected_prob = sigmoid(hours_remaining, *sigmoid_params)
                linear_distance = current_prob - expected_prob
                
                signal_direction, signal_strength = self.calculate_signal_strength(
                    linear_distance, hours_remaining
                )
                
                total_signals_generated += 1
                
                if signal_direction != 'HOLD':
                    market_signals += 1
                    
                    self.execute_trade(
                        signal_direction, signal_strength, crypto_price,
                        current_prob, expected_prob, linear_distance,
                        timestamp, hours_remaining, filename
                    )
                    
                    if self.current_position is not None:  # Trade was actually executed
                        valid_signals += 1
            
            # Close position at market end if still open
            if self.current_position is not None:
                final_data_point = data_points[-1]
                final_price = get_btc_price_from_point(final_data_point)
                if final_price:
                    self.close_position(final_price, end_timestamp, filename)
            
            processed_markets += 1
            #if market_signals > 0:
                #print(f"📊 Market {processed_markets}: {filename[:25]}... | "
                      #f"Signals: {market_signals} | Executed: {valid_signals}")
        
        print(f"\n✅ PROCESSING COMPLETE")
        print(f"📈 Markets Processed: {processed_markets}")
        print(f"🎯 Total Signals Generated: {total_signals_generated}")
        print(f"🎯 Valid Signals Executed: {valid_signals}")
        print(f"🛡️  Signals Filtered by Timing: {self.signals_filtered_by_timing}")
        print(f"🛡️  Signals Filtered by Smoothing: {self.signals_filtered_by_smoothing}")
        print(f"🛡️  Signals Filtered by Profit: {self.signals_filtered_by_profit}")
        print(f"💰 Fee Efficiency: {(1 - self.total_fees_paid/self.initial_capital)*100:.1f}%")
        
        return processed_markets, valid_signals

    def calculate_daily_return_percentage(self):
        """Calculate daily return percentage based on trading period"""
        if self.start_timestamp is None or self.end_timestamp is None:
            return 0.0
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        time_duration = self.end_timestamp - self.start_timestamp
        days = time_duration / (24 * 3600)
        
        if days <= 0:
            return 0.0
        
        daily_return_pct = (total_return / days) * 100
        return daily_return_pct

    def generate_performance_report(self, verbose=False):
        """Generate performance report with fee analysis"""
        if not self.trades:
            return None
        
        close_trades = [t for t in self.trades if t['action'] == 'CLOSE']
        if close_trades:
            profitable = sum(1 for t in close_trades if t['net_pnl'] > 0)
            self.win_rate = profitable / len(close_trades)
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        daily_return_pct = self.calculate_daily_return_percentage()
        fee_percentage = (self.total_fees_paid / self.initial_capital) * 100
        
        # Calculate average trade size and fee per trade
        if close_trades:
            avg_trade_size = np.mean([t['trade_value'] for t in self.trades if t['action'] == 'OPEN'])
            avg_fee_per_trade = self.total_fees_paid / len(close_trades)
            avg_hold_time = np.mean([t['hold_duration'] for t in close_trades]) / 3600  # Convert to hours
        else:
            avg_trade_size = 0
            avg_fee_per_trade = 0
            avg_hold_time = 0
        
        performance_metrics = {
            'total_return': total_return,
            'daily_return_pct': daily_return_pct,
            'win_rate': self.win_rate,
            'total_trades': len(close_trades),
            'max_drawdown': self.max_drawdown,
            'final_portfolio': self.portfolio_value,
            'total_fees_paid': self.total_fees_paid,
            'fee_percentage': fee_percentage,
            'avg_trade_size': avg_trade_size,
            'avg_fee_per_trade': avg_fee_per_trade,
            'avg_hold_time_hours': avg_hold_time,
            'signal_threshold': self.signal_threshold,
            'min_trade_interval': self.min_trade_interval,
            'signal_smoothing_window': self.signal_smoothing_window,
            'min_expected_profit': self.min_expected_profit,
            'signals_filtered_timing': self.signals_filtered_by_timing,
            'signals_filtered_smoothing': self.signals_filtered_by_smoothing,
            'signals_filtered_profit': self.signals_filtered_by_profit
        }
        
        if verbose:
            print(f"📊 Threshold: {self.signal_threshold:.3f} | Interval: {self.min_trade_interval}min | "
                  f"Return: {total_return:+.2%} | Daily: {daily_return_pct:+.3f}%/day | "
                  f"Trades: {len(close_trades)} | Win Rate: {self.win_rate:.1%} | "
                  f"Fees: {fee_percentage:.2f}% | Hold: {avg_hold_time:.1f}h")
        
        return performance_metrics

def create_parameter_analysis_plots(df_results, crypto="bitcoin"):
    """Create comprehensive plots analyzing the impact of each parameter"""
    
    if df_results is None or len(df_results) == 0:
        print("❌ No results to plot!")
            return
        
    print(f"📊 Creating plots for {len(df_results)} results...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs("analysis_results/parameter_analysis", exist_ok=True)
    
    try:
        # 1. SIGNAL THRESHOLD ANALYSIS
        print("  📈 Signal threshold analysis...")
        create_signal_threshold_plots(df_results, crypto)
        
        # 2. TRADE INTERVAL ANALYSIS  
        print("  ⏱️  Trade interval analysis...")
        create_trade_interval_plots(df_results, crypto)
        
        # 3. MIN EXPECTED PROFIT ANALYSIS
        print("  💰 Min profit analysis...")
        create_min_profit_plots(df_results, crypto)
        
        # 4. SIGNAL SMOOTHING ANALYSIS
        print("  🔄 Smoothing analysis...")
        create_smoothing_plots(df_results, crypto)
        
        # 5. COMBINED CORRELATION ANALYSIS
        print("  🔗 Correlation analysis...")
        create_correlation_analysis(df_results, crypto)
        
        print(f"✅ All parameter analysis plots saved to analysis_results/parameter_analysis/")
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        import traceback
        traceback.print_exc()

def create_signal_threshold_plots(df_results, crypto):
    """Analyze impact of signal threshold parameter"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Signal Threshold Impact Analysis - {crypto.upper()}', fontsize=16, fontweight='bold')
    
    # Group by threshold for analysis
    threshold_groups = df_results.groupby('signal_threshold')
    
    # Plot 1: Daily Return vs Threshold
        ax1 = axes[0, 0]
    thresholds = []
    returns_mean = []
    returns_std = []
    
    for threshold, group in threshold_groups:
        thresholds.append(threshold)
        returns_mean.append(group['daily_return_pct'].mean())
        returns_std.append(group['daily_return_pct'].std())
    
    ax1.errorbar(thresholds, returns_mean, yerr=returns_std, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('Signal Threshold')
    ax1.set_ylabel('Daily Return %')
    ax1.set_title('Daily Return vs Signal Threshold')
        ax1.grid(True, alpha=0.3)
        
    # Plot 2: Trade Frequency vs Threshold
        ax2 = axes[0, 1]
    trades_mean = []
    trades_std = []
    
    for threshold, group in threshold_groups:
        trades_mean.append(group['total_trades'].mean())
        trades_std.append(group['total_trades'].std())
    
    ax2.errorbar(thresholds, trades_mean, yerr=trades_std, marker='s', color='orange', capsize=5, capthick=2)
    ax2.set_xlabel('Signal Threshold')
    ax2.set_ylabel('Number of Trades')
    ax2.set_title('Trade Frequency vs Signal Threshold')
            ax2.grid(True, alpha=0.3)
        
    # Plot 3: Win Rate vs Threshold
    ax3 = axes[0, 2]
    winrate_mean = []
    winrate_std = []
    
    for threshold, group in threshold_groups:
        winrate_mean.append(group['win_rate'].mean())
        winrate_std.append(group['win_rate'].std())
    
    ax3.errorbar(thresholds, winrate_mean, yerr=winrate_std, marker='^', color='green', capsize=5, capthick=2)
    ax3.set_xlabel('Signal Threshold')
    ax3.set_ylabel('Win Rate')
    ax3.set_title('Win Rate vs Signal Threshold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Fee Percentage vs Threshold
    ax4 = axes[1, 0]
    fees_mean = []
    fees_std = []
    
    for threshold, group in threshold_groups:
        fees_mean.append(group['fee_percentage'].mean())
        fees_std.append(group['fee_percentage'].std())
    
    ax4.errorbar(thresholds, fees_mean, yerr=fees_std, marker='d', color='red', capsize=5, capthick=2)
    ax4.set_xlabel('Signal Threshold')
    ax4.set_ylabel('Fee Percentage %')
    ax4.set_title('Fee Impact vs Signal Threshold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Risk-Adjusted Return (Sharpe-like)
    ax5 = axes[1, 1]
    risk_adj_returns = []
    
    for threshold, group in threshold_groups:
        mean_return = group['daily_return_pct'].mean()
        std_return = group['daily_return_pct'].std()
        sharpe_like = mean_return / (std_return + 0.001)  # Add small epsilon to avoid division by zero
        risk_adj_returns.append(sharpe_like)
    
    ax5.plot(thresholds, risk_adj_returns, marker='o', color='purple', linewidth=2, markersize=8)
    ax5.set_xlabel('Signal Threshold')
    ax5.set_ylabel('Risk-Adjusted Return')
    ax5.set_title('Risk-Adjusted Return vs Threshold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Scatter plot of all data points
    ax6 = axes[1, 2]
    scatter = ax6.scatter(df_results['signal_threshold'], df_results['daily_return_pct'], 
                         c=df_results['total_trades'], cmap='viridis', alpha=0.7)
    ax6.set_xlabel('Signal Threshold')
    ax6.set_ylabel('Daily Return %')
    ax6.set_title('Return vs Threshold (colored by trade count)')
    plt.colorbar(scatter, ax=ax6, label='Number of Trades')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis_results/parameter_analysis/signal_threshold_analysis_{crypto}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_trade_interval_plots(df_results, crypto):
    """Analyze impact of trade interval parameter"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Trade Interval Impact Analysis - {crypto.upper()}', fontsize=16, fontweight='bold')
    
    # Group by interval for analysis
    interval_groups = df_results.groupby('min_trade_interval')
    
    # Plot 1: Daily Return vs Interval
    ax1 = axes[0, 0]
    intervals = []
    returns_mean = []
    returns_std = []
    
    for interval, group in interval_groups:
        intervals.append(interval)
        returns_mean.append(group['daily_return_pct'].mean())
        returns_std.append(group['daily_return_pct'].std())
    
    ax1.errorbar(intervals, returns_mean, yerr=returns_std, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('Min Trade Interval (minutes)')
    ax1.set_ylabel('Daily Return %')
    ax1.set_title('Daily Return vs Trade Interval')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Trade Frequency vs Interval (should be inversely related)
    ax2 = axes[0, 1]
    trades_mean = []
    trades_std = []
    
    for interval, group in interval_groups:
        trades_mean.append(group['total_trades'].mean())
        trades_std.append(group['total_trades'].std())
    
    ax2.errorbar(intervals, trades_mean, yerr=trades_std, marker='s', color='orange', capsize=5, capthick=2)
    ax2.set_xlabel('Min Trade Interval (minutes)')
    ax2.set_ylabel('Number of Trades')
    ax2.set_title('Trade Frequency vs Interval')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fee Efficiency vs Interval
    ax3 = axes[0, 2]
    fee_efficiency = []
    fee_efficiency_std = []
    
    for interval, group in interval_groups:
        efficiency = 100 - group['fee_percentage']
        fee_efficiency.append(efficiency.mean())
        fee_efficiency_std.append(efficiency.std())
    
    ax3.errorbar(intervals, fee_efficiency, yerr=fee_efficiency_std, marker='^', color='green', capsize=5, capthick=2)
    ax3.set_xlabel('Min Trade Interval (minutes)')
    ax3.set_ylabel('Fee Efficiency %')
    ax3.set_title('Fee Efficiency vs Interval')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average Hold Time vs Interval
    ax4 = axes[1, 0]
    hold_time_mean = []
    hold_time_std = []
    
    for interval, group in interval_groups:
        hold_time_mean.append(group['avg_hold_time_hours'].mean())
        hold_time_std.append(group['avg_hold_time_hours'].std())
    
    ax4.errorbar(intervals, hold_time_mean, yerr=hold_time_std, marker='d', color='red', capsize=5, capthick=2)
    ax4.set_xlabel('Min Trade Interval (minutes)')
    ax4.set_ylabel('Average Hold Time (hours)')
    ax4.set_title('Hold Time vs Trade Interval')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Return per Trade vs Interval
    ax5 = axes[1, 1]
    return_per_trade = []
    return_per_trade_std = []
    
    for interval, group in interval_groups:
        rpt = group['total_return'] / (group['total_trades'] + 0.001)  # Avoid division by zero
        return_per_trade.append(rpt.mean())
        return_per_trade_std.append(rpt.std())
    
    ax5.errorbar(intervals, return_per_trade, yerr=return_per_trade_std, marker='o', color='purple', capsize=5, capthick=2)
    ax5.set_xlabel('Min Trade Interval (minutes)')
    ax5.set_ylabel('Return per Trade')
    ax5.set_title('Efficiency: Return per Trade vs Interval')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Box plot of returns by interval
    ax6 = axes[1, 2]
    interval_data = []
    interval_labels = []
    
    for interval, group in interval_groups:
        interval_data.append(group['daily_return_pct'].values)
        interval_labels.append(f'{interval}min')
    
    ax6.boxplot(interval_data, labels=interval_labels)
    ax6.set_xlabel('Min Trade Interval')
    ax6.set_ylabel('Daily Return %')
    ax6.set_title('Return Distribution by Interval')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis_results/parameter_analysis/trade_interval_analysis_{crypto}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_min_profit_plots(df_results, crypto):
    """Analyze impact of minimum expected profit parameter"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Minimum Expected Profit Impact Analysis - {crypto.upper()}', fontsize=16, fontweight='bold')
    
    # Group by min_expected_profit for analysis
    profit_groups = df_results.groupby('min_expected_profit')
    
    # Plot 1: Daily Return vs Min Expected Profit
    ax1 = axes[0, 0]
    profits = []
    returns_mean = []
    returns_std = []
    
    for profit, group in profit_groups:
        profits.append(profit * 100)  # Convert to percentage
        returns_mean.append(group['daily_return_pct'].mean())
        returns_std.append(group['daily_return_pct'].std())
    
    ax1.errorbar(profits, returns_mean, yerr=returns_std, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('Min Expected Profit Threshold (%)')
    ax1.set_ylabel('Daily Return %')
    ax1.set_title('Daily Return vs Min Profit Threshold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signal Filtering Efficiency
    ax2 = axes[0, 1]
    filtered_signals = []
    filtered_std = []
    
    for profit, group in profit_groups:
        total_filtered = group['signals_filtered_profit']
        filtered_signals.append(total_filtered.mean())
        filtered_std.append(total_filtered.std())
    
    ax2.errorbar(profits, filtered_signals, yerr=filtered_std, marker='s', color='orange', capsize=5, capthick=2)
    ax2.set_xlabel('Min Expected Profit Threshold (%)')
    ax2.set_ylabel('Signals Filtered by Profit')
    ax2.set_title('Signal Filtering vs Profit Threshold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Trade Quality (Win Rate vs Profit Threshold)
    ax3 = axes[0, 2]
    winrate_mean = []
    winrate_std = []
    
    for profit, group in profit_groups:
        winrate_mean.append(group['win_rate'].mean())
        winrate_std.append(group['win_rate'].std())
    
    ax3.errorbar(profits, winrate_mean, yerr=winrate_std, marker='^', color='green', capsize=5, capthick=2)
    ax3.set_xlabel('Min Expected Profit Threshold (%)')
    ax3.set_ylabel('Win Rate')
    ax3.set_title('Trade Quality vs Profit Threshold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Risk vs Reward Balance
    ax4 = axes[1, 0]
    risk_reward = []
    
    for profit, group in profit_groups:
        mean_return = group['daily_return_pct'].mean()
        max_dd = group['max_drawdown'].mean()
        rr_ratio = mean_return / (max_dd + 0.001)  # Risk-reward ratio
        risk_reward.append(rr_ratio)
    
    ax4.plot(profits, risk_reward, marker='d', color='red', linewidth=2, markersize=8)
    ax4.set_xlabel('Min Expected Profit Threshold (%)')
    ax4.set_ylabel('Return/Drawdown Ratio')
    ax4.set_title('Risk-Reward Balance vs Threshold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Fee Impact Reduction
    ax5 = axes[1, 1]
    fee_impact = []
    fee_impact_std = []
    
    for profit, group in profit_groups:
        fee_impact.append(group['fee_percentage'].mean())
        fee_impact_std.append(group['fee_percentage'].std())
    
    ax5.errorbar(profits, fee_impact, yerr=fee_impact_std, marker='o', color='purple', capsize=5, capthick=2)
    ax5.set_xlabel('Min Expected Profit Threshold (%)')
    ax5.set_ylabel('Fee Percentage %')
    ax5.set_title('Fee Impact vs Profit Threshold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Trade Selectivity (Trades vs Threshold)
    ax6 = axes[1, 2]
    trades_mean = []
    trades_std = []
    
    for profit, group in profit_groups:
        trades_mean.append(group['total_trades'].mean())
        trades_std.append(group['total_trades'].std())
    
    ax6.errorbar(profits, trades_mean, yerr=trades_std, marker='s', color='brown', capsize=5, capthick=2)
    ax6.set_xlabel('Min Expected Profit Threshold (%)')
    ax6.set_ylabel('Number of Trades')
    ax6.set_title('Trade Selectivity vs Threshold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis_results/parameter_analysis/min_profit_analysis_{crypto}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_smoothing_plots(df_results, crypto):
    """Analyze impact of signal smoothing parameter"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Signal Smoothing Impact Analysis - {crypto.upper()}', fontsize=16, fontweight='bold')
    
    # Group by smoothing window for analysis
    smoothing_groups = df_results.groupby('signal_smoothing_window')
    
    # Plot 1: Return vs Smoothing Window
    ax1 = axes[0, 0]
    windows = []
    returns_mean = []
    returns_std = []
    
    for window, group in smoothing_groups:
        windows.append(window)
        returns_mean.append(group['daily_return_pct'].mean())
        returns_std.append(group['daily_return_pct'].std())
    
    ax1.errorbar(windows, returns_mean, yerr=returns_std, marker='o', capsize=5, capthick=2)
    ax1.set_xlabel('Signal Smoothing Window')
    ax1.set_ylabel('Daily Return %')
    ax1.set_title('Daily Return vs Smoothing Window')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Signal Filtering by Smoothing
    ax2 = axes[0, 1]
    filtered_smoothing = []
    filtered_smoothing_std = []
    
    for window, group in smoothing_groups:
        filtered_smoothing.append(group['signals_filtered_smoothing'].mean())
        filtered_smoothing_std.append(group['signals_filtered_smoothing'].std())
    
    ax2.errorbar(windows, filtered_smoothing, yerr=filtered_smoothing_std, marker='s', color='orange', capsize=5, capthick=2)
    ax2.set_xlabel('Signal Smoothing Window')
    ax2.set_ylabel('Signals Filtered by Smoothing')
    ax2.set_title('Signal Filtering vs Smoothing')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Win Rate vs Smoothing (should improve with smoothing)
        ax3 = axes[1, 0]
    winrate_mean = []
    winrate_std = []
    
    for window, group in smoothing_groups:
        winrate_mean.append(group['win_rate'].mean())
        winrate_std.append(group['win_rate'].std())
    
    ax3.errorbar(windows, winrate_mean, yerr=winrate_std, marker='^', color='green', capsize=5, capthick=2)
    ax3.set_xlabel('Signal Smoothing Window')
    ax3.set_ylabel('Win Rate')
    ax3.set_title('Win Rate vs Smoothing Window')
                ax3.grid(True, alpha=0.3)
                
    # Plot 4: Return Volatility vs Smoothing
    ax4 = axes[1, 1]
    volatility = []
    
    for window, group in smoothing_groups:
        vol = group['daily_return_pct'].std()
        volatility.append(vol)
    
    ax4.plot(windows, volatility, marker='d', color='red', linewidth=2, markersize=8)
    ax4.set_xlabel('Signal Smoothing Window')
    ax4.set_ylabel('Return Volatility')
    ax4.set_title('Return Stability vs Smoothing')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'analysis_results/parameter_analysis/smoothing_analysis_{crypto}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_analysis(df_results, crypto):
    """Create correlation matrix and interaction plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Parameter Correlation Analysis - {crypto.upper()}', fontsize=16, fontweight='bold')
    
    # Select relevant columns for correlation
    corr_columns = ['signal_threshold', 'min_trade_interval', 'min_expected_profit', 
                   'signal_smoothing_window', 'daily_return_pct', 'win_rate', 
                   'total_trades', 'fee_percentage', 'max_drawdown']
    
    corr_data = df_results[corr_columns]
    
    # Plot 1: Correlation heatmap
    ax1 = axes[0, 0]
    correlation_matrix = corr_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('Parameter Correlation Matrix')
    
    # Plot 2: Parameter vs Daily Return scatter with trend lines (FIXED)
    ax2 = axes[0, 1]
    
    # Normalize parameters for comparison
    normalized_df = df_results.copy()
    param_cols = ['signal_threshold', 'min_trade_interval', 'min_expected_profit']
    
    colors = ['blue', 'red', 'green']
    for i, param in enumerate(param_cols):
        # Check if parameter has variation
        if df_results[param].nunique() > 1:
            # Normalize only if there's variation
            param_min = df_results[param].min()
            param_max = df_results[param].max()
            if param_max != param_min:
                normalized_df[f'{param}_norm'] = (df_results[param] - param_min) / (param_max - param_min)
            else:
                normalized_df[f'{param}_norm'] = 0.5  # Set to middle if no variation
            
            x = normalized_df[f'{param}_norm']
            y = df_results['daily_return_pct']
            
            # Only plot if we have valid data
            if len(x) > 1 and x.std() > 1e-10 and y.std() > 1e-10:
                ax2.scatter(x, y, alpha=0.6, color=colors[i], label=param.replace('_', ' ').title())
                
                # Add trend line with error handling
                try:
                    # Remove any NaN values
                    mask = ~(np.isnan(x) | np.isnan(y))
                    if mask.sum() > 2:  # Need at least 3 points
                        x_clean = x[mask]
                        y_clean = y[mask]
                        
                        # Only fit if there's actual variation
                        if x_clean.std() > 1e-10:
                            z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                            x_sorted = np.sort(x_clean)
                            ax2.plot(x_sorted, p(x_sorted), color=colors[i], linestyle='--', alpha=0.8)
                except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                    # Skip trend line if fitting fails
                    print(f"Warning: Could not fit trend line for {param}")
                    pass
    
    ax2.set_xlabel('Normalized Parameter Value')
    ax2.set_ylabel('Daily Return %')
    ax2.set_title('Normalized Parameters vs Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 3D-like interaction plot (Threshold vs Interval, colored by Return)
    ax3 = axes[1, 0]
    if len(df_results) > 0:
        scatter = ax3.scatter(df_results['signal_threshold'], df_results['min_trade_interval'], 
                             c=df_results['daily_return_pct'], cmap='RdYlGn', s=60, alpha=0.7)
        ax3.set_xlabel('Signal Threshold')
        ax3.set_ylabel('Min Trade Interval (minutes)')
        ax3.set_title('Threshold vs Interval (colored by Return)')
        plt.colorbar(scatter, ax=ax3, label='Daily Return %')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance efficiency scatter (FIXED)
        ax4 = axes[1, 1]
    if len(df_results) > 0 and df_results['total_trades'].max() > 0:
        efficiency_x = df_results['total_trades'] / (df_results['total_trades'].max() + 1)
        
        # Avoid division by zero in fee calculation
        fee_denominator = df_results['fee_percentage'] + 0.01  # Minimum 0.01% to avoid division issues
        efficiency_y = df_results['daily_return_pct'] / fee_denominator
        
        scatter2 = ax4.scatter(efficiency_x, efficiency_y, 
                              c=df_results['win_rate'], cmap='viridis', s=60, alpha=0.7)
        ax4.set_xlabel('Normalized Trade Frequency')
        ax4.set_ylabel('Return per Fee Ratio')
        ax4.set_title('Trading Efficiency (colored by Win Rate)')
        plt.colorbar(scatter2, ax=ax4, label='Win Rate')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'analysis_results/parameter_analysis/correlation_analysis_{crypto}.png', 
                dpi=300, bbox_inches='tight')
        plt.close()
        
def run_fee_optimized_grid_search(crypto="bitcoin"):
    """Run grid search optimized for fee reduction with comprehensive analysis"""
    
    print(f"🛡️  RUNNING FEE-OPTIMIZED GRID SEARCH FOR {crypto.upper()}")
    print("=" * 80)
    
    # UPDATED PARAMETERS based on analysis insights
    signal_thresholds = [0.06, 0.07, 0.075, 0.08, 0.09, 0.1, 0.11]  # Focused around sweet spot
    min_trade_intervals = [150, 180, 200, 220, 240, 300, 360]  # Focused on optimal range
    min_expected_profits = [0.0005, 0.001, 0.002, 0.003]  # Keep low but explore slightly higher
    signal_smoothing_windows = [2, 3, 4, 5]  # Focused on effective range
    
    results = []
    best_daily_return = -float('inf')
    best_config = None
    
    param_combinations = list(product(signal_thresholds, min_trade_intervals, 
                                    min_expected_profits, signal_smoothing_windows))
    
    print(f"🔢 Total combinations: {len(param_combinations)}")
    print(f"📈 Signal Thresholds: {signal_thresholds}")
    print(f"⏱️  Trade Intervals: {min_trade_intervals} minutes")
    print(f"💰 Min Expected Profits: {min_expected_profits}")
    print(f"🔄 Smoothing Windows: {signal_smoothing_windows}")
    
    for i, (threshold, interval, min_profit, smoothing) in enumerate(param_combinations):
        print(f"\n[{i+1}/{len(param_combinations)}] Testing: "
              f"Threshold={threshold:.3f}, Interval={interval}min, "
              f"MinProfit={min_profit:.4f}, Smoothing={smoothing}")
        
        try:
            strategy = FeeOptimizedTradingStrategy(
                initial_capital=10000,
                signal_threshold=threshold,
                maker_fee=0.0005,  # With BNB discount
                taker_fee=0.00095,  # With BNB discount
                max_position_size=0.4,
                min_trade_interval=interval,
                signal_smoothing_window=smoothing,
                min_expected_profit=min_profit,
                position_scaling=True
            )
            
            processed_markets, valid_signals = strategy.process_market_data(crypto=crypto)
            performance = strategy.generate_performance_report(verbose=True)
            
            if performance and performance['total_trades'] > 0:
                results.append(performance)
                
                if performance['daily_return_pct'] > best_daily_return:
                    best_daily_return = performance['daily_return_pct']
                    best_config = performance.copy()
        
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if not results:
        print("❌ No valid results!")
        return None
    
    # Convert to DataFrame for analysis
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values('daily_return_pct', ascending=False)
    
    # Generate comprehensive plots
    print(f"\n📊 Generating parameter analysis plots...")
    create_parameter_analysis_plots(df_results, crypto)
    
    print(f"\n{'='*140}")
    print(f"🛡️  FEE-OPTIMIZED RESULTS - {crypto.upper()}")
    print(f"{'='*140}")
    print(f"{'Threshold':<10} | {'Interval':<9} | {'MinProfit':<12} | {'Smooth':<7} | {'Return':<8} | {'Daily%':<9} | {'Trades':<7} | {'WinRate':<8} | {'Fees%':<6} | {'Hold(h)':<8}")
    print("-" * 140)
    
    for _, row in df_sorted.head(15).iterrows():  # Top 15 results
        print(f"{row['signal_threshold']:9.3f} | {row['min_trade_interval']:8.0f} | "
              f"{row['min_expected_profit']:11.4f} | {row['signal_smoothing_window']:6.0f} | "
              f"{row['total_return']:7.1%} | {row['daily_return_pct']:8.3f} | "
              f"{row['total_trades']:6.0f} | {row['win_rate']:7.1%} | "
              f"{row['fee_percentage']:5.2f} | {row['avg_hold_time_hours']:7.1f}")
    
    if best_config:
        print(f"\n{'='*60}")
        print("🏆 BEST FEE-OPTIMIZED CONFIGURATION")
        print(f"{'='*60}")
        print(f"📊 Signal Threshold: {best_config['signal_threshold']:.3f}")
        print(f"⏱️  Trade Interval: {best_config['min_trade_interval']:.0f} minutes")
        print(f"💰 Min Expected Profit: {best_config['min_expected_profit']:.4f}")
        print(f"🔄 Smoothing Window: {best_config['signal_smoothing_window']:.0f}")
        print(f"📈 Daily Return: {best_config['daily_return_pct']:+.3f}%/day")
        print(f"🎯 Win Rate: {best_config['win_rate']:.1%}")
        print(f"💸 Fee Efficiency: {100-best_config['fee_percentage']:.1f}%")
        print(f"🎯 Total Trades: {best_config['total_trades']:.0f}")
        print(f"⏰ Avg Hold Time: {best_config['avg_hold_time_hours']:.1f} hours")
        print(f"🛡️  Filter Efficiency:")
        print(f"   - Timing: {best_config['signals_filtered_timing']:.0f}")
        print(f"   - Smoothing: {best_config['signals_filtered_smoothing']:.0f}")
        print(f"   - Profit: {best_config['signals_filtered_profit']:.0f}")
        
        # Calculate risk metrics
        risk_reward_ratio = best_config['daily_return_pct'] / (best_config['max_drawdown'] + 0.001)
        trade_efficiency = best_config['total_return'] / (best_config['total_trades'] + 1)
        
        print(f"\n📊 ADDITIONAL METRICS:")
        print(f"   - Risk/Reward Ratio: {risk_reward_ratio:.2f}")
        print(f"   - Return per Trade: {trade_efficiency:.4f}")
        print(f"   - Max Drawdown: {best_config['max_drawdown']:.2%}")
        print(f"   - Avg Fee per Trade: ${best_config['avg_fee_per_trade']:.2f}")
    
    # Performance summary statistics
    print(f"\n📈 GRID SEARCH SUMMARY:")
    print(f"   - Configurations Tested: {len(param_combinations)}")
    print(f"   - Successful Runs: {len(results)}")
    print(f"   - Success Rate: {len(results)/len(param_combinations)*100:.1f}%")
    print(f"   - Avg Daily Return: {df_results['daily_return_pct'].mean():.3f}%")
    print(f"   - Best Daily Return: {df_results['daily_return_pct'].max():.3f}%")
    print(f"   - Avg Win Rate: {df_results['win_rate'].mean():.1%}")
    print(f"   - Avg Trades per Config: {df_results['total_trades'].mean():.1f}")
    
    return df_results

# Also add a function to run focused optimization around best parameters
def run_focused_optimization(crypto="bitcoin", base_config=None):
    """Run focused optimization around a base configuration"""
    
    if base_config is None:
        # Use the best config from previous analysis
        base_config = {
            'signal_threshold': 0.075,
            'min_trade_interval': 200,
            'min_expected_profit': 0.001,
            'signal_smoothing_window': 3
        }
    
    print(f"🎯 RUNNING FOCUSED OPTIMIZATION FOR {crypto.upper()}")
    print(f"Base config: {base_config}")
    print("=" * 80)
    
    # Fine-tuned parameters around the best configuration
    base_threshold = base_config['signal_threshold']
    base_interval = base_config['min_trade_interval']
    base_profit = base_config['min_expected_profit']
    base_smoothing = base_config['signal_smoothing_window']
    
    signal_thresholds = [base_threshold * x for x in [0.9, 0.95, 1.0, 1.05, 1.1]]
    min_trade_intervals = [int(base_interval * x) for x in [0.8, 0.9, 1.0, 1.1, 1.2]]
    min_expected_profits = [base_profit * x for x in [0.5, 0.75, 1.0, 1.5, 2.0]]
    signal_smoothing_windows = [max(1, base_smoothing + x) for x in [-1, 0, 1, 2]]
    
    results = []
    param_combinations = list(product(signal_thresholds, min_trade_intervals, 
                                    min_expected_profits, signal_smoothing_windows))
    
    print(f"🔍 Fine-tuning {len(param_combinations)} combinations...")
    
    for i, (threshold, interval, min_profit, smoothing) in enumerate(param_combinations):
        try:
            strategy = FeeOptimizedTradingStrategy(
                initial_capital=10000,
            signal_threshold=threshold, 
                maker_fee=0.0005,
                taker_fee=0.00095,
                max_position_size=0.4,
                min_trade_interval=interval,
                signal_smoothing_window=smoothing,
                min_expected_profit=min_profit,
                position_scaling=True
            )
            
            processed_markets, valid_signals = strategy.process_market_data(crypto=crypto)
            performance = strategy.generate_performance_report(verbose=False)
            
            if performance and performance['total_trades'] > 0:
                results.append(performance)
        
        except Exception as e:
            continue
    
    if results:
        df_results = pd.DataFrame(results)
        best_result = df_results.loc[df_results['daily_return_pct'].idxmax()]
        
        print(f"\n🏆 OPTIMIZED CONFIGURATION:")
        print(f"📊 Signal Threshold: {best_result['signal_threshold']:.4f}")
        print(f"⏱️  Trade Interval: {best_result['min_trade_interval']:.0f} minutes")
        print(f"💰 Min Expected Profit: {best_result['min_expected_profit']:.5f}")
        print(f"🔄 Smoothing Window: {best_result['signal_smoothing_window']:.0f}")
        print(f"📈 Daily Return: {best_result['daily_return_pct']:+.4f}%/day")
        print(f"🎯 Win Rate: {best_result['win_rate']:.2%}")
        
        return best_result.to_dict()
    
    return None

# Update the main execution
if __name__ == "__main__":
    # Run comprehensive grid search
    print("🚀 Starting comprehensive grid search...")
    btc_results = run_fee_optimized_grid_search("bitcoin")
    eth_results = run_fee_optimized_grid_search("ethereum")
    
    # Run focused optimization on best performers
    if btc_results is not None and len(btc_results) > 0:
        best_btc = btc_results.loc[btc_results['daily_return_pct'].idxmax()]
        print(f"\n🎯 Running focused optimization for Bitcoin...")
        optimized_btc = run_focused_optimization("bitcoin", best_btc.to_dict())
    
    if eth_results is not None and len(eth_results) > 0:
        best_eth = eth_results.loc[eth_results['daily_return_pct'].idxmax()]
        print(f"\n🎯 Running focused optimization for Ethereum...")
        optimized_eth = run_focused_optimization("ethereum", best_eth.to_dict())
    