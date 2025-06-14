"""
Core trading bot functions for Polymarket signal-based Bitcoin trading
"""

import os
import json
import time
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# Import existing utilities
from utils import (
    read_json_file, sigmoid, fit_hourly_sigmoid, 
    calculate_hours_remaining, get_market_midpoint_probability,
    get_btc_price_from_point
)
from get_market_token import get_current_market_token
from data_collector import PolymarketHelper

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingBotCore:
    def __init__(self):
        """Initialize the trading bot with configuration"""
        self.config = self.load_config()
        self.state = self.load_state()
        self.polymarket_helper = PolymarketHelper()
        
        # Trading parameters from analysis
        self.signal_threshold = 0.080
        self.min_trade_interval = 200  # minutes
        self.min_expected_profit = 0.0005
        self.smoothing_window = 4
        self.min_hours = 1
        self.max_hours = 15
        
        # Risk management
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.15'))
        self.initial_capital = float(os.getenv('INITIAL_CAPITAL', '10000'))
        
        # Signal smoothing
        self.signal_history = []
        
        # Load sigmoid parameters from collected data
        self.sigmoid_params = self.load_sigmoid_parameters()
    
    def load_config(self) -> Dict:
        """Load configuration from environment and files"""
        return {
            'binance_api_key': os.getenv('BINANCE_API_KEY'),
            'binance_secret_key': os.getenv('BINANCE_SECRET_KEY'),
            'binance_testnet': os.getenv('BINANCE_TESTNET', 'false').lower() == 'true',
            'initial_capital': float(os.getenv('INITIAL_CAPITAL', '10000')),
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.15')),
        }
    
    def load_state(self) -> Dict:
        """Load bot state from JSON file"""
        state_file = Path('bot_state.json')
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                logger.info("Loaded existing bot state")
                return state
            except Exception as e:
                logger.error(f"Error loading state: {e}")
        
        # Initialize default state
        default_state = {
            'portfolio_value': self.config['initial_capital'],
            'current_position': None,
            'last_trade_time': None,
            'total_trades': 0,
            'profitable_trades': 0,
            'current_market_id': None,
            'signal_history': [],
            'trade_history': []
        }
        self.save_state(default_state)
        return default_state
    
    def save_state(self, state: Dict = None):
        """Save bot state to JSON file"""
        if state is None:
            state = self.state
        
        try:
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_sigmoid_parameters(self) -> Dict:
        """Load sigmoid parameters from collected data"""
        logger.info("Loading sigmoid parameters from collected data...")
        
        try:
            # Use existing functions to process collected data
            from glob import glob
            
            data_files = glob("collected_data/bitcoin-up-or-down-*.json")
            if not data_files:
                logger.error("No collected data files found!")
                return {}
            
            # Take the most recent market for sigmoid fitting
            latest_file = max(data_files, key=os.path.getctime)
            logger.info(f"Using {latest_file} for sigmoid parameters")
            
            data = read_json_file(latest_file)
            if not data:
                logger.error("Could not read data file")
                return {}
            
            # Extract hourly data for sigmoid fitting
            end_timestamp = data.get("collection_metadata", {}).get("collection_period", {}).get("end")
            if not end_timestamp:
                logger.error("No end timestamp found")
                return {}
            
            hourly_data = {}
            for point in data.get("data_points", []):
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
                logger.error("Insufficient hourly data for sigmoid fitting")
                return {}
            
            # Fit sigmoid
            params = fit_hourly_sigmoid(hourly_averages)
            if params is None:
                logger.error("Failed to fit sigmoid")
                return {}
            
            logger.info(f"Sigmoid parameters loaded: L={params[0]:.4f}, k={params[1]:.4f}, x0={params[2]:.4f}, b={params[3]:.4f}")
            return {
                'L': params[0],
                'k': params[1], 
                'x0': params[2],
                'b': params[3],
                'fitted_from': latest_file,
                'fitted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading sigmoid parameters: {e}")
            return {}
    
    def get_current_btc_price(self) -> Optional[float]:
        """Get current Bitcoin price from Binance"""
        try:
            response = requests.get(
                'https://api.binance.com/api/v3/ticker/price',
                params={'symbol': 'BTCUSDT'},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            else:
                logger.error(f"Binance API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching BTC price: {e}")
            return None
    
    def get_current_market_data(self) -> Optional[Dict]:
        """Get current Polymarket data for Bitcoin up/down"""
        try:
            # Get current market info
            markets = get_current_market_token(crypto_name="bitcoin", market_type="daily_up_down")
            print(markets)
            input()
            if not markets:
                logger.error("No current Bitcoin markets found")
                return None
            
            # Get the active market
            current_market = markets[0]  # Assuming first is the active one
            token_id = current_market.get('token_id')
            
            if not token_id:
                logger.error("No token ID found in market data")
                return None
            
            # Get orderbook
            orderbook = self.polymarket_helper.get_order_book(token_id)
            if not orderbook:
                logger.error("Could not fetch orderbook")
                return None
            
            # Calculate market probability
            market_prob = get_market_midpoint_probability(orderbook)
            if market_prob is None:
                logger.error("Could not calculate market probability")
                return None
            
            # Calculate hours remaining
            end_time = current_market.get('end_date_iso')
            if not end_time:
                logger.error("No end time found in market data")
                return None
            
            end_timestamp = pd.to_datetime(end_time).timestamp()
            current_timestamp = datetime.now().timestamp()
            hours_remaining = (end_timestamp - current_timestamp) / 3600
            
            return {
                'market_id': current_market.get('market_slug'),
                'token_id': token_id,
                'market_prob': market_prob,
                'hours_remaining': hours_remaining,
                'end_timestamp': end_timestamp,
                'orderbook': orderbook
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def calculate_signal(self, market_data: Dict, btc_price: float) -> Optional[Dict]:
        """Calculate trading signal based on sigmoid deviation"""
        try:
            if not self.sigmoid_params:
                logger.error("No sigmoid parameters available")
                return None
            
            hours_remaining = market_data['hours_remaining']
            market_prob = market_data['market_prob']
            
            # Check if within trading window
            if hours_remaining < self.min_hours or hours_remaining > self.max_hours:
                return {
                    'signal': 'HOLD',
                    'strength': 0.0,
                    'reason': f'Outside trading window ({hours_remaining:.1f}h)'
                }
            
            # Calculate expected probability using sigmoid
            expected_prob = sigmoid(
                hours_remaining,
                self.sigmoid_params['L'],
                self.sigmoid_params['k'],
                self.sigmoid_params['x0'],
                self.sigmoid_params['b']
            )
            
            # Calculate linear distance (main signal)
            linear_distance = market_prob - expected_prob
            
            # Apply signal threshold
            if abs(linear_distance) < self.signal_threshold:
                return {
                    'signal': 'HOLD',
                    'strength': abs(linear_distance),
                    'reason': f'Below threshold ({abs(linear_distance):.4f} < {self.signal_threshold})'
                }
            
            # Determine signal direction and strength
            signal_direction = 'SHORT' if linear_distance > 0 else 'LONG'
            signal_strength = min(abs(linear_distance) / 0.1, 1.0)  # Normalize to 0-1
            
            return {
                'signal': signal_direction,
                'strength': signal_strength,
                'linear_distance': linear_distance,
                'market_prob': market_prob,
                'expected_prob': expected_prob,
                'hours_remaining': hours_remaining,
                'btc_price': btc_price
            }
            
        except Exception as e:
            logger.error(f"Error calculating signal: {e}")
            return None
    
    def smooth_signal(self, new_signal: Dict) -> Optional[Dict]:
        """Apply signal smoothing to reduce noise"""
        try:
            self.signal_history.append(new_signal)
            
            # Keep only recent history
            if len(self.signal_history) > self.smoothing_window * 2:
                self.signal_history = self.signal_history[-self.smoothing_window * 2:]
            
            if len(self.signal_history) < self.smoothing_window:
                return None  # Not enough data
            
            # Get recent signals
            recent_signals = self.signal_history[-self.smoothing_window:]
            
            # Check signal consistency
            signals = [s['signal'] for s in recent_signals if s['signal'] != 'HOLD']
            
            if len(signals) < self.smoothing_window * 0.6:  # Less than 60% actual signals
                return {'signal': 'HOLD', 'reason': 'Insufficient signal consistency'}
            
            # Check direction consistency
            if signals:
                direction_consistency = signals.count(new_signal['signal']) / len(signals)
                if direction_consistency < 0.7:  # Less than 70% consistency
                    return {'signal': 'HOLD', 'reason': 'Direction inconsistency'}
            
            # Calculate smoothed strength
            smoothed_strength = np.mean([s['strength'] for s in recent_signals])
            
            return {
                'signal': new_signal['signal'],
                'strength': smoothed_strength,
                'linear_distance': new_signal.get('linear_distance'),
                'market_prob': new_signal.get('market_prob'),
                'expected_prob': new_signal.get('expected_prob'),
                'hours_remaining': new_signal.get('hours_remaining'),
                'btc_price': new_signal.get('btc_price'),
                'smoothed': True
            }
            
        except Exception as e:
            logger.error(f"Error smoothing signal: {e}")
            return None
    
    def can_trade_now(self) -> bool:
        """Check if enough time has passed since last trade"""
        if not self.state.get('last_trade_time'):
            return True
        
        last_trade = pd.to_datetime(self.state['last_trade_time'])
        time_since_last = datetime.now() - last_trade
        
        return time_since_last.total_seconds() >= (self.min_trade_interval * 60)
    
    def calculate_position_size(self, signal_strength: float) -> float:
        """Calculate position size based on signal strength"""
        base_position = 0.15  # From analysis
        position_multiplier = 1 + (signal_strength ** 2) * 2
        
        position_size = min(base_position * position_multiplier, self.max_position_size)
        return position_size
    
    def estimate_expected_profit(self, signal_strength: float, hours_remaining: float) -> float:
        """Estimate expected profit based on historical analysis"""
        base_move_pct = 0.02  # 2% base expected move
        expected_move = base_move_pct * signal_strength
        
        # Time decay factor
        time_factor = 1.0 + (self.max_hours - hours_remaining) / self.max_hours * 0.5
        expected_move *= time_factor
        
        return expected_move
    
    def binance_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False):
        """Make authenticated request to Binance API"""
        if not self.config['binance_api_key'] or not self.config['binance_secret_key']:
            raise Exception("Binance API credentials not configured")
        
        base_url = 'https://testnet.binance.vision' if self.config['binance_testnet'] else 'https://api.binance.com'
        url = f"{base_url}{endpoint}"
        
        headers = {'X-MBX-APIKEY': self.config['binance_api_key']}
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = hmac.new(
                self.config['binance_secret_key'].encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
        
        response = requests.request(method, url, headers=headers, params=params, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"Binance API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def get_account_info(self) -> Dict:
        """Get Binance account information"""
        return self.binance_request('GET', '/api/v3/account', signed=True)
    
    def place_order(self, symbol: str, side: str, quantity: float, order_type: str = 'MARKET') -> Dict:
        """Place order on Binance"""
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity
        }
        
        return self.binance_request('POST', '/api/v3/order', params=params, signed=True)
    
    def execute_trade(self, signal_data: Dict) -> bool:
        """Execute trade based on signal"""
        try:
            # Check if we can trade
            if not self.can_trade_now():
                logger.info("Trade interval not met, skipping")
                return False
            
            signal = signal_data['signal']
            strength = signal_data['strength']
            btc_price = signal_data['btc_price']
            
            # Calculate position size
            position_size_pct = self.calculate_position_size(strength)
            position_value = self.state['portfolio_value'] * position_size_pct
            
            # Estimate expected profit
            expected_profit = self.estimate_expected_profit(
                strength, signal_data['hours_remaining']
            )
            
            if expected_profit < self.min_expected_profit:
                logger.info(f"Expected profit too low: {expected_profit:.4f}")
                return False
            
            # Close existing position if any
            if self.state['current_position']:
                self.close_position()
            
            # Calculate quantity for BTC
            quantity = position_value / btc_price
            quantity = round(quantity, 6)  # Round to 6 decimal places for BTC
            
            # Determine order side
            side = 'BUY' if signal == 'LONG' else 'SELL'
            
            logger.info(f"Executing {side} order: {quantity} BTC at ~${btc_price:,.2f}")
            logger.info(f"Signal strength: {strength:.4f}, Expected profit: {expected_profit:.4f}")
            
            # Place order (commented out for safety - uncomment when ready to trade)
            # order_result = self.place_order('BTCUSDT', side, quantity)
            # logger.info(f"Order result: {order_result}")
            
            # For now, simulate the trade
            order_result = {
                'orderId': f"SIM_{int(time.time())}",
                'executedQty': str(quantity),
                'status': 'FILLED'
            }
            
            # Update state
            self.state['current_position'] = {
                'side': side,
                'quantity': quantity,
                'entry_price': btc_price,
                'entry_time': datetime.now().isoformat(),
                'signal_data': signal_data,
                'order_id': order_result['orderId']
            }
            
            self.state['last_trade_time'] = datetime.now().isoformat()
            self.state['total_trades'] += 1
            
            # Add to trade history
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'OPEN',
                'side': side,
                'quantity': quantity,
                'price': btc_price,
                'signal_strength': strength,
                'expected_profit': expected_profit,
                'signal_data': signal_data
            }
            
            self.state['trade_history'].append(trade_record)
            self.save_state()
            
            logger.info(f"✅ Trade executed successfully: {side} {quantity} BTC")
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def close_position(self) -> bool:
        """Close current position"""
        try:
            if not self.state['current_position']:
                return True
            
            position = self.state['current_position']
            current_price = self.get_current_btc_price()
            
            if not current_price:
                logger.error("Could not get current BTC price for closing position")
                return False
            
            # Determine close side (opposite of open)
            close_side = 'SELL' if position['side'] == 'BUY' else 'BUY'
            quantity = position['quantity']
            
            logger.info(f"Closing position: {close_side} {quantity} BTC at ~${current_price:,.2f}")
            
            # Place close order (commented out for safety)
            # close_result = self.place_order('BTCUSDT', close_side, quantity)
            
            # Simulate close
            close_result = {
                'orderId': f"SIM_CLOSE_{int(time.time())}",
                'executedQty': str(quantity),
                'status': 'FILLED'
            }
            
            # Calculate P&L
            entry_price = position['entry_price']
            if position['side'] == 'BUY':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            pnl_usd = pnl_pct * (quantity * entry_price)
            
            # Update portfolio value
            self.state['portfolio_value'] += pnl_usd
            
            if pnl_usd > 0:
                self.state['profitable_trades'] += 1
            
            # Add to trade history
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'action': 'CLOSE',
                'side': close_side,
                'quantity': quantity,
                'price': current_price,
                'pnl_usd': pnl_usd,
                'pnl_pct': pnl_pct,
                'hold_time_hours': (datetime.now() - pd.to_datetime(position['entry_time'])).total_seconds() / 3600
            }
            
            self.state['trade_history'].append(trade_record)
            
            # Clear current position
            self.state['current_position'] = None
            self.save_state()
            
            logger.info(f"✅ Position closed. P&L: ${pnl_usd:.2f} ({pnl_pct*100:.2f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def run_cycle(self):
        """Run one trading cycle"""
        try:
            logger.info("=== Starting Trading Cycle ===")
            
            # Get current market data
            market_data = self.get_current_market_data()
            if not market_data:
                logger.error("Could not get market data")
                return
            
            # Get current BTC price
            btc_price = self.get_current_btc_price()
            if not btc_price:
                logger.error("Could not get BTC price")
                return
            
            logger.info(f"BTC Price: ${btc_price:,.2f}")
            logger.info(f"Market Prob: {market_data['market_prob']:.4f}")
            logger.info(f"Hours Remaining: {market_data['hours_remaining']:.1f}")
            
            # Calculate signal
            signal_data = self.calculate_signal(market_data, btc_price)
            if not signal_data:
                logger.error("Could not calculate signal")
                return
            
            logger.info(f"Raw Signal: {signal_data['signal']} (strength: {signal_data['strength']:.4f})")
            
            # Apply signal smoothing
            smoothed_signal = self.smooth_signal(signal_data)
            if not smoothed_signal:
                logger.info("Signal not ready (smoothing)")
                return
            
            if smoothed_signal['signal'] == 'HOLD':
                logger.info(f"Signal: HOLD ({smoothed_signal.get('reason', 'smoothing')})")
                return
            
            logger.info(f"Smoothed Signal: {smoothed_signal['signal']} (strength: {smoothed_signal['strength']:.4f})")
            
            # Execute trade if signal is strong enough
            if smoothed_signal['strength'] >= self.signal_threshold:
                success = self.execute_trade(smoothed_signal)
                if success:
                    logger.info("Trade executed successfully")
                else:
                    logger.error("Trade execution failed")
            else:
                logger.info(f"Signal too weak: {smoothed_signal['strength']:.4f}")
            
            # Update state with current cycle info
            self.state['last_cycle'] = datetime.now().isoformat()
            self.state['current_market_id'] = market_data['market_id']
            self.save_state()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'portfolio_value': self.state['portfolio_value'],
            'total_trades': self.state['total_trades'],
            'profitable_trades': self.state['profitable_trades'],
            'win_rate': self.state['profitable_trades'] / max(self.state['total_trades'], 1),
            'current_position': self.state['current_position'],
            'last_cycle': self.state.get('last_cycle'),
            'sigmoid_params_loaded': bool(self.sigmoid_params)
        }
