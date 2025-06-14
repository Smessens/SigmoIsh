#!/usr/bin/env python3
"""
Main trading bot script - runs the minute-by-minute trading cycle
"""

import time
import signal
import sys
from datetime import datetime
import logging
from core import TradingBotCore
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.core = TradingBotCore()
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self):
        """Main bot loop"""
        logger.info("🚀 Starting Polymarket Signal Trading Bot")
        logger.info(f"Signal threshold: {self.core.signal_threshold}")
        logger.info(f"Trading window: {self.core.min_hours}-{self.core.max_hours} hours")
        
        # Print initial status
        status = self.core.get_status()
        logger.info(f"Initial portfolio value: ${status['portfolio_value']:,.2f}")
        logger.info(f"Total trades: {status['total_trades']}")
        logger.info(f"Win rate: {status['win_rate']*100:.1f}%")
        
        if not status['sigmoid_params_loaded']:
            logger.warning("⚠️  No sigmoid parameters loaded - check collected data")
        
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                logger.info(f"\n=== Cycle {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                
                # Run trading cycle
                self.core.run_cycle()
                
                # Print status every 10 cycles
                if cycle_count % 10 == 0:
                    status = self.core.get_status()
                    logger.info(f"📊 Status Update - Portfolio: ${status['portfolio_value']:,.2f}, "
                              f"Trades: {status['total_trades']}, Win Rate: {status['win_rate']*100:.1f}%")
                
                # Wait 60 seconds for next cycle
                logger.info("⏰ Waiting 60 seconds for next cycle...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.info("Continuing after error...")
                time.sleep(60)  # Wait before retrying
        
        # Cleanup
        logger.info("🛑 Bot stopping...")
        
        # Close any open positions
        if self.core.state.get('current_position'):
            logger.info("Closing open position before shutdown...")
            self.core.close_position()
        
        # Final status
        final_status = self.core.get_status()
        logger.info(f"Final portfolio value: ${final_status['portfolio_value']:,.2f}")
        logger.info(f"Total trades executed: {final_status['total_trades']}")
        logger.info(f"Final win rate: {final_status['win_rate']*100:.1f}%")
        
        logger.info("👋 Trading bot stopped successfully")

    def calculate_position_size_small_account(self, signal_strength: float, account_balance: float) -> float:
        """Calculate position size for small accounts (under $500)"""
        
        # For accounts under $500, use more conservative sizing
        if account_balance < 500:
            # Use 5-15% of account per trade instead of the original 15%
            base_position = 0.05  # 5% base position
            max_position = 0.15   # 15% maximum
            
            # Scale based on signal strength
            position_multiplier = 1 + (signal_strength ** 2) * 2
            position_size = min(base_position * position_multiplier, max_position)
            
            # Ensure minimum order value is met ($10 minimum)
            min_order_value = 10.0  # $10 minimum
            position_value = account_balance * position_size
            
            if position_value < min_order_value:
                # If calculated position is too small, use minimum viable position
                position_size = min_order_value / account_balance
                
                # But don't exceed maximum position size
                if position_size > max_position:
                    return 0  # Skip trade if minimum order would be too large
            
            return position_size
        else:
            # Use original logic for larger accounts
            return self.calculate_position_size(signal_strength)

    def validate_order_size(self, quantity: float, price: float) -> Tuple[bool, str]:
        """Validate if order meets Binance minimum requirements"""
        
        notional_value = quantity * price
        min_notional = 10.0  # $10 minimum (conservative estimate)
        min_quantity = 0.00001  # Minimum BTC quantity
        
        if notional_value < min_notional:
            return False, f"Order value ${notional_value:.2f} below minimum ${min_notional}"
        
        if quantity < min_quantity:
            return False, f"Quantity {quantity:.8f} below minimum {min_quantity}"
        
        return True, "Order size valid"

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
