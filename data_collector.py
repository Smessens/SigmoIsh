import requests
from get_market_token import get_current_market_token
import json
import os
from dotenv import load_dotenv
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.constants import POLYGON
from py_clob_client.order_builder.constants import BUY, SELL
import time
from datetime import datetime, timedelta
import pytz
import subprocess
import logging
import signal
from functools import wraps
import threading
import queue

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Setup file handler (daily rotating log file)
    log_filename = f"logs/market_data_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('market_data')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def robust_timeout_wrapper(func, timeout_seconds, default_return=None):
    """
    A more robust timeout wrapper using threading instead of signals
    Works better with network operations and cross-platform
    """
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def target():
        try:
            result = func()
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, it timed out
        logger.warning(f"Function {func.__name__ if hasattr(func, '__name__') else 'anonymous'} timed out after {timeout_seconds} seconds")
        return default_return
    
    # Check if there was an exception
    if not exception_queue.empty():
        raise exception_queue.get()
    
    # Get the result
    if not result_queue.empty():
        return result_queue.get()
    
    return default_return

def safe_network_call(func, max_retries=3, retry_delay=2, timeout=15, default_return=None, operation_name="network operation"):
    """
    A comprehensive wrapper for network calls with timeout, retry, and error isolation
    """
    for attempt in range(max_retries):
        try:
            # Use robust timeout wrapper
            result = robust_timeout_wrapper(func, timeout, default_return)
            if result is not default_return or default_return is None:
                return result
            else:
                raise Exception(f"Operation returned default value, treating as failure")
                
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {operation_name}: {str(e)}")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                continue
            else:
                logger.error(f"All {max_retries} attempts failed for {operation_name}: {str(e)}")
                return default_return

def timeout_handler(signum, frame):
    """Handler for timeout signal (legacy, kept for compatibility)"""
    raise TimeoutError("Operation timed out")

def with_timeout(timeout_seconds=30):
    """Decorator to add timeout to any function (legacy, kept for compatibility)"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use the new robust timeout wrapper instead
            def target_func():
                return func(*args, **kwargs)
            return robust_timeout_wrapper(target_func, timeout_seconds)
        return wrapper
    return decorator

# Create logger instance
logger = setup_logging()

# Load environment variables
load_dotenv()
BRANCH_NAME = os.getenv('branch_name', 'main')  # Default to 'main' if not specified

class PolymarketHelper:
    def __init__(self):
        load_dotenv()
        self.host = "https://clob.polymarket.com"
        self.gamma_api = "https://gamma-api.polymarket.com"
        self.chain_id = POLYGON
        self.key = os.getenv('KEY')        
        self.funder = os.getenv('FUNDER')
        self.client = None
        self._setup_client_safe()

    def _setup_client_safe(self):
        """Initialize and setup the CLOB client with robust error handling"""
        def setup_client_internal():
            client = ClobClient(
                self.host,
                key=self.key,
                chain_id=self.chain_id,
                funder=self.funder,
                signature_type=1,
            )

            api_creds = client.create_or_derive_api_creds()
            time.sleep(1)  # Add small delay
            client.set_api_creds(api_creds)
            
            # Verify the client is authenticated
            orders = client.get_orders()  # Test API connection
            logger.info(f"Number of orders: {len(orders)}")
            return client
        
        # Use safe network call for client setup
        self.client = safe_network_call(
            setup_client_internal,
            max_retries=3,
            retry_delay=5,
            timeout=30,
            operation_name="client setup"
        )
        
        if self.client is None:
            logger.error("Failed to initialize client after all retries. Will continue with degraded functionality.")
    
    def _get_order_book_internal(self, token_id: str):
        """Internal order book fetch without timeout decoration"""
        if self.client is None:
            raise Exception("Client not initialized")
        return self.client.get_order_book(token_id)
    
    def get_order_book(self, token_id: str) -> dict:
        """Fetch order book for a specific market token ID using authenticated client with robust retry mechanism."""
        if self.client is None:
            logger.warning(f"Client not available, returning empty order book for token {token_id}")
            return {"bids": [], "asks": []}
        
        def fetch_orderbook():
            time.sleep(0.01)  # Small delay to avoid rate limiting
            order_book = self._get_order_book_internal(token_id)
            
            if order_book is None:
                raise Exception("Order book request returned None")
            
            # Convert to list of dicts and sort by price
            bids = [{"price": float(bid.price), "size": float(bid.size)} for bid in order_book.bids]
            asks = [{"price": float(ask.price), "size": float(ask.size)} for ask in order_book.asks]
            # Sort bids in descending order (highest first)
            bids = sorted(bids, key=lambda x: x['price'], reverse=True)
            # Sort asks in ascending order (lowest first)
            asks = sorted(asks, key=lambda x: x['price'])
            return {
                "bids": bids, 
                "asks": asks
            }
        
        return safe_network_call(
            fetch_orderbook,
            max_retries=3,
            retry_delay=2,
            timeout=15,
            default_return={"bids": [], "asks": []},
            operation_name=f"order book for token {token_id}"
        )

    def get_current_price(self, symbol):
        """Get current price from Binance with robust retry mechanism and timeout"""
        def fetch_price():
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url, timeout=10)  # Shorter timeout for faster recovery
            response.raise_for_status()
            return float(response.json()['price'])
        
        return safe_network_call(
            fetch_price,
            max_retries=4,  # More retries for price data since it's critical
            retry_delay=3,
            timeout=10,
            operation_name=f"price for {symbol}"
        )

    def collect_market_data(self, market_details, current_price):
        """Collect market data with error handling and timeout protection"""
        if not market_details:
            return None
        
        try:
            token_yes = market_details.get('token_yes')
            token_no = market_details.get('token_no')
            
            if not token_yes or not token_no:
                logger.warning(f"Missing token IDs for market {market_details.get('main_market_slug')}")
                return None
            
            # Get order books with retry mechanism and timeout
            order_book_yes = self.get_order_book(token_yes)
            order_book_no = self.get_order_book(token_no)
            
            # If either order book is None, skip this data point
            if not order_book_yes or not order_book_no:
                logger.warning(f"Skipping data collection for {market_details.get('main_market_slug')} due to missing order book data")
                return None
            
            # Process the data as before
            data_point = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'current_price': current_price,
                'yes_best_bid': order_book_yes['bids'][0]['price'] if order_book_yes['bids'] else None,
                'yes_best_ask': order_book_yes['asks'][0]['price'] if order_book_yes['asks'] else None,
                'no_best_bid': order_book_no['bids'][0]['price'] if order_book_no['bids'] else None,
                'no_best_ask': order_book_no['asks'][0]['price'] if order_book_no['asks'] else None
            }
            
            # Only return data point if we have at least some valid price data
            if any(v is not None for v in data_point.values()):
                return data_point
            return None
        
        except Exception as e:
            logger.error(f"Error collecting market data for {market_details.get('main_market_slug')}: {e}")
            return None

def get_crypto_price(symbol="BTCUSDT"):
    """
    Get the current crypto price from Binance with timeout and retry
    Args:
        symbol (str): Trading symbol for Binance (e.g., "BTCUSDT", "ETHUSDT")
    Returns: float - current crypto price
    """
    def fetch_price():
        # Binance API endpoint for crypto price
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
        
        # Make the request with timeout
        response = requests.get(url, timeout=10)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        
        # Extract and return the price as a float
        return float(data['price'])
    
    return safe_network_call(
        fetch_price,
        max_retries=4,
        retry_delay=3,
        timeout=10,
        operation_name=f"crypto price for {symbol}"
    )

def ensure_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = "collected_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

def get_current_market_period():
    """
    Determine the current market period based on ET timezone
    Returns:
        tuple: (start_date, end_date, is_new_period_needed)
    """
    et_tz = pytz.timezone('America/New_York')
    
    # Get current time in ET directly
    now_et = datetime.now(et_tz)
    
    # Market period starts at 12:00 ET (noon) the day before
    # and ends at 12:00 ET (noon) the current day
    current_period_end = now_et.replace(hour=12, minute=0, second=0, microsecond=0)
    
    # If current time is before noon, the period started yesterday
    if now_et.hour < 12:
        current_period_end = current_period_end
        current_period_start = current_period_end - timedelta(days=1)
    else:
        # If current time is after noon, the period started today and ends tomorrow
        current_period_start = current_period_end
        current_period_end = current_period_end + timedelta(days=1)
    
    # Check if we're within 1 minute of the period transition
    minutes_to_transition = abs((current_period_end - now_et).total_seconds() / 60)
    is_new_period_needed = minutes_to_transition <= 1
    
    return current_period_start, current_period_end, is_new_period_needed

def get_output_filename(market_info):
    """Generate filename based on market info"""
    if not market_info:
        # Fallback filename if market_info is None
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"collected_data/unknown_market_{timestamp}.json"
    
    # For monthly markets, create subdirectory based on general event slug
    if market_info.get("general_event_slug") != market_info.get("main_market_slug"):
        # This is a monthly market (general_event_slug is different from specific market slug)
        event_dir = f"collected_data/{market_info['general_event_slug']}"
        os.makedirs(event_dir, exist_ok=True)
        return f"{event_dir}/{market_info['main_market_slug']}.json"
    else:
        # This is a daily market (general_event_slug same as market slug)
        return f"collected_data/{market_info['main_market_slug']}.json"

def initialize_market_details(crypto_name="bitcoin", market_type="daily_up_down"):
    """Initialize market identification details for a given crypto with robust error handling"""
    def get_markets():
        return get_current_market_token(crypto_name=crypto_name, market_type=market_type)
    
    markets = safe_network_call(
        get_markets,
        max_retries=3,
        retry_delay=3,
        timeout=20,
        default_return=[],
        operation_name=f"market details for {crypto_name} {market_type}"
    )
    
    if not markets:
        return None
        
    # Convert the new format to match what the rest of the code expects
    market_details_list = []
    for market in markets:
        details = {
            "slug": market["main_market_slug"],  # For backwards compatibility
            "market_id": market["main_market_id"],
            "up_token": market["token_yes"],
            "down_token": market["token_no"],
            "crypto_name": crypto_name,
            # New fields for enhanced structure
            "main_market_slug": market["main_market_slug"],
            "general_event_slug": market["general_event_slug"],
            "event_question": market["event_question"],
            "market_start_date_iso": market["market_start_date_iso"],
            "market_end_date_iso": market["market_end_date_iso"]
        }
        market_details_list.append(details)
    
    return market_details_list

def create_json_file(filename, market_info, period_start, period_end, crypto_symbol="BTCUSDT"):
    """Create a new JSON file with metadata header only if it doesn't exist or period/market changes"""
    logger.info(f"\nChecking file: {filename}")
    
    new_file_needed = False
    if os.path.exists(filename):
        logger.info("File exists, checking if it's for the same market period...")
        try:
            with open(filename, 'r') as f:
                existing_data = json.load(f)
                existing_market_slug = existing_data["collection_metadata"]["market"]
                existing_start_ts = existing_data["collection_metadata"]["collection_period"]["start"]
                existing_end_ts = existing_data["collection_metadata"]["collection_period"]["end"]
                
                new_start_ts = int(period_start.timestamp())
                new_end_ts = int(period_end.timestamp())

                logger.info(f"Existing market slug: {existing_market_slug}")
                logger.info(f"Existing period: {datetime.fromtimestamp(existing_start_ts)} to {datetime.fromtimestamp(existing_end_ts)}")
                logger.info(f"New proposed market slug: {market_info['slug']}")
                logger.info(f"New proposed period: {period_start} to {period_end}")

                if (existing_market_slug == market_info["slug"] and
                    existing_start_ts == new_start_ts and
                    existing_end_ts == new_end_ts):
                    logger.info(f"✓ Same market slug and period - continuing with existing file")
                    if "data_points" in existing_data:
                         logger.info(f"Current data points: {len(existing_data['data_points'])}")
                    else:
                         logger.info(f"Current data points: 0 (no 'data_points' key found)")
                    return # No new file needed
                else:
                    logger.info("× Different market slug or period - will create new file")
                    new_file_needed = True
            
        except (json.JSONDecodeError, KeyError) as e: # Added KeyError for missing keys
            logger.info(f"× File exists but is corrupted or has unexpected structure ({e}) - will create new file")
            new_file_needed = True
    else:
        logger.info("File doesn't exist - creating new file")
        new_file_needed = True
    
    if new_file_needed:
        # Fetch starting price when creating a new file - with robust error handling
        starting_price = get_crypto_price(symbol=crypto_symbol)
        if starting_price is None:
            logger.warning(f"Warning: Could not fetch starting price for {crypto_symbol}. 'starting_price' will be null.")
        
        metadata = {
            "collection_metadata": {
                "market": market_info["slug"],
                "market_id": market_info["market_id"],
                "starting_price": starting_price, # Added starting price
                "collection_period": {
                    "start": int(period_start.timestamp()),
                    "end": int(period_end.timestamp()),
                    "timezone": "America/New_York"
                },
                "tokens": {
                    "up": market_info["up_token"],
                    "down": market_info["down_token"]
                },
                "crypto_symbol_for_price": crypto_symbol # Store which symbol was used
            },
            "data_points": []
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"✓ Created/Replaced collection file: {filename}")
            logger.info(f"Collection period: {period_start.strftime('%Y-%m-%d %H:%M')} ET to {period_end.strftime('%Y-%m-%d %H:%M')} ET")
            if starting_price is not None:
                logger.info(f"Starting price for {crypto_symbol}: {starting_price}")
        except IOError as e:
            logger.error(f"Error writing to file {filename}: {e}")

def append_to_json_file(filename, data_point_to_add, market_info_for_new_file, period_start_for_new_file, period_end_for_new_file, crypto_symbol_for_new_file):
    """Append data point to the data_points array. Creates file if it doesn't exist or is malformed."""
    try:
        # Ensure file exists and has correct structure before appending
        if not os.path.exists(filename):
             logger.info(f"File {filename} not found by append_to_json_file. Creating it.")
             create_json_file(filename, market_info_for_new_file, period_start_for_new_file, period_end_for_new_file, crypto_symbol_for_new_file)

        with open(filename, 'r+') as f:
            try:
                content = json.load(f)
                if "data_points" not in content or not isinstance(content["data_points"], list):
                    logger.info(f"File {filename} is malformed (missing 'data_points' list). Re-creating.")
                    # Need to pass all necessary args to create_json_file
                    raise json.JSONDecodeError("Malformed JSON", "", 0) # Trigger recreation

            except json.JSONDecodeError:
                 # File is empty or corrupted, re-initialize it
                f.close() # Close before create_json_file tries to write
                logger.info(f"File {filename} is empty or corrupted. Re-creating it.")
                create_json_file(filename, market_info_for_new_file, period_start_for_new_file, period_end_for_new_file, crypto_symbol_for_new_file)
                # Re-open and load after creation
                with open(filename, 'r+') as f_new:
                    content = json.load(f_new)
            
            # Append to data_points array
            content["data_points"].append(data_point_to_add)
            
            # Write updated content
            f.seek(0)
            f.truncate() # Clear the file before writing new content
            json.dump(content, f, indent=2)

    except Exception as e:
        logger.error(f"Error appending to file {filename}: {e}")

def collect_market_data(helper, market_info, current_price):
    """Collect market data using existing helper and market info with timeout protection"""
    # Get timestamp in ET
    et_tz = pytz.timezone('America/New_York')
    timestamp = int(datetime.now(et_tz).timestamp())
    
    # Fetch BTCUSDC price with robust error handling
    btcusdc_price = helper.get_current_price("BTCUSDC")
    
    # Remove the redundant price fetch since we now pass it in
    orderbook_data = {"bids": [], "asks": []} # Default empty orderbook
    
    # Ensure market_info is not None and contains 'up_token'
    if market_info and market_info.get("up_token"): 
        try:
            # Use helper method with timeout protection
            orderbook_data = helper.get_order_book(market_info["up_token"])
        except Exception as e:
            logger.warning(f"Failed to get order book: {e}")
            # Continue with empty orderbook rather than failing
    
    return {
        "timestamp": timestamp,
        "price": current_price,
        "btcusdc_price": btcusdc_price,  # Add BTCUSDC price to the data point
        "orderbook": orderbook_data
    }

def _run_git_command_and_log(command_list, success_message="Command successful", failure_message="Command failed"):
    """Runs a git command with the correct branch name from .env and timeout protection"""
    # If command is checkout or push, add branch name
    if 'checkout' in command_list:
        command_list.append(BRANCH_NAME)
    elif 'push' in command_list:
        command_list.extend(['origin', BRANCH_NAME])
        
    logger.info(f"Running git command: {' '.join(command_list)}")
    try:
        process = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            check=False,
            timeout=60  # 60 second timeout for git operations
        )
        if process.stdout:
            logger.info(f"Stdout:\n{process.stdout.strip()}")
        if process.stderr:
            logger.info(f"Stderr:\n{process.stderr.strip()}")
        
        if process.returncode == 0:
            logger.info(success_message)
            return True
        else:
            if "commit" in command_list and ("nothing to commit" in process.stdout.lower() or "nothing to commit" in process.stderr.lower()):
                logger.info("Nothing to commit, working tree clean.")
                return "no_changes" 
            logger.error(f"{failure_message}. Return code: {process.returncode}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Git command timed out after 60 seconds: {' '.join(command_list)}")
        return False
    except Exception as e:
        logger.error(f"Exception running git command: {e}")
        return False

def perform_git_operations(commit_message):
    """Adds all .json files, commits, and pushes to the correct branch"""
    logger.info(f"\nAttempting Git operations with commit message: '{commit_message}'")
    
    # Ensure we're on the correct branch
    _run_git_command_and_log(
        ["git", "checkout"],
        success_message=f"Switched to branch {BRANCH_NAME}",
        failure_message=f"Failed to switch to branch {BRANCH_NAME}"
    )

    # Add all .json files
    add_result = _run_git_command_and_log(
        ["git", "add", "*.json"],
        success_message="Added all .json files",
        failure_message="Failed to add .json files"
    )

    # Commit
    commit_result = _run_git_command_and_log(
        ["git", "commit", "-m", commit_message],
        success_message="Commit successful",
        failure_message="Commit failed"
    )

    if commit_result == "no_changes":
        logger.info("No changes to commit")
        return 
    if not commit_result:
        logger.error("Aborting git push due to commit error")
        return

    # Push
    push_result = _run_git_command_and_log(
        ["git", "push"],
        success_message="Git push successful",
        failure_message="Git push failed"
    )

def get_current_json_points(filename):
    """Get the number of points in a JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = json.load(f)
                return len(content.get("data_points", []))
    except Exception as e:
        logger.error(f"Error reading points from {filename}: {e}")
    return 0

def should_send_email():
    """
    Check if we should send the email based on time and tracking file
    Returns: bool
    """
    # Get current time in Belgian timezone
    belgian_tz = pytz.timezone('Europe/Brussels')
    now = datetime.now(belgian_tz)
    
    # Check if it's 7:45
    target_hour = 8
    target_minute = 45
    
    # Create or load email tracking file
    tracking_file = 'email_tracking.json'
    today_date = now.strftime('%Y-%m-%d')
    
    try:
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r') as f:
                tracking_data = json.load(f)
        else:
            tracking_data = {"last_email_date": ""}
            
        last_email_date = tracking_data.get("last_email_date", "")
        
        # If we already sent email today, don't send again
        if last_email_date == today_date:
            return False
            
        # Check if it's time to send (7:45 or later if we missed it)
        if (now.hour > target_hour or 
            (now.hour == target_hour and now.minute >= target_minute)):
            # Update tracking file
            tracking_data["last_email_date"] = today_date
            with open(tracking_file, 'w') as f:
                json.dump(tracking_data, f)
            return True
            
    except Exception as e:
        logger.error(f"Error checking email schedule: {e}")
    
    return False

def check_and_send_email():
    """
    Check if it's time to send email and trigger analysis if needed
    """
    if should_send_email():
        perform_git_operations("daily backup")
        try:
            # Import analyze_yesterday from graph.py
            from graph import analyze_yesterday
            logger.info("Triggering daily analysis and email report...")
            analyze_yesterday()
        except Exception as e:
            logger.error(f"Error sending daily email report: {e}")

def collect_data_continuously(cryptos_to_track, interval_seconds=59):
    """Main collection loop with improved error handling and detailed statistics"""
    ensure_data_directory()
    
    # Initialize helper with robust error handling - don't fail if it doesn't work
    shared_helper = None
    for attempt in range(3):
        try:
            shared_helper = PolymarketHelper()
            if shared_helper.client is not None:
                logger.info("Shared PolymarketHelper initialized successfully.")
                break
            else:
                logger.warning(f"Helper client initialization failed (attempt {attempt + 1}/3)")
                time.sleep(10)
        except Exception as e:
            logger.warning(f"Failed to initialize helper (attempt {attempt + 1}/3): {e}")
            time.sleep(10)
    
    if shared_helper is None or shared_helper.client is None:
        logger.error("Failed to initialize helper after all attempts. Continuing with degraded functionality.")
        # Create a dummy helper that will gracefully fail
        shared_helper = type('DummyHelper', (), {
            'get_current_price': lambda self, symbol: None,
            'get_order_book': lambda self, token_id: {"bids": [], "asks": []},
            'client': None
        })()
    
    crypto_states = {}
    initial_period_start, initial_period_end, _ = get_current_market_period()
    
    consecutive_failures = 0
    max_consecutive_failures = 10  # Increased threshold before helper restart
    total_cycles = 0
    last_successful_cycle = 0
    
    while True:
        cycle_start_time = time.time()
        cycle_success = False
        total_cycles += 1
        
        try:
            # Add email check at the start of each cycle (with error isolation)
            try:
                check_and_send_email()
            except Exception as e:
                logger.error(f"Error in email check (non-critical): {e}")
            
            # Enhanced statistics tracking
            cycle_stats = {
                "total_points": 0,
                "by_crypto": {},
                "errors": {
                    "price_fetch": 0,
                    "market_details": 0,
                    "orderbook": 0,
                    "file_operations": 0,
                    "other": 0
                },
                "active_markets": 0,
                "json_files": {}  # Track points in JSON files
            }
            
            for crypto_config in cryptos_to_track:
                try:
                    crypto_name = crypto_config["name"]
                    crypto_symbol = crypto_config["symbol"]
                    market_type = crypto_config.get("market_type", "daily_up_down")
                    
                    # Initialize stats for this crypto if not exists
                    if crypto_name not in cycle_stats["by_crypto"]:
                        cycle_stats["by_crypto"][crypto_name] = {
                            "daily": {"points": 0, "markets": 0, "errors": 0},
                            "monthly": {"points": 0, "markets": 0, "errors": 0}
                        }
                    
                    logger.info(f"\n--- Processing {crypto_name} ({market_type}) ---")
                    
                    # Get current price (with robust retry mechanism)
                    current_price = None
                    try:
                        current_price = shared_helper.get_current_price(crypto_symbol)
                    except Exception as e:
                        logger.warning(f"Exception getting price for {crypto_symbol}: {e}")
                    
                    if current_price is None:
                        cycle_stats["errors"]["price_fetch"] += 1
                        logger.warning(f"Skipping {crypto_name} this cycle due to missing price data")
                        continue
                    
                    # Get market details with robust error handling
                    market_details_list = None
                    try:
                        market_details_list = initialize_market_details(
                            crypto_name=crypto_name,
                            market_type=market_type
                        )
                    except Exception as e:
                        logger.warning(f"Exception getting market details for {crypto_name}: {e}")
                        cycle_stats["errors"]["market_details"] += 1
                    
                    if not market_details_list:
                        logger.debug(f"No active markets found for {crypto_name} ({market_type})")
                        continue
                    
                    # Update active markets count
                    market_count = len(market_details_list)
                    cycle_stats["active_markets"] += market_count
                    stats_key = "daily" if market_type == "daily_up_down" else "monthly"
                    cycle_stats["by_crypto"][crypto_name][stats_key]["markets"] = market_count
                    
                    for market_details in market_details_list:
                        try:
                            market_slug = market_details["main_market_slug"]
                            
                            # Check if market period has changed
                            current_period_start, current_period_end, is_new_period_needed = get_current_market_period()
                            
                            if is_new_period_needed:
                                logger.info(f"New period detected for market: {market_slug}")
                                crypto_states[market_slug] = {
                                    "current_period_start": current_period_start,
                                    "current_period_end": current_period_end,
                                    "filename": get_output_filename(market_details)
                                }
                                
                                try:
                                    create_json_file(
                                        crypto_states[market_slug]["filename"],
                                        market_details,
                                        current_period_start,
                                        current_period_end,
                                        crypto_symbol
                                    )
                                except Exception as e:
                                    logger.error(f"Error creating JSON file: {e}")
                                    cycle_stats["errors"]["file_operations"] += 1
                                    continue
                            
                            # Collect market data with robust error handling
                            data_point = None
                            try:
                                data_point = collect_market_data(shared_helper, market_details, current_price)
                            except Exception as e:
                                logger.warning(f"Error collecting market data for {market_slug}: {e}")
                                cycle_stats["errors"]["orderbook"] += 1
                            
                            if data_point:
                                try:
                                    filename = crypto_states.get(market_slug, {}).get("filename") or get_output_filename(market_details)
                                    append_to_json_file(
                                        filename,
                                        data_point,
                                        market_details,
                                        current_period_start,
                                        current_period_end,
                                        crypto_symbol
                                    )
                                    cycle_stats["total_points"] += 1
                                    cycle_stats["by_crypto"][crypto_name][stats_key]["points"] += 1
                                    cycle_success = True
                                except Exception as e:
                                    logger.error(f"Error saving data point: {e}")
                                    cycle_stats["errors"]["file_operations"] += 1
                                
                        except Exception as e:
                            cycle_stats["by_crypto"][crypto_name][stats_key]["errors"] += 1
                            cycle_stats["errors"]["other"] += 1
                            logger.error(f"Error processing market {market_details.get('main_market_slug')}: {e}")
                            continue
                            
                except Exception as e:
                    cycle_stats["errors"]["other"] += 1
                    logger.error(f"Error processing cryptocurrency {crypto_config.get('name', 'unknown')}: {e}")
                    continue
            
            # Update failure tracking
            if cycle_success:
                consecutive_failures = 0
                last_successful_cycle = total_cycles
            else:
                consecutive_failures += 1
                logger.warning(f"Cycle failed, consecutive failures: {consecutive_failures}")
            
            # Restart helper if too many consecutive failures and we have a helper
            if consecutive_failures >= max_consecutive_failures and shared_helper.client is not None:
                logger.warning(f"Too many consecutive failures ({consecutive_failures}), attempting to reinitialize helper...")
                try:
                    new_helper = PolymarketHelper()
                    if new_helper.client is not None:
                        shared_helper = new_helper
                        consecutive_failures = 0
                        logger.info("Helper reinitialized successfully")
                    else:
                        logger.warning("Helper reinitialize returned None client")
                except Exception as e:
                    logger.error(f"Failed to reinitialize helper: {e}")
                    # Don't reset consecutive_failures, let it keep trying
            
            # File scanning for statistics (with error isolation)
            try:
                # Before printing summary, get points in current JSON files
                for crypto_name, crypto_stats in cycle_stats["by_crypto"].items():
                    cycle_stats["json_files"][crypto_name] = {
                        "daily": {"files": [], "total_points": 0},
                        "monthly": {"files": [], "total_points": 0}
                    }
                    
                    # Scan collected_data directory for relevant files
                    if os.path.exists("collected_data"):
                        for filename in os.listdir("collected_data"):
                            filepath = os.path.join("collected_data", filename)
                            
                            # Skip directories (monthly market subdirectories)
                            if os.path.isdir(filepath):
                                # Process files in monthly subdirectories
                                if crypto_name in filename:
                                    try:
                                        for subfile in os.listdir(filepath):
                                            sub_filepath = os.path.join(filepath, subfile)
                                            if sub_filepath.endswith('.json'):
                                                points = get_current_json_points(sub_filepath)
                                                cycle_stats["json_files"][crypto_name]["monthly"]["files"].append(sub_filepath)
                                                cycle_stats["json_files"][crypto_name]["monthly"]["total_points"] += points
                                    except Exception as e:
                                        logger.warning(f"Error scanning monthly directory {filepath}: {e}")
                                continue
                            
                            # Process daily market files
                            if filepath.endswith('.json') and crypto_name in filename:
                                if "up-or-down" in filename:  # Daily market file
                                    points = get_current_json_points(filepath)
                                    cycle_stats["json_files"][crypto_name]["daily"]["files"].append(filepath)
                                    cycle_stats["json_files"][crypto_name]["daily"]["total_points"] += points
            except Exception as e:
                logger.warning(f"Error during file scanning for statistics: {e}")
            
            # Calculate process duration
            process_duration = time.time() - cycle_start_time
            
            # Enhanced cycle summary logging
            logger.info("\n=== Cycle Summary ===")
            logger.info(f"Cycle: {total_cycles} | Last Success: {last_successful_cycle}")
            logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Process Duration: {process_duration:.2f} seconds")
            logger.info(f"Total Active Markets: {cycle_stats['active_markets']}")
            logger.info(f"Total Data Points Collected This Cycle: {cycle_stats['total_points']}")
            if process_duration > 0:
                logger.info(f"Collection Rate: {cycle_stats['total_points']/process_duration:.2f} points/second")
            
            logger.info("\nBy Cryptocurrency:")
            
            for crypto_name, crypto_stats in cycle_stats["by_crypto"].items():
                logger.info(f"\n{crypto_name.upper()}:")
                logger.info(f"  Daily Markets:")
                logger.info(f"    - Active Markets: {crypto_stats['daily']['markets']}")
                logger.info(f"    - Points Collected This Cycle: {crypto_stats['daily']['points']}")
                if crypto_name in cycle_stats["json_files"]:
                    logger.info(f"    - Total Points in Files: {cycle_stats['json_files'][crypto_name]['daily']['total_points']}")
                logger.info(f"    - Errors: {crypto_stats['daily']['errors']}")
                
                logger.info(f"  Monthly Markets:")
                logger.info(f"    - Active Markets: {crypto_stats['monthly']['markets']}")
                logger.info(f"    - Points Collected This Cycle: {crypto_stats['monthly']['points']}")
                if crypto_name in cycle_stats["json_files"]:
                    logger.info(f"    - Total Points in Files: {cycle_stats['json_files'][crypto_name]['monthly']['total_points']}")
                logger.info(f"    - Errors: {crypto_stats['monthly']['errors']}")
            
            logger.info("\nError Summary:")
            logger.info(f"  - Price Fetch Errors: {cycle_stats['errors']['price_fetch']}")
            logger.info(f"  - Market Details Errors: {cycle_stats['errors']['market_details']}")
            logger.info(f"  - Order Book Errors: {cycle_stats['errors']['orderbook']}")
            logger.info(f"  - File Operation Errors: {cycle_stats['errors']['file_operations']}")
            logger.info(f"  - Other Errors: {cycle_stats['errors']['other']}")
            
            # Calculate and log success rate
            total_attempts = cycle_stats['total_points'] + sum(cycle_stats['errors'].values())
            success_rate = (cycle_stats['total_points'] / total_attempts * 100) if total_attempts > 0 else 0
            logger.info(f"\nSuccess Rate: {success_rate:.1f}%")
            
            # Calculate and log time stats
            sleep_time = max(0, interval_seconds - process_duration)
            logger.info(f"\nTiming:")
            logger.info(f"  - Process Duration: {process_duration:.2f}s")
            logger.info(f"  - Sleep Time: {sleep_time:.2f}s")
            logger.info(f"  - Total Cycle Time: {(process_duration + sleep_time):.2f}s")
            logger.info(f"  - Consecutive Failures: {consecutive_failures}")
            logger.info(f"  - Helper Status: {'Active' if shared_helper.client is not None else 'Degraded'}")
            logger.info("=" * 50)
            
            # Sleep for the remaining time
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Critical error in main collection loop: {e}")
            consecutive_failures += 1
            
            # Calculate process duration for failed cycles too
            process_duration = time.time() - cycle_start_time
            sleep_time = max(0, interval_seconds - process_duration)
            
            logger.info(f"\nError Recovery:")
            logger.info(f"  - Process Duration: {process_duration:.2f}s")
            logger.info(f"  - Sleep Time: {sleep_time:.2f}s")
            logger.info(f"  - Consecutive Failures: {consecutive_failures}")
            logger.info("=" * 50)
            
            # Sleep even on error to maintain cycle timing
            time.sleep(sleep_time)

if __name__ == "__main__":
    # Example usage
    price = get_crypto_price("BTCUSDT")
    if price:
        logger.info(f"Current BTC price: ${price:,.2f}")

    cryptos_to_monitor = [
        {"name": "bitcoin", "symbol": "BTCUSDT", "market_type": "daily_up_down"},
        {"name": "bitcoin", "symbol": "BTCUSDT", "market_type": "monthly_hit"},
        {"name": "bitcoin", "symbol": "BTCUSDT", "market_type": "price_range_5pm"},
        {"name": "ethereum", "symbol": "ETHUSDT", "market_type": "daily_up_down"},
        {"name": "ethereum", "symbol": "ETHUSDT", "market_type": "monthly_hit"}
    ]
    
    collect_data_continuously(cryptos_to_track=cryptos_to_monitor, interval_seconds=59)
    

