import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import json
import logging
import time

# Use the same logger as data_collector.py
logger = logging.getLogger('market_data')

__all__ = ['get_current_market_token']

# Add at module level
_market_cache = {}
_cache_duration = 300  # 5 minutes

def get_all_gamma_markets_cached(limit=500, offset=0):
    """Cached version of get_all_gamma_markets"""
    current_time = time.time()
    cache_key = f"{limit}_{offset}"
    
    if (cache_key in _market_cache and 
        current_time - _market_cache[cache_key]['timestamp'] < _cache_duration):
        return _market_cache[cache_key]['data']
    
    # Fresh fetch
    data = get_all_gamma_markets(limit, offset)
    _market_cache[cache_key] = {
        'data': data,
        'timestamp': current_time
    }
    return data

def get_all_gamma_markets(limit=500, offset=0):
    """
    Fetch markets from Polymarket Gamma API with pagination
    
    Args:
        limit (int): Number of markets to fetch per request
        offset (int): Starting offset for pagination
    
    Returns:
        list: List of all markets
    """
    base_url = "https://gamma-api.polymarket.com"
    endpoint = f"{base_url}/markets"
    
    all_markets = []
    
    try:
        while True:
            params = {
                "limit": limit,
                "offset": offset,
                "active": True,
                "archived": False,
                "closed": False,
                # "tag_id": 3 # Removed to allow fetching all active markets for broader slug matching
            }
            
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data: # Check if data is empty list
                break
                
            all_markets.extend(data)
    
            offset += limit
            if len(data) < limit:
                break
                
            # print(f"Fetched {len(all_markets)} markets so far...") # Can be noisy, commented out
            
        df = pd.DataFrame(all_markets)
        
            
        if 'endDate' in df.columns and 'startDate' in df.columns: # Ensure columns exist
            # Store original ISO format dates
            df['market_start_date_iso'] = df['startDate']
            df['market_end_date_iso'] = df['endDate']
            # Also keep a formatted one for display if ever needed by older parts (though not primary)
            df['end_date_formatted'] = pd.to_datetime(df['endDate']).dt.strftime('%Y-%m-%d %H:%M:%S')
        elif 'endDate' in df.columns: # Only endDate
             df['market_end_date_iso'] = df['endDate']
             df['end_date_formatted'] = pd.to_datetime(df['endDate']).dt.strftime('%Y-%m-%d %H:%M:%S')
             df['market_start_date_iso'] = None # Explicitly set to None
        elif 'startDate' in df.columns: # Only startDate
             df['market_start_date_iso'] = df['startDate']
             df['market_end_date_iso'] = None
        else: # Neither
            df['market_start_date_iso'] = None
            df['market_end_date_iso'] = None

        return df
        

        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def filter_markets_by_slug(df, slug_pattern):
    """
    Filter markets DataFrame to keep only those whose slug *exactly matches* the pattern (case-insensitive).
    
    Args:
        df (pandas.DataFrame): DataFrame containing all markets
        slug_pattern (str): Pattern to match in market slugs (case-insensitive)
    
    Returns:
        pandas.DataFrame: Filtered DataFrame
    """
    if df is None or df.empty:
        return None
    if not isinstance(slug_pattern, str):
        print(f"Warning: slug_pattern is not a string: {slug_pattern}")
        return pd.DataFrame()
    # Use str.fullmatch for exact match, or str.contains if partial is still sometimes needed
    # For "what-price-will-bitcoin-hit-in-may", an exact match to the general slug is often desired first.
    return df[df['slug'].str.fullmatch(slug_pattern, case=False, na=False)]

def display_market_info_list(market_info_list):
    """
    Display relevant market information from a list of market dicts,
    matching the new structure from get_current_market_token.
    """
    if not market_info_list:
        print("No market information to display.")
        return
        
    print("\nFound Market Events/Targets:")
    print("=" * 100) # Increased width for more fields
    for i, market_event in enumerate(market_info_list):
        print(f"--- Market {i+1} ---")
        print(f"  Specific Market ID:       {market_event.get('main_market_id')}")
        print(f"  Specific Market Slug:     {market_event.get('main_market_slug')} (for filename)")
        print(f"  Specific Market Question: {market_event.get('event_question')}")
        print(f"  General Event Slug:       {market_event.get('general_event_slug')} (for folder name / search pattern)")
        print(f"  Token YES (Outcome 1):    {market_event.get('token_yes')}")
        print(f"  Token NO (Outcome 2):     {market_event.get('token_no')}")
        print(f"  Market Start Date (ISO):  {market_event.get('market_start_date_iso')}")
        print(f"  Market End Date (ISO):    {market_event.get('market_end_date_iso')}")
        print("-" * 40)

def get_target_slug(crypto_name="bitcoin", market_type="daily_up_down"):
    """
    Generate the target slug pattern.
    
    Args:
        crypto_name (str): The name of the cryptocurrency (e.g., "bitcoin", "ethereum").
        market_type (str): Type of market slug: "daily_up_down", "monthly_hit", or "price_range_5pm".
    
    Returns:
        str: Slug pattern for the target market.
    """
    et_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(et_tz)
    
    crypto_name_formatted = crypto_name.lower()
    
    if market_type == "daily_up_down":
        # Market resolves at 12:00 ET (noon)
        resolution_time = now_et.replace(hour=12, minute=0, second=0, microsecond=0)
        # If current ET time is past noon, we look for tomorrow's market
        if now_et.hour >= 12:
            resolution_time += timedelta(days=1)
        date_part = resolution_time.strftime("%B-%-d").lower() # e.g., "may-15"
        return f"{crypto_name_formatted}-up-or-down-on-{date_part}"
    
    elif market_type == "monthly_hit":
        # For "what-price-will-X-hit-in-Month"
        month_name = now_et.strftime("%B").lower() # e.g., "may"
        return f"what-price-will-{crypto_name_formatted}-hit-in-{month_name}"
    
    elif market_type == "price_range_5pm":
        # Market resolves at 17:00 ET (5pm)
        resolution_time = now_et.replace(hour=17, minute=0, second=0, microsecond=0)
        # If current ET time is past 5pm, we look for tomorrow's market
        if now_et.hour >= 17:
            resolution_time += timedelta(days=1)
        date_part = resolution_time.strftime("%B-%-d-5pm-et").lower() # e.g., "june-3-5pm-et"
        return f"{crypto_name_formatted}-price-{date_part}"
    
    else:
        print(f"Warning: Unknown market_type '{market_type}'. Returning None.")
        return None

def get_current_market_token(crypto_name="bitcoin", market_type="daily_up_down"):
    """
    Gets information for active markets.
    For "daily_up_down", returns a list with a single dictionary.
    For "monthly_hit", returns a list of dictionaries, one for each specific price target market
    found whose parent event slug matches the general monthly pattern.
    For "price_range_5pm", returns a list of dictionaries for each price range market within the event.
    
    Output dictionary structure for each item in the list:
    {
        "main_market_id": str,        // Specific market's ID
        "main_market_slug": str,      // Specific market's slug (for filename)
        "event_question": str,        // Specific market's question (e.g., "Will BTC reach $X?")
        "general_event_slug": str,    // General event slug (for folder name, e.g., "bitcoin-price-june-3-5pm-et")
        "token_yes": str_token_id_1,  // "Yes" or "Up" token for the specific market
        "token_no": str_token_id_2,   // "No" or "Down" token
        "market_start_date_iso": str, // Specific market's original startDate (ISO 8601 format string from API)
        "market_end_date_iso": str    // Specific market's original endDate (ISO 8601 format string from API)
    }
    Returns an empty list if no relevant markets or events are found.
    """
    market_details_list = []
    try:
        all_markets_df = get_all_gamma_markets_cached()
        
        if all_markets_df is None or all_markets_df.empty:
            return market_details_list

        target_pattern_for_search = get_target_slug(crypto_name=crypto_name, market_type=market_type)
        logger.info(f"Searching for {market_type} market with pattern: {target_pattern_for_search}")
        if not target_pattern_for_search:
            return market_details_list
        
        markets_to_process_rows = []

        if market_type == "daily_up_down":
            # For daily, filter by the market's own slug directly matching target_pattern_for_search
            df_filtered = all_markets_df[all_markets_df['slug'].str.fullmatch(target_pattern_for_search, case=False, na=False)]
            if df_filtered is not None and not df_filtered.empty:
                markets_to_process_rows = [row for _, row in df_filtered.iterrows()]
            # Fallback to 'contains' if fullmatch yields nothing
            if not markets_to_process_rows: 
                 df_filtered_contains = all_markets_df[all_markets_df['slug'].str.contains(target_pattern_for_search, case=False, na=False)]
                 if df_filtered_contains is not None and not df_filtered_contains.empty:
                    markets_to_process_rows = [row for _, row in df_filtered_contains.iterrows()]

        elif market_type == "monthly_hit":
            # For monthly_hit, find markets if any of their event slugs match target_pattern_for_search
            unique_market_ids_found = set()
            for _, market_row_series in all_markets_df.iterrows():
                events_data_raw = market_row_series.get('events')
                events_list = []
                
                if isinstance(events_data_raw, str):
                    try: events_list = json.loads(events_data_raw)
                    except json.JSONDecodeError: pass 
                elif isinstance(events_data_raw, list):
                    events_list = events_data_raw
                
                if not events_list or not isinstance(events_list, list): continue

                for event in events_list:
                    if isinstance(event, dict):
                        event_slug_from_data = event.get('slug', '')
                        if event_slug_from_data.lower() == target_pattern_for_search.lower():
                            market_id = market_row_series.get('id')
                            if market_id not in unique_market_ids_found:
                                markets_to_process_rows.append(market_row_series)
                                unique_market_ids_found.add(market_id)
                            break

        elif market_type == "price_range_5pm":
            # For price_range_5pm, find markets if any of their event slugs match target_pattern_for_search
            # These are sub-markets within an event that has multiple price ranges
            unique_market_ids_found = set()
            for _, market_row_series in all_markets_df.iterrows():
                events_data_raw = market_row_series.get('events')
                events_list = []
                
                if isinstance(events_data_raw, str):
                    try: events_list = json.loads(events_data_raw)
                    except json.JSONDecodeError: pass 
                elif isinstance(events_data_raw, list):
                    events_list = events_data_raw
                
                if not events_list or not isinstance(events_list, list): continue

                for event in events_list:
                    if isinstance(event, dict):
                        event_slug_from_data = event.get('slug', '')
                        if event_slug_from_data.lower() == target_pattern_for_search.lower():
                            market_id = market_row_series.get('id')
                            if market_id not in unique_market_ids_found:
                                markets_to_process_rows.append(market_row_series)
                                unique_market_ids_found.add(market_id)
                            break
        
        if not markets_to_process_rows:
            return market_details_list
        
        for market_row in markets_to_process_rows:
            specific_market_id = market_row.get('id')
            specific_market_slug = market_row.get('slug') 
            specific_market_question = market_row.get('question', 'N/A') 
            market_start_date_iso_val = market_row.get('market_start_date_iso')
            market_end_date_iso_val = market_row.get('market_end_date_iso')
            
            token_ids_raw = market_row.get('clobTokenIds')
            token_yes, token_no = None, None

            if isinstance(token_ids_raw, list) and len(token_ids_raw) >= 1:
                token_yes = token_ids_raw[0]
                if len(token_ids_raw) >= 2:
                    token_no = token_ids_raw[1]
            elif isinstance(token_ids_raw, str):
                try:
                    parsed_ids = json.loads(token_ids_raw)
                    if isinstance(parsed_ids, list) and len(parsed_ids) >=1:
                        token_yes = parsed_ids[0]
                        if len(parsed_ids) >=2:
                            token_no = parsed_ids[1]
                except json.JSONDecodeError: pass 

            item = {
                "main_market_id": specific_market_id,
                "main_market_slug": specific_market_slug, 
                "event_question": specific_market_question,
                "token_yes": token_yes,
                "token_no": token_no,
                "market_start_date_iso": market_start_date_iso_val,
                "market_end_date_iso": market_end_date_iso_val
            }

            if market_type == "daily_up_down":
                item["general_event_slug"] = specific_market_slug 
                market_details_list.append(item)
            elif market_type == "monthly_hit":
                item["general_event_slug"] = target_pattern_for_search 
                market_details_list.append(item)
            elif market_type == "price_range_5pm":
                # For 5pm price range markets, use the target pattern as general event slug
                item["general_event_slug"] = target_pattern_for_search 
                market_details_list.append(item)
        
        return market_details_list
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR in get_current_market_token: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return market_details_list

def main():
    print("--- Testing Daily Up/Down Market (Bitcoin) ---")
    daily_markets = get_current_market_token(crypto_name="bitcoin", market_type="daily_up_down")
    if daily_markets:
        print(f"Daily market(s) found for Bitcoin. Count: {len(daily_markets)}")
        display_market_info_list(daily_markets)
    else:
        print("No active daily Bitcoin market found for the current/next period.")

    print("\n--- Testing Monthly Hit Market (Bitcoin) ---")
    monthly_btc_markets = get_current_market_token(crypto_name="bitcoin", market_type="monthly_hit")
    if monthly_btc_markets:
        print(f"Monthly 'hit' markets found for Bitcoin. Count: {len(monthly_btc_markets)}")
        display_market_info_list(monthly_btc_markets)
    else:
        print("No active monthly Bitcoin 'what price will hit' market found for the current month with the expected event slug pattern.")

    print("\n--- Testing Price Range 5PM Market (Bitcoin) ---")
    price_range_5pm_markets = get_current_market_token(crypto_name="bitcoin", market_type="price_range_5pm")
    if price_range_5pm_markets:
        print(f"Price range 5PM markets found for Bitcoin. Count: {len(price_range_5pm_markets)}")
        display_market_info_list(price_range_5pm_markets)
    else:
        print("No active Bitcoin price range 5PM market found for the current/next period.")

    print("\n--- Testing Monthly Hit Market (Ethereum) ---")
    monthly_eth_markets = get_current_market_token(crypto_name="ethereum", market_type="monthly_hit")
    if monthly_eth_markets:
        print(f"Monthly 'hit' markets found for Ethereum. Count: {len(monthly_eth_markets)}")
        display_market_info_list(monthly_eth_markets)
    else:
        print("No active monthly Ethereum 'what price will hit' market found for the current month with the expected event slug pattern.")

if __name__ == "__main__":
    main()