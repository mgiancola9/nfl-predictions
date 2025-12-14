import pandas as pd
import requests
import time
import os
from io import StringIO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Data_Total', 'games_2016_2025.csv')

# ------------------------------------------------------------------------------
# COORDINATES & MAPPINGS
# ------------------------------------------------------------------------------
TEAM_COORDS = {
    'ARI': (33.5276, -112.2626), 'ATL': (33.7554, -84.4008), 'BAL': (39.2780, -76.6227),
    'BUF': (42.7737, -78.7870),  'CAR': (35.2258, -80.8528), 'CHI': (41.8623, -87.6167),
    'CIN': (39.0955, -84.5161),  'CLE': (41.5061, -81.6995), 'DAL': (32.7473, -97.0945),
    'DEN': (39.7439, -105.0201), 'DET': (42.3400, -83.0456), 'GB':  (44.5013, -88.0622),
    'HOU': (29.6847, -95.4107),  'IND': (39.7601, -86.1639), 'JAX': (30.3240, -81.6375),
    'KC':  (39.0489, -94.4839),  'LAC': (33.9534, -118.3390), 'LA':  (33.9534, -118.3390),
    'LAR': (33.9534, -118.3390), 'LV':  (36.0909, -115.1833), 'MIA': (25.9580, -80.2389),
    'MIN': (44.9735, -93.2575),  'NE':  (42.0909, -71.2643), 'NO':  (29.9511, -90.0812),
    'NYG': (40.8135, -74.0745),  'NYJ': (40.8135, -74.0745), 'PHI': (39.9008, -75.1675),
    'PIT': (40.4468, -80.0158),  'SEA': (47.5952, -122.3316), 'SF':  (37.4032, -121.9698),
    'TB':  (27.9759, -82.5033),  'TEN': (36.1665, -86.7713), 'WAS': (38.9077, -76.8645)
}

NEUTRAL_STADIUMS = {
    'Tottenham': (51.6042, -0.0662), 'Wembley': (51.5560, -0.2795),
    'Allianz': (48.2188, 11.6247),   'Azteca': (19.3029, -99.1505),
    'Deutsche': (50.0686, 8.6455),   'Bernabeu': (40.4530, -3.6883),
    'Corinthians': (-23.5453, -46.4742)
}

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------
def get_coords(row):
    """Resolve coordinates using stadium name (neutral) or home team."""
    # Check Neutral
    if str(row.get('location', '')).title() == 'Neutral':
        stadium = str(row.get('stadium', ''))
        for key, coords in NEUTRAL_STADIUMS.items():
            if key in stadium: return coords
    # Check Home Team
    return TEAM_COORDS.get(row['home_team'], (None, None))

def fetch_weather_api(url, params, date_str, hour_str):
    """Generic handler for Open-Meteo API (Forecast & Historical)."""
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if 'hourly' not in data: return None, None

        # API returns "YYYY-MM-DDTHH:MM"
        target_iso = f"{date_str}T{hour_str}" # e.g. "2023-09-10T13"
        
        times = data['hourly']['time']
        for i, t in enumerate(times):
            if t.startswith(target_iso): # Match date and hour
                return data['hourly']['temperature_2m'][i], data['hourly']['wind_speed_10m'][i]
        return None, None
    except Exception as e:
        print(f"  API Error: {e}")
        return None, None

# ------------------------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------------------------
def update_games_weather():
    if not os.path.exists(DATA_PATH):
        print(f"CRITICAL: {DATA_PATH} not found. Run pull_data.py first.")
        return

    print(f"Loading games from {DATA_PATH}...")
    games = pd.read_csv(DATA_PATH)
    
    # 1. STANDARDIZE DOMES (Past & Future)
    # ---------------------------------------------------------
    # Fixes roof data issues and prevents API calls for indoor games
    dome_mask = games['roof'].astype(str).str.lower().isin(['dome', 'closed', 'fixed'])
    games.loc[dome_mask, 'weather_temp'] = 70
    games.loc[dome_mask, 'weather_wind_mph'] = 0
    print(f"  - Standardized {dome_mask.sum()} dome/closed games.")

    # 2. IDENTIFY TARGETS (Missing Weather & Outdoors)
    # ---------------------------------------------------------
    # Finds rows where temp is NaN AND roof is NOT closed
    target_mask = games['weather_temp'].isna() & ~dome_mask
    targets = games[target_mask].copy()
    
    if targets.empty:
        print("  - No missing weather data found.")
    else:
        print(f"  - Found {len(targets)} games with missing weather. Processing...")

        for idx, row in targets.iterrows():
            lat, lon = get_coords(row)
            if lat is None: continue

            # Handle time (Default to 13:00 if missing)
            gametime = str(row.get('gametime', ''))
            if pd.isna(gametime) or gametime == 'nan': 
                hour_str = "13"
                gametime_clean = "13:00"
            else:
                hour_str = gametime.split(':')[0].zfill(2)
                gametime_clean = gametime

            date_str = row['gameday']
            is_future = pd.isna(row['home_score'])
            
            # 3. ROUTE TO CORRECT API
            # ---------------------------------------------------------
            if is_future:
                # FORECAST API (Upcoming games)
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    "latitude": lat, "longitude": lon,
                    "hourly": "temperature_2m,wind_speed_10m",
                    "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                    "timezone": "America/New_York", "forecast_days": 16
                }
                api_type = "Forecast"
            else:
                # HISTORICAL API (Past games - The "Self-Healing" Logic)
                url = "https://archive-api.open-meteo.com/v1/archive"
                params = {
                    "latitude": lat, "longitude": lon,
                    "start_date": date_str, "end_date": date_str,
                    "hourly": "temperature_2m,wind_speed_10m",
                    "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                    "timezone": "America/New_York"
                }
                api_type = "History"

            # Fetch
            temp, wind = fetch_weather_api(url, params, date_str, hour_str)
            
            if temp is not None:
                games.loc[idx, 'weather_temp'] = temp
                games.loc[idx, 'weather_wind_mph'] = wind
                print(f"    [{api_type}] Fixed {row['game_id']}: {temp}F, {wind}mph")
                time.sleep(0.2) # Rate limit respect
            else:
                print(f"    [{api_type}] Failed/Skipped {row['game_id']}")

    # Save
    games.to_csv(DATA_PATH, index=False)
    print("Weather update complete. Saved to CSV.")

if __name__ == "__main__":
    update_games_weather()