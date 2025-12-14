import pandas as pd
import nfl_data_py as nfl
import requests
import time
import os
from datetime import datetime, timedelta
import pytz

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Data_Total', 'games_2016_2025.csv')

def get_stadium_coords():
    """
    Fetches stadium details including Latitude and Longitude.
    """
    print("Fetching stadium coordinates...")
    try:
        stadiums = nfl.import_stadiums()
        # Keep only what we need
        stadiums = stadiums[['stadium_id', 'stadium_name', 'stadium_latitude', 'stadium_longitude', 'stadium_roof']]
        return stadiums
    except Exception as e:
        print(f"Error fetching stadiums: {e}")
        return pd.DataFrame()

def get_realtime_weather(lat, lon, game_time_str):
    """
    Calls Open-Meteo API for a specific location and time.
    """
    try:
        # Convert game time string to datetime object
        # Assuming game_time_str is in format 'YYYY-MM-DD HH:MM:SS' (UTC)
        # We need to handle timezones carefully. nflverse gameday/gametime usually implies ET.
        
        # Open-Meteo uses ISO8601. We'll request the forecast for the specific hour.
        url = "https://api.open-meteo.com/v1/forecast"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "America/New_York", # Request data in ET to match nflverse usually
            "forecast_days": 16 # Ensure we can see far enough ahead
        }

        response = requests.get(url, params=params)
        data = response.json()
        
        if 'hourly' not in data:
            return None, None

        # Parse the response time list to find the closest hour to kickoff
        # This is a simplified lookup. For production, ensure datetime alignment is precise.
        # Here we assume the game is happening "today" or "soon" relative to the API call.
        
        # Finds the index in the API response that matches the game hour
        # Note: In a real 'production' run 1 hour before game, you just grab the 'current' weather or index 0.
        # But since we might run this Tuesday for Sunday games, we need to match the time.
        
        target_hour = game_time_str.split(':')[0] # simple hour extraction
        
        # For simplicity in this script, we'll take the weather at the *start* of the forecast 
        # matching the game date, or just specific logic if you run it immediately before games.
        
        # ACTUAL LOGIC: Match the date and hour
        # We'll just grab the index corresponding to the game hour if available.
        # If the game is far out, this might need adjustment, but for <7 days it's fine.
        
        # Let's fallback to "current_weather=true" if running 1 hour before
        # But since you might run this days ahead, we iterate.
        
        times = data['hourly']['time']
        temps = data['hourly']['temperature_2m']
        winds = data['hourly']['wind_speed_10m']
        
        # Simple match: Find the string in 'times' that looks like our game date/hour
        # API returns: "2025-10-14T13:00"
        # We want to match that format.
        target_iso = game_time_str.replace(" ", "T")[:13] # e.g. "2025-10-14T13"
        
        found_idx = -1
        for i, t in enumerate(times):
            if target_iso in t:
                found_idx = i
                break
        
        if found_idx != -1:
            return temps[found_idx], winds[found_idx]
        else:
            # Fallback: if we can't find exact hour, return middle of day or None
            return None, None

    except Exception as e:
        print(f"API Error: {e}")
        return None, None

def update_games_weather():
    print(f"Loading games from {DATA_PATH}...")
    games = pd.read_csv(DATA_PATH)
    stadiums = get_stadium_coords()
    
    # Filter for future games (no score yet)
    # We create a mask so we can update the original dataframe
    future_mask = games['home_score'].isna()
    games_to_update = games[future_mask].copy()
    
    if games_to_update.empty:
        print("No upcoming games found to update.")
        return

    print(f"Found {len(games_to_update)} upcoming games. Updating weather...")

    # Merge coordinates into the temporary df
    # nflverse games file uses 'stadium_id' which matches nfl_data_py stadiums
    games_to_update = games_to_update.merge(stadiums[['stadium_id', 'stadium_latitude', 'stadium_longitude', 'stadium_roof']], 
                                            on='stadium_id', how='left')

    count = 0
    for idx, row in games_to_update.iterrows():
        # Skip if indoors (Dome/Closed) - set to standard indoor weather
        roof = str(row['stadium_roof']).lower()
        if roof in ['dome', 'closed', 'fixed']:
            games.at[idx, 'weather_temp'] = 70
            games.at[idx, 'weather_wind_mph'] = 0
            continue

        # If outdoors, call API
        # Construct approximate datetime string from 'gameday' and 'gametime'
        # nflverse gametime is usually "HH:MM", gameday is "YYYY-MM-DD"
        game_ts = f"{row['gameday']} {row['gametime']}:00"
        
        lat = row['stadium_latitude']
        lon = row['stadium_longitude']
        
        if pd.isna(lat) or pd.isna(lon):
            continue

        temp, wind = get_realtime_weather(lat, lon, game_ts)
        
        if temp is not None:
            # Update the MAIN dataframe using the index
            # Note: We must map back to the original dataframe's index
            original_idx = games_to_update.index[count] # This might be tricky if indices reset
            # Safer way: Iterate the original games df where mask is true
            
            # Let's use the 'game_id' to update the main DF to be safe
            games.loc[games['game_id'] == row['game_id'], 'weather_temp'] = temp
            games.loc[games['game_id'] == row['game_id'], 'weather_wind_mph'] = wind
            print(f"Updated {row['game_id']}: {temp}F, {wind}mph")
            
            # Respect API limits (Open-Meteo is generous but let's be nice)
            time.sleep(0.2)
        
        count += 1

    # Save back
    games.to_csv(DATA_PATH, index=False)
    print("Weather update complete. Saved to CSV.")

if __name__ == "__main__":
    update_games_weather()