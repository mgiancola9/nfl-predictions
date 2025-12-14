import pandas as pd
import requests
import time
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Data_Total', 'games_2016_2025.csv')

# HARDCODED COORDINATES (Lat, Lon)
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
    'Tottenham Hotspur Stadium': (51.6042, -0.0662),
    'Wembley Stadium': (51.5560, -0.2795),
    'Allianz Arena': (48.2188, 11.6247),
    'Estadio Azteca': (19.3029, -99.1505),
    'Deutsche Bank Park': (50.0686, 8.6455),
    'Santiago Bernabeu': (40.4530, -3.6883),
    'Corinthians Arena': (-23.5453, -46.4742)
}

def get_coords(row):
    if str(row.get('location', '')).title() == 'Neutral':
        stadium_name = str(row.get('stadium', ''))
        for name, coords in NEUTRAL_STADIUMS.items():
            if name in stadium_name: return coords
        return None, None
    
    team = row['home_team']
    return TEAM_COORDS.get(team, (None, None))

def get_realtime_weather(lat, lon, game_time_str):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "timezone": "America/New_York", "forecast_days": 16
        }
        response = requests.get(url, params=params)
        data = response.json()
        if 'hourly' not in data: return None, None

        target_iso = game_time_str.replace(" ", "T")[:13]
        times = data['hourly']['time']
        for i, t in enumerate(times):
            if target_iso in t:
                return data['hourly']['temperature_2m'][i], data['hourly']['wind_speed_10m'][i]
        return None, None
    except Exception:
        return None, None

def update_games_weather():
    if not os.path.exists(DATA_PATH):
        print("Run pull_data.py first!")
        return

    print(f"Loading games from {DATA_PATH}...")
    games = pd.read_csv(DATA_PATH)

    # ---------------------------------------------------------
    # PART 1: GLOBAL DOME STANDARDIZATION (Past & Future)
    # ---------------------------------------------------------
    # Apply to ALL rows where roof is dome/closed/fixed
    # This overwrites any existing data with the standard 70F/0mph
    dome_mask = games['roof'].astype(str).str.lower().isin(['dome', 'closed', 'fixed'])
    
    games.loc[dome_mask, 'weather_temp'] = 70
    games.loc[dome_mask, 'weather_wind_mph'] = 0
    
    print(f"Standardized {dome_mask.sum()} dome/closed games (past & future) to 70F/0mph.")

    # ---------------------------------------------------------
    # PART 2: UPDATE FUTURE OUTDOOR GAMES
    # ---------------------------------------------------------
    future_mask = games['home_score'].isna() & ~dome_mask
    games_to_update = games[future_mask].copy()
    
    if games_to_update.empty:
        print("No upcoming outdoor games found to update.")
    else:
        print(f"Updating weather for {len(games_to_update)} upcoming outdoor games...")

        for idx, row in games_to_update.iterrows():
            lat, lon = get_coords(row)
            if lat is None: continue

            gametime = str(row.get('gametime', ''))
            if pd.isna(gametime) or gametime == 'nan': continue
            
            game_ts = f"{row['gameday']} {gametime}:00"
            temp, wind = get_realtime_weather(lat, lon, game_ts)
            
            if temp is not None:
                games.loc[games['game_id'] == row['game_id'], 'weather_temp'] = temp
                games.loc[games['game_id'] == row['game_id'], 'weather_wind_mph'] = wind
                print(f"Updated {row['home_team']} vs {row['away_team']}: {temp}F")
                time.sleep(0.2)

    games.to_csv(DATA_PATH, index=False)
    print("Weather update complete. Saved to CSV.")

if __name__ == "__main__":
    update_games_weather()