import os
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime
import requests
from io import StringIO
# ==============================================================================
# CONFIGURATION
# ==============================================================================
SEASONS_TO_FETCH = list(range(2016, 2026)) # 2016 through 2025
CURRENT_SEASON = 2025

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data_Total')

def fetch_schedules():
    """
    Uses nfl_data_py to fetch schedule data.
    Pulls ALL columns to ensure we have stadium names and IDs.
    """
    print(f"Fetching Schedules for {SEASONS_TO_FETCH}...")
    
    df_games = nfl.import_schedules(SEASONS_TO_FETCH)
    
    # Rename weather columns to match your pipeline
    df_games = df_games.rename(columns={
        'temp': 'weather_temp', 
        'wind': 'weather_wind_mph'
    })

    # NO FILTERING: Keep all columns (including 'stadium', 'location', 'roof')
    # df_games = df_games[cols] 
    
    output_path = os.path.join(DATA_DIR, "games_2016_2025.csv")
    df_games.to_csv(output_path, index=False)
    print(f"SUCCESS: Saved games file to {output_path}")

def fetch_team_stats():
    """
    Manually fetches team stats from the 'stats_team' release tag.
    Bypasses nflreadpy errors and GitHub 404s.
    """
    print("Fetching Weekly Team Stats (Manual Mode)...")
    
    # CORRECTED URL PATTERN: Note the 'stats_team' tag
    url_template = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for season in SEASONS_TO_FETCH:
        target_url = url_template.format(season=season)
        print(f"  - Attempting download for {season}...")
        
        try:
            response = requests.get(target_url, headers=headers)
            
            if response.status_code == 200:
                # Load content into pandas to verify it's valid CSV
                df = pd.read_csv(StringIO(response.text))
                
                if not df.empty:
                    output_filename = f"stats_team_week_{season}.csv"
                    output_path = os.path.join(DATA_DIR, output_filename)
                    df.to_csv(output_path, index=False)
                    print(f"    SUCCESS: Saved {output_filename} ({len(df)} rows)")
                else:
                    print(f"    WARNING: File for {season} was empty.")
            else:
                # If 404, try the 'pbp' tag just in case they moved it there
                print(f"    FAILURE: Status {response.status_code} with tag 'stats_team'.")
                
        except Exception as e:
            print(f"    CRITICAL ERROR for {season}: {e}")

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    print("--- Starting nfl_data_py Pipeline ---")
    
    fetch_schedules()
    fetch_team_stats()
    
    print("--- Pipeline Complete ---")

if __name__ == "__main__":
    main()