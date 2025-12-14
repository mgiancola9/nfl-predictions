import os
import pandas as pd
import nfl_data_py as nfl
from datetime import datetime

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
    This is superior to raw scraping because it handles column formatting
    and updates automatically.
    """
    print(f"Fetching Schedules for {SEASONS_TO_FETCH}...")
    
    # nfl_data_py handles the caching and downloading logic
    df_games = nfl.import_schedules(SEASONS_TO_FETCH)
    
    # Filter for the columns your pipeline expects
    # Note: nflverse uses 'home_team' and 'away_team', which matches your needs.
    cols = ['season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'game_id', 'gameday']
    df_games = df_games[cols]
    
    # Save to CSV
    output_path = os.path.join(DATA_DIR, "games_2016_2025.csv")
    df_games.to_csv(output_path, index=False)
    print(f"SUCCESS: Saved games file to {output_path}")

def fetch_team_stats():
    """
    Fetches weekly team stats directly from nflverse data releases.
    We do this manually because nfl_data_py lacks a specific 'import_team_stats' 
    function that returns the pre-aggregated weekly files you use.
    """
    print("Fetching Weekly Team Stats...")
    
    # The nflverse URL pattern for team stats
    url_template = "https://github.com/nflverse/nflverse-data/releases/download/team_stats/stats_team_week_{season}.csv"
    
    for season in SEASONS_TO_FETCH:
        try:
            # We only really need to re-download the current season every week,
            # but downloading all ensures history is consistent if nflverse updates past data.
            # To speed this up, you could add: if season < CURRENT_SEASON and os.path.exists(...): continue
            
            target_url = url_template.format(season=season)
            print(f"  - Downloading {season} stats from {target_url}...")
            
            df = pd.read_csv(target_url)
            
            # Save using your specific naming convention: 'stats_team_week_YYYY.csv'
            output_filename = f"stats_team_week_{season}.csv"
            output_path = os.path.join(DATA_DIR, output_filename)
            
            df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"FAILURE: Could not download stats for {season}. Error: {e}")

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