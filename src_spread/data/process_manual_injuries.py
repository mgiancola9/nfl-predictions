#!/usr/bin/env python3
"""
Aggregates 32 manual injury CSVs into a single 2025 master file.
Handles PFR-specific team codes (e.g., 'oti' -> 'TEN').
"""

import pandas as pd
from pathlib import Path
import glob

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "manual_injuries"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "injuries_2025.csv"

# Map PFR codes (filenames) to Standard Codes (nflverse)
TEAM_MAP = {
    "crd": "ARI", "atl": "ATL", "rav": "BAL", "buf": "BUF", "car": "CAR",
    "chi": "CHI", "cin": "CIN", "cle": "CLE", "dal": "DAL", "den": "DEN",
    "det": "DET", "gnb": "GB",  "htx": "HOU", "clt": "IND", "jax": "JAX",
    "kan": "KC",  "rai": "LV",  "sdg": "LAC", "ram": "LA",  "mia": "MIA", 
    "min": "MIN", "nwe": "NE",  "nor": "NO",  "nyg": "NYG", "nyj": "NYJ",
    "phi": "PHI", "pit": "PIT", "sfo": "SF",  "sea": "SEA", "tam": "TB",
    "oti": "TEN", "was": "WAS"
}

def process_files():
    print("--- Processing Manual Injury Files ---")
    
    csv_files = list(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        print(f" No files found in {INPUT_DIR}")
        return

    all_data = []
    
    for file_path in csv_files:
        # 1. Identify Team from Filename
        pfr_code = file_path.stem.lower() # e.g., 'oti' from 'oti.csv'
        
        # Default to uppercase if not in map, but use map if available
        team_code = TEAM_MAP.get(pfr_code, pfr_code.upper())
        
        print(f"   Processing {pfr_code} -> {team_code}...", end=" ")
        
        try:
            # 2. Load Data
            # Try finding the header row automatically
            df = pd.read_csv(file_path)
            
            # If 'Player' isn't in columns, it might be on row 2 (common in PFR copy-paste)
            if 'Player' not in df.columns:
                df = pd.read_csv(file_path, header=1)
            
            if 'Player' not in df.columns:
                print(" Skipping (Could not find 'Player' column)")
                continue

            # 3. Identify Week Columns
            # Looks for columns starting with "Week" (case insensitive)
            week_cols = [c for c in df.columns if str(c).lower().startswith('week')]
            
            if not week_cols:
                print(" Skipping (No 'Week' columns found)")
                continue

            # 4. Melt to Long Format (Player | Week | Status)
            melted = df.melt(
                id_vars=["Player"], 
                value_vars=week_cols, 
                var_name="week_str", 
                value_name="report_status"
            )
            
            # 5. Clean Data
            # Drop rows where status is empty (Player wasn't injured that week)
            melted = melted.dropna(subset=["report_status"])
            
            # Extract Week Number
            # "Week 1" -> 1, "Week 10" -> 10
            # We use regex to pull the first number found in the header
            melted["week"] = melted["week_str"].str.extract(r'(\d+)').astype(int)
            
            # Add Metadata
            melted["team"] = team_code
            melted["season"] = 2025
            
            # Rename Player to standard
            melted = melted.rename(columns={"Player": "full_name"})
            
            all_data.append(melted)
            
        except Exception as e:
            print(f" Error: {e}")

    if all_data:
        # 6. Save Master File
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Normalize Statuses (Optional but good for consistency)
        # We leave 'Questionable' as is, so we can filter it later
        status_map = {
            "IR": "Injured Reserve",
            "D": "Doubtful",
            "O": "Out",
            "Q": "Questionable"
        }
        final_df["report_status"] = final_df["report_status"].replace(status_map)
        
        # Select final columns
        out_cols = ["season", "week", "team", "full_name", "report_status"]
        final_df[out_cols].to_csv(OUTPUT_FILE, index=False)
        
        print(f"\n SUCCESS: Combined {len(final_df)} injuries into {OUTPUT_FILE}")
    else:
        print("\n Failed to aggregate any data.")

if __name__ == "__main__":
    process_files()