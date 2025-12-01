#!/usr/bin/env python3
"""
Calculate "Value Above Minimum" (VAM) lost due to injuries for each game.

Strategy:
1. Load LOCAL player stats from the PRIOR season (2020-2024).
2. Build Injury History:
   - 2021-2024: Fetch from nflverse API (Official).
   - 2025: Load from local 'injuries_2025.csv' (Manual Master).
3. Map injuries to Player Value (VAM):
   - 2021-2024: Merges on Player ID (gsis_id).
   - 2025: Merges on NORMALIZED Name (removes Jr, Sr, II, punctuation).
4. Sum the VAM of injured players to create 'vam_lost' features.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import re
from pathlib import Path

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

def normalize_name(name):
    """
    Normalizes player names for fuzzy matching.
    Ex: "Patrick Mahomes II" -> "patrick mahomes"
    """
    if not isinstance(name, str):
        return ""
    # Lowercase and strip whitespace
    name = name.lower().strip()
    # Remove punctuation (periods, apostrophes)
    name = re.sub(r"[.']", "", name)
    # Remove common suffixes (Jr, Sr, II, III, IV)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name)
    return name.strip()

def load_prior_year_vam(year):
    """
    Loads 'stats_player_reg_{year}.csv' to calculate VAM scores.
    """
    file_path = DATA_RAW_DIR / f"stats_player_reg_{year}.csv"
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path, low_memory=False)
    
    # Fill NAs
    cols = ['passing_epa', 'rushing_epa', 'receiving_epa', 'def_sacks', 'def_interceptions', 'def_fumbles_forced']
    for c in cols:
        if c not in df.columns: df[c] = 0
        df[c] = df[c].fillna(0)

    # Calculate VAM (Simple Composite)
    df['offense_vam'] = df['passing_epa'] + df['rushing_epa'] + df['receiving_epa']
    df['defense_vam'] = (df['def_sacks'] * 2.0) + (df['def_interceptions'] * 3.0) + (df['def_fumbles_forced'] * 2.0)
    df['player_vam'] = (df['offense_vam'] + df['defense_vam']).clip(lower=0)
    
    # Create Normalized Name for matching
    df['norm_name'] = df['player_display_name'].apply(normalize_name)
    
    return df[['season', 'player_id', 'player_display_name', 'norm_name', 'recent_team', 'player_vam']]

def get_all_priors(start, end):
    """Loads VAM scores for multiple years."""
    frames = []
    print(f"   Building Player Value map (Stats from {start}-{end})...")
    for y in range(start, end + 1):
        df = load_prior_year_vam(y)
        if not df.empty: frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def get_combined_injuries(start_year, end_year):
    """
    Combines API history (2021-2024) with Local Master File (2025).
    """
    all_frames = []
    
    # --- 1. API History (2021-2024) ---
    api_years = [y for y in range(start_year, end_year + 1) if y < 2025]
    if api_years:
        print(f"   Fetching official history ({min(api_years)}-{max(api_years)})...")
        try:
            inj = nfl.import_injuries(api_years)
            if 'gsis_id' in inj.columns:
                inj = inj.rename(columns={'gsis_id': 'player_id'})
            
            # Keep necessary columns
            if 'report_status' in inj.columns:
                inj = inj[['season', 'week', 'team', 'player_id', 'full_name', 'report_status']]
                all_frames.append(inj)
        except Exception as e:
            print(f"   ⚠️ API fetch failed: {e}")

    # --- 2. Local Master File (2025) ---
    if end_year >= 2025:
        # Check standard location first, then current directory
        local_path = DATA_RAW_DIR / "injuries_2025.csv"
        if not local_path.exists():
            local_path = Path("injuries_2025.csv") # Check root if just uploaded
            
        if local_path.exists():
            print("   ✅ Loading local 'injuries_2025.csv'...")
            inj_2025 = pd.read_csv(local_path)
            if 'player_id' not in inj_2025.columns:
                inj_2025['player_id'] = np.nan 
            all_frames.append(inj_2025)
        else:
            print("   ❌ Missing 'injuries_2025.csv'.")

    if not all_frames:
        return pd.DataFrame()
        
    full_df = pd.concat(all_frames, ignore_index=True)
    
    # --- 3. Filter for "True" Absences ---
    # EXCLUDE 'Questionable' (Active ~90% of time)
    # MATCH 'Out', 'Doubtful', 'IR', 'Pup', 'Suspended', 'NFI'
    mask = full_df['report_status'].astype(str).str.contains(
        '|'.join(['Out', 'Doubtful', 'Res', 'PUP', 'Sus', 'NFI']), 
        case=False, 
        na=False
    )
    
    filtered = full_df[mask].copy()
    
    # Add normalized name for 2025 matching
    filtered['norm_name'] = filtered['full_name'].apply(normalize_name)
    
    print(f"   Filtered to {len(filtered)} significant injuries (Out/Doubtful/IR).")
    return filtered

def aggregate_vam_lost(injuries, vam_scores):
    if injuries.empty or vam_scores.empty: return pd.DataFrame()
    
    print("   Mapping injuries to player values...")
    injuries['prior_season'] = injuries['season'] - 1
    
    # --- SPLIT LOGIC: Use IDs where possible, Names where not ---
    
    # 1. 2021-2024 (Has IDs)
    mask_has_id = injuries['player_id'].notna() & (injuries['season'] < 2025)
    inj_ids = injuries[mask_has_id].copy()
    
    merged_ids = inj_ids.merge(
        vam_scores,
        left_on=['prior_season', 'player_id'],
        right_on=['season', 'player_id'],
        how='left',
        suffixes=('', '_stats')
    )
    
    # 2. 2025 (No IDs - Use Normalized Names)
    mask_no_id = ~mask_has_id
    inj_names = injuries[mask_no_id].copy()
    
    # Aggregate VAM by (season, norm_name) to handle duplicates (take Max VAM)
    vam_by_name = vam_scores.groupby(['season', 'norm_name'])['player_vam'].max().reset_index()
    
    merged_names = inj_names.merge(
        vam_by_name,
        left_on=['prior_season', 'norm_name'],
        right_on=['season', 'norm_name'],
        how='left',
        suffixes=('', '_stats')
    )
    
    # --- COMBINE ---
    final_merged = pd.concat([merged_ids, merged_names], ignore_index=True)
    
    # Fill Missing VAM with 0 (Replacement Level)
    final_merged['player_vam'] = final_merged['player_vam'].fillna(0)
    
    print("   Aggregating VAM lost per game...")
    # Group by the original injury season/week/team
    result = final_merged.groupby(['season', 'week', 'team'])['player_vam'].sum().reset_index()
    return result.rename(columns={'player_vam': 'vam_lost'})

def main():
    print("--- Generating Injury Features (Final Robust) ---")
    
    # 1. Load VAM Stats (2020-2024)
    vam_scores = get_all_priors(2020, 2024)
    
    # 2. Get Injuries (Official + Manual 2025)
    injuries = get_combined_injuries(2021, 2025)
    
    # 3. Calculate Loss
    vam_lost_df = aggregate_vam_lost(injuries, vam_scores)
    
    # 4. Save
    output_path = DATA_INTERIM_DIR / "injury_features.csv"
    vam_lost_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved injury features to: {output_path}")
    print(f"   Rows: {len(vam_lost_df)}")
    
    # Validation Check
    if not vam_lost_df.empty:
        print("\n   Validation (Mean VAM Lost per Game):")
        print(vam_lost_df.groupby('season')['vam_lost'].mean())

if __name__ == "__main__":
    main()