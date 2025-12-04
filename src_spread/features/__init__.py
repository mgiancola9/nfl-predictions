# src/features/__init__.py

# 1. Rolling Stats (No-Leak Team Strengths)
from .rolling_stats import get_weekly_stats

# 2. Injury Features (VAM Lost)
from .injury_features import (
    load_all_player_stats,
    calculate_player_vam,
    get_historical_injuries,
    aggregate_vam_lost
)

# 3. Context Building (Merging Stats into Games)
from .build_features import (
    load_rolling_stats,
    load_priors,
    join_stats,
    calculate_final_features
)

# 4. Market Features (Placeholder - Add this later)
# from .market_features import get_market_deltas

__all__ = [
    # Rolling Stats
    "get_weekly_stats",
    
    # Injury Features
    "load_all_player_stats",
    "calculate_player_vam",
    "get_historical_injuries",
    "aggregate_vam_lost",
    
    # Context Building
    "load_rolling_stats",
    "load_priors",
    "join_stats",
    "calculate_final_features",
]