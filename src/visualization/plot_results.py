import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
INPUT_FILE = DATA_DIR / "predictions_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_predictions():
    if not INPUT_FILE.exists():
        raise FileNotFoundError("Run train_model.py first!")
    return pd.read_csv(INPUT_FILE)

def plot_confidence_tiers(df):
    """
    Figure 1: The "Filter" Proof.
    Shows that higher confidence = higher accuracy.
    """
    # Create bins for confidence
    conditions = [
        (df['edge'] < 0.025),               # Low (50-52.5%)
        (df['edge'] >= 0.025) & (df['edge'] < 0.05), # Med (52.5-55%)
        (df['edge'] >= 0.05)                # High (>55%)
    ]
    choices = ['Low (50-52.5%)', 'Med (52.5-55%)', 'High (>55%)']
    
    # FIX: Added default='Unknown' to prevent int/string type error
    df['confidence_tier'] = np.select(conditions, choices, default='Unknown')
    
    # Calculate stats
    stats = df.groupby('confidence_tier').apply(
        lambda x: pd.Series({
            'accuracy': (x['home_cover'] == x['pred_pick']).mean(),
            'count': len(x)
        })
    ).reset_index()
    
    # Filter out 'Unknown' if it exists (shouldn't if data is clean)
    stats = stats[stats['confidence_tier'] != 'Unknown']
    
    # Sort for plotting order
    sort_order = ['Low (50-52.5%)', 'Med (52.5-55%)', 'High (>55%)']
    stats['confidence_tier'] = pd.Categorical(stats['confidence_tier'], categories=sort_order, ordered=True)
    stats = stats.sort_values('confidence_tier')

    # Plot
    plt.figure(figsize=(10, 6))
    bars = sns.barplot(data=stats, x='confidence_tier', y='accuracy', palette='Blues')
    
    # Add Breakeven Line
    plt.axhline(0.5238, color='red', linestyle='--', linewidth=2, label='Breakeven (52.4%)')
    
    # Add labels
    for i, row in stats.iterrows():
        # Skip if NaN
        if pd.isna(row['accuracy']): continue
        plt.text(i, row.accuracy + 0.01, f"{row.accuracy:.1%}\n(n={int(row['count'])})", 
                 ha='center', fontsize=12, fontweight='bold')
    
    plt.ylim(0.40, 0.65)
    plt.title('Model Accuracy by Confidence Level', fontsize=14, fontweight='bold')
    plt.ylabel('Win Rate')
    plt.xlabel('Model Confidence Tier')
    plt.legend()
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig1_confidence_tiers.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_bankroll_simulation(df):
    """
    Figure 2: The "Profit" Proof.
    Shows the cumulative return of the High Confidence strategy vs. Blind Betting.
    """
    df = df.sort_values(['week', 'game_id'])
    
    # Logic: Bet 1.1u to win 1u (Standard -110 odds)
    # Win = +1.0, Loss = -1.1
    df['correct'] = (df['home_cover'] == df['pred_pick'])
    df['profit_unit'] = np.where(df['correct'], 1.0, -1.1)
    
    # Strategy 1: Bet Everything
    df['cum_profit_all'] = df['profit_unit'].cumsum()
    
    # Strategy 2: High Confidence Only (>55%)
    df['profit_high_conf'] = np.where(df['edge'] >= 0.05, df['profit_unit'], 0)
    df['cum_profit_high'] = df['profit_high_conf'].cumsum()
    
    # Group by week for cleaner lines
    weekly = df.groupby('week')[['cum_profit_all', 'cum_profit_high']].last().reset_index()
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(weekly['week'], weekly['cum_profit_all'], color='gray', linestyle=':', linewidth=2, label='Strategy: Bet All Games')
    plt.plot(weekly['week'], weekly['cum_profit_high'], color='green', linewidth=3, marker='o', label='Strategy: Bet >55% Conf.')
    
    plt.axhline(0, color='black', linewidth=1)
    plt.title('Profitability Simulation (2025 Season)', fontsize=14, fontweight='bold')
    plt.ylabel('Profit (Units)')
    plt.xlabel('Week')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig2_profitability.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_smoothed_accuracy(df):
    """
    Figure 3: The "Learning" Curve (Smoothed).
    Uses a 4-week rolling average to hide weekly noise.
    """
    weekly_acc = df.groupby('week').apply(
        lambda x: (x['home_cover'] == x['pred_pick']).mean()
    ).reset_index(name='accuracy')
    
    # 4-Week Rolling Average
    weekly_acc['rolling_acc'] = weekly_acc['accuracy'].rolling(window=4, min_periods=1).mean()
    
    plt.figure(figsize=(12, 6))
    
    # Scatter plot for raw weekly (faded)
    plt.scatter(weekly_acc['week'], weekly_acc['accuracy'], color='gray', alpha=0.4, label='Weekly Raw')
    
    # Line for trend
    plt.plot(weekly_acc['week'], weekly_acc['rolling_acc'], color='blue', linewidth=3, label='4-Week Trend')
    
    plt.axhline(0.5238, color='red', linestyle='--', label='Breakeven')
    
    plt.title('Model Consistency: 4-Week Rolling Accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xlabel('Week')
    plt.ylim(0.3, 0.8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "fig3_learning_trend.png"
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def main():
    try:
        df = load_predictions()
        plot_confidence_tiers(df)
        plot_bankroll_simulation(df)
        plot_smoothed_accuracy(df)
        print("\n New figures generated in reports/figures/")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()