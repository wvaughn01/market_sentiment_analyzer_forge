import pandas as pd
import numpy as np

def compute_behavioral_features(df):
    """
    Adds behavioral economics-inspired features to the market DataFrame.
    These are numeric 'proxies' for human reactions like loss aversion or herding.
    Handles missing data and ensures enough history for rolling calculations.
    """
    # Ensure we have core price/volume columns
    required = ['Close', 'Volume']
    if not all(col in df.columns for col in required):
        raise ValueError(f"DataFrame missing required columns: {[col for col in required if col not in df.columns]}")
    
    # Calculate daily percent change in closing price
    df['price_change'] = df['Close'].pct_change().fillna(0)
    
    # Compute volatility with min_periods=1 to handle sparse data
    df['volatility'] = df['price_change'].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Compute loss aversion score (clipped to handle extreme values)
    mean_change = df['price_change'].abs().mean()
    if mean_change == 0:
        mean_change = 1e-5  # prevent division by zero
    df['loss_aversion_score'] = (abs(df['price_change']) / mean_change).clip(0, 10)
    
    # Reaction speed using volume (handle missing data)
    if 'Volume' in df.columns:
        vol_change = df['Volume'].pct_change().abs()
        df['reaction_speed'] = vol_change.rolling(window=3, min_periods=1).mean().fillna(0)
    else:
        df['reaction_speed'] = 0  # fallback if no volume data
    
    # Herding index (normalized to handle extreme values)
    df['herding_index'] = (df['price_change'] * df['reaction_speed']).clip(-5, 5)
    
    return df

def label_reaction(delta):
    """
    Categorize a reaction as positive, negative, or neutral based on sentiment change.
    """
    if delta > 0.05:
        return "Positive"
    elif delta < -0.05:
        return "Negative"
    else:
        return "Neutral"
