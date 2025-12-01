# === main.py ===
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sentiment_utils import FinancialNewsAggregator
from features import compute_behavioral_features, label_reaction
from model_train import train_behavioral_model
import joblib

# Get S&P 500 tickers from Wikipedia (fallback if needed)
def get_sp500_tickers():
    try:
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        return df['Symbol'].tolist()
    except Exception as e:
        print("❌ Failed to get tickers:", e)
        return []

# Dynamic ticker set (top 50)
tickers = get_sp500_tickers()[:50]
news_aggregator = FinancialNewsAggregator()
records = []

# 30-day window per ticker
end = datetime.today()
start = end - timedelta(days=30)

for ticker in tickers:
    print(f"\n Processing: {ticker}")
    
    # Market data
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        print("No market data found.")
        continue

    df = compute_behavioral_features(df)
    price_change = df['price_change'].mean()

    # News + sentiment
    news_data = news_aggregator.get_sentiment_data(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if news_data.empty:
        print("No news data found.")
        continue

    pre_news = news_data[news_data['date'] < end - timedelta(days=15)]
    post_news = news_data[news_data['date'] >= end - timedelta(days=15)]
    pre_sent = news_aggregator.get_avg_sentiment(pre_news)
    post_sent = news_aggregator.get_avg_sentiment(post_news)
    sentiment_delta = post_sent - pre_sent
    reaction_label = label_reaction(sentiment_delta)

    # Record
    records.append({
        "ticker": ticker,
        "event_date": end.strftime("%Y-%m-%d"),
        "price_change": price_change,
        "volatility": df['volatility'].mean(),
        "loss_aversion_score": df['loss_aversion_score'].mean(),
        "reaction_speed": df['reaction_speed'].mean(),
        "herding_index": df['herding_index'].mean(),
        "sentiment_delta": sentiment_delta,
        "reaction_label": reaction_label
    })

# Final dataset
df_final = pd.DataFrame(records)
df_final.to_csv("live_training_data.csv", index=False)
print("\n✅ Saved data to live_training_data.csv")
print(df_final.head())

# Train model
model = train_behavioral_model(df_final)
joblib.dump(model, "trained_behavioral_model.pkl")
print("✅ Model saved as trained_behavioral_model.pkl")


