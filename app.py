import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# 🔄 ADDED
from sentiment_utils import FinancialNewsAggregator

# Load the trained model
model = joblib.load("behavioral_model.pkl")

# Reaction interpretation map
reaction_map = {
    0: "📉 Negative Reaction – Expect market downturn",
    1: "📊 Neutral Reaction – Little or no movement",
    2: "📈 Positive Reaction – Expect market uptrend"
}

# Page title
st.set_page_config(page_title="Market Reaction Predictor", page_icon="🧠", layout="centered")
st.title("🧠 Behavioral Economics Market Reaction Predictor")
st.markdown("Enter market features below to predict likely reaction to news or behavior.")

# === ✅ SINGLE PREDICTION SECTION ===
st.header("📈 Predict a Single Scenario")

# Input sliders
price_change = st.slider("Price Change (%)", -10.0, 10.0, 0.0, step=0.1)
volatility = st.slider("Volatility (std dev)", 0.0, 5.0, 1.0, step=0.1)
loss_aversion = st.slider("Loss Aversion Score", 0.0, 1.0, 0.5, step=0.01)
reaction_speed = st.slider("Reaction Speed", 0.0, 1.0, 0.5, step=0.01)
herding_index = st.slider("Herding Index", 0.0, 1.0, 0.5, step=0.01)

# === 🔄 NEW: Get sentiment from ticker input ===
st.markdown("#### 📰 News Sentiment Analysis from Ticker")
ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA, GOOGL):", value="AAPL")
use_scraped_sentiment = st.checkbox("✅ Use sentiment score from recent news in prediction")

# Default neutral sentiment
sentiment_score = 0.0
sentiment_interp = "😐 Neutral – No strong market sentiment"

if ticker:
    start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    aggregator = FinancialNewsAggregator(use_mock=False)
    df_news = aggregator.get_sentiment_data(ticker, start_date, end_date)

    if not df_news.empty:
        sentiment_score = aggregator.get_avg_sentiment(df_news)

        # Human-friendly interpretation
        def interpret_sentiment(score):
            if score > 0.5:
                return "😊 Strong Positive – Market optimism likely"
            elif score > 0.1:
                return "🙂 Mild Positive – Slightly favorable outlook"
            elif score < -0.5:
                return "😠 Strong Negative – Expect significant market concern"
            elif score < -0.1:
                return "😟 Mild Negative – Subtle worry or doubt"
            else:
                return "😐 Neutral – No clear directional bias"

        sentiment_interp = interpret_sentiment(sentiment_score)

        # Display scraped result
        st.markdown(f"**🧠 VADER Sentiment Score for {ticker.upper()}:** `{sentiment_score:.3f}`")
        st.markdown(f"**🔍 Interpretation:** {sentiment_interp}")
        st.caption("This sentiment score is calculated from the latest news articles about the stock.")

    else:
        st.warning("No recent news found or API unavailable. Sentiment score will remain neutral.")

# If user chooses to use the sentiment, include it in model input
model_sentiment = sentiment_score if use_scraped_sentiment else 0.0

# Prepare input data
new_data = pd.DataFrame([[price_change, volatility, loss_aversion, reaction_speed, herding_index, model_sentiment]],
                        columns=["price_change", "volatility", "loss_aversion_score", "reaction_speed", "herding_index", "sentiment_score"])

# Session log
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = pd.DataFrame(columns=[
        "Price Change", "Volatility", "Loss Aversion", "Reaction Speed", "Herding Index", "Sentiment",
        "Prediction", "Confidence"
    ])

# Predict
if st.button("🧠 Predict Market Reaction"):
    prediction = model.predict(new_data)[0]
    confidence = model.predict_proba(new_data).max() * 100

    st.markdown(f"### 🔍 Prediction: {reaction_map.get(prediction)}")
    st.markdown(f"**🔢 Model Confidence:** {confidence:.2f}%")

    # Log
    log_row = pd.DataFrame([{
        "Price Change": price_change,
        "Volatility": volatility,
        "Loss Aversion": loss_aversion,
        "Reaction Speed": reaction_speed,
        "Herding Index": herding_index,
        "Sentiment": sentiment_score,
        "Prediction": reaction_map.get(prediction),
        "Confidence": f"{confidence:.2f}%"
    }])
    st.session_state.prediction_log = pd.concat([st.session_state.prediction_log, log_row], ignore_index=True)

# Show logs
if not st.session_state.prediction_log.empty:
    st.markdown("## 📋 Prediction History")
    st.dataframe(st.session_state.prediction_log, use_container_width=True)
    csv = st.session_state.prediction_log.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Log as CSV", data=csv, file_name="prediction_log.csv", mime="text/csv")

# === ✅ CSV Upload for Batch Prediction ===
st.header("📁 Upload CSV for Batch Predictions")
uploaded_file = st.file_uploader("Upload CSV with appropriate columns", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        expected_cols = ["price_change", "volatility", "loss_aversion_score", "reaction_speed", "herding_index", "sentiment_score"]
        if not all(col in df.columns for col in expected_cols):
            st.error(f"CSV is missing required columns: {expected_cols}")
        else:
            preds = model.predict(df[expected_cols])
            probs = model.predict_proba(df[expected_cols])
            df["Prediction"] = [reaction_map.get(p) for p in preds]
            df["Confidence"] = [f"{max(p)*100:.2f}%" for p in probs]
            st.markdown("### 📊 Batch Predictions")
            st.dataframe(df, use_container_width=True)
            batch_csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions CSV", data=batch_csv, file_name="batch_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
