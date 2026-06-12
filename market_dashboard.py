import smtplib
from email.message import EmailMessage
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AI Market Command Center", layout="wide")

risk_assets = [
    "AAPL", "MSFT", "NVDA", "META", "AMZN",
    "GOOGL", "AVGO", "TSM", "JPM", "XOM"
]

etfs = ["SPY", "QQQ", "DIA", "VTI", "VOO", "XLK", "XLF", "XLE", "XLV"]
defensive_assets = ["TLT", "IEF", "SHY", "GLD", "SLV", "UUP"]
commodities = ["USO", "UNG", "DBA"]

all_assets = risk_assets + etfs + defensive_assets + commodities
benchmark = "QQQ"

st.title("Phase 13: AI Market Command Center")

st.sidebar.header("Settings")
initial_capital = st.sidebar.number_input("Initial Capital", value=10000)
top_n_assets = st.sidebar.slider("Number of Assets to Hold", 2, 8, 4)
train_window = st.sidebar.slider("Training Window Days", 252, 1260, 756)
future_days = st.sidebar.slider("Prediction Horizon Days", 5, 63, 21)

@st.cache_data
def download_data(tickers):
    return yf.download(tickers, start="2015-01-01", auto_adjust=True)

data = download_data(all_assets)

close = data["Close"]
volume = data["Volume"]
returns = close.pct_change()

def send_email_alert(subject, body, to_email):
    sender_email = st.secrets["EMAIL_ADDRESS"]
    sender_password = st.secrets["EMAIL_PASSWORD"]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
        
def build_features(ticker):
    df = pd.DataFrame()
    df["Close"] = close[ticker]
    df["Volume"] = volume[ticker]
    df["return"] = df["Close"].pct_change()

    df["momentum_21"] = df["Close"] / df["Close"].shift(21) - 1
    df["momentum_63"] = df["Close"] / df["Close"].shift(63) - 1
    df["momentum_126"] = df["Close"] / df["Close"].shift(126) - 1

    df["volatility_21"] = df["return"].rolling(21).std()
    df["volatility_63"] = df["return"].rolling(63).std()

    df["ma_200"] = df["Close"].rolling(200).mean()
    df["above_200ma"] = (df["Close"] > df["ma_200"]).astype(int)

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["volume_trend"] = df["Volume"] / df["Volume"].rolling(21).mean() - 1
    df["future_return"] = df["Close"].shift(-future_days) / df["Close"] - 1

    return df.dropna()

features = [
    "return",
    "momentum_21",
    "momentum_63",
    "momentum_126",
    "volatility_21",
    "volatility_63",
    "above_200ma",
    "RSI",
    "volume_trend"
]

scores = []

for ticker in all_assets:
    try:
        df = build_features(ticker)

        if len(df) < train_window:
            continue

        train_df = df.iloc[-train_window:]

        X_train = train_df[features]
        y_train = train_df["future_return"]

        X_latest = df[features].iloc[[-1]]

        model = RandomForestRegressor(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=10,
            random_state=42
        )

        model.fit(X_train, y_train)
        expected_return = model.predict(X_latest)[0]

        latest = df.iloc[-1]

        volatility = latest["volatility_63"]
        if volatility <= 0:
            continue

        final_score = (
            expected_return
            + 0.4 * latest["momentum_63"]
            + 0.3 * latest["momentum_126"]
            + 0.05 * latest["above_200ma"]
        ) / volatility

        if latest["above_200ma"] == 1 and latest["RSI"] < 75:
            action = "BUY / HOLD"
        elif latest["RSI"] > 75:
            action = "WATCH - POSSIBLY OVERBOUGHT"
        else:
            action = "AVOID / SELL"

        scores.append({
            "Ticker": ticker,
            "Expected Return": expected_return,
            "Momentum 63D": latest["momentum_63"],
            "Momentum 126D": latest["momentum_126"],
            "Volatility 63D": volatility,
            "RSI": latest["RSI"],
            "Above 200MA": latest["above_200ma"],
            "Final Score": final_score,
            "Action": action
        })

    except Exception:
        pass

score_table = pd.DataFrame(scores).sort_values("Final Score", ascending=False)

selected = score_table[
    (score_table["Action"] == "BUY / HOLD")
].head(top_n_assets)

if len(selected) == 0:
    selected = score_table[
        score_table["Ticker"].isin(defensive_assets)
    ].head(top_n_assets)

inverse_vols = 1 / selected["Volatility 63D"]
weights = inverse_vols / inverse_vols.sum()

selected = selected.copy()
selected["Portfolio Weight"] = weights.values

benchmark_price = close[benchmark]
benchmark_ma200 = benchmark_price.rolling(200).mean()

market_above_200 = benchmark_price.iloc[-1] > benchmark_ma200.iloc[-1]
benchmark_return_21 = benchmark_price.iloc[-1] / benchmark_price.iloc[-21] - 1
benchmark_volatility = returns[benchmark].rolling(21).std().iloc[-1]

if market_above_200 and benchmark_return_21 > 0:
    market_weather = "Sunny / Risk-On"
    risk_level = "Low to Medium"
elif market_above_200 and benchmark_return_21 <= 0:
    market_weather = "Cloudy / Neutral"
    risk_level = "Medium"
else:
    market_weather = "Stormy / Defensive"
    risk_level = "High"

col1, col2, col3 = st.columns(3)

col1.metric("Market Weather", market_weather)
col2.metric("Risk Level", risk_level)
col3.metric("Benchmark", benchmark)

st.subheader("Latest AI Asset Ranking")
st.dataframe(score_table, use_container_width=True)

st.subheader("Recommended Portfolio")
st.dataframe(
    selected[[
        "Ticker",
        "Expected Return",
        "Final Score",
        "RSI",
        "Action",
        "Portfolio Weight"
    ]],
    use_container_width=True
)

st.subheader("AI Guidance")

if market_weather == "Sunny / Risk-On":
    st.success("Market conditions support holding stronger risk assets.")
elif market_weather == "Cloudy / Neutral":
    st.warning("Market is mixed. Hold strongest assets, avoid weak momentum assets.")
else:
    st.error("Market conditions are defensive. Reduce risk and favor defensive assets.")

st.write("Suggested actions:")
for _, row in selected.iterrows():
    st.write(
        f"- **{row['Ticker']}**: {row['Action']} | Weight: {round(row['Portfolio Weight'] * 100, 2)}%"
    )

st.subheader("Benchmark Price vs 200-Day Moving Average")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(benchmark_price.index, benchmark_price, label=benchmark)
ax.plot(benchmark_ma200.index, benchmark_ma200, label="200-Day MA")
ax.set_title(f"{benchmark} Trend Filter")
ax.legend()
st.pyplot(fig)

st.subheader("Top Asset Scores")

fig2, ax2 = plt.subplots(figsize=(12, 5))
top_scores = score_table.head(10)
ax2.bar(top_scores["Ticker"], top_scores["Final Score"])
ax2.set_title("Top 10 AI Scores")
st.pyplot(fig2)

st.subheader("Alerts")

alerts = []

if not market_above_200:
    alerts.append(f"{benchmark} is below its 200-day moving average.")

if benchmark_volatility > returns[benchmark].rolling(252).std().mean():
    alerts.append("Market volatility is elevated.")

overbought = score_table[score_table["RSI"] > 75]["Ticker"].tolist()
if overbought:
    alerts.append("Overbought assets: " + ", ".join(overbought))

weak_assets = score_table[score_table["Above 200MA"] == 0]["Ticker"].tolist()
if weak_assets:
    alerts.append("Weak trend assets below 200MA: " + ", ".join(weak_assets[:10]))

if alerts:
    for alert in alerts:
        st.warning(alert)
else:
    st.success("No major danger alerts detected.")

st.subheader("Update Frequency")

st.write("""
This dashboard updates when you refresh the page.  
For monthly portfolio decisions, check near the end of each month.  
For daily risk monitoring, refresh once per day after market close.
""")
st.subheader("Phase 14: Alert Center")

alert_level = "LOW"
major_alerts = []
watch_alerts = []

if market_weather == "Stormy / Defensive":
    alert_level = "HIGH"
    major_alerts.append("Market is in defensive/stormy condition.")

if benchmark_volatility > returns[benchmark].rolling(252).std().mean() * 1.5:
    alert_level = "HIGH"
    major_alerts.append("Benchmark volatility is unusually high.")

if len(overbought) >= 3:
    if alert_level != "HIGH":
        alert_level = "MEDIUM"
    watch_alerts.append("Several assets are overbought.")

if len(weak_assets) >= 5:
    if alert_level != "HIGH":
        alert_level = "MEDIUM"
    watch_alerts.append("Many assets are below their 200-day moving average.")

if len(selected) > 0:
    top_asset = selected.iloc[0]["Ticker"]
    watch_alerts.append(f"Current top-ranked asset: {top_asset}")

st.write("Current Alert Level:")

if alert_level == "HIGH":
    st.error("HIGH RISK ALERT")
elif alert_level == "MEDIUM":
    st.warning("MEDIUM RISK ALERT")
else:
    st.success("LOW RISK / NORMAL CONDITIONS")

if major_alerts:
    st.write("Major Alerts:")
    for alert in major_alerts:
        st.error(alert)

if watch_alerts:
    st.write("Watch Alerts:")
    for alert in watch_alerts:
        st.warning(alert)
st.subheader("Phase 17: Email Alert System")

user_email = st.text_input("Enter your email for alerts")

send_test_alert = st.button("Send Test Alert")

if send_test_alert and user_email:
    send_email_alert(
        subject="Pactolus Test Alert",
        body="This is a test alert from Pactolus. Your email alert system is working.",
        to_email=user_email
    )
    st.success("Test alert sent.")
