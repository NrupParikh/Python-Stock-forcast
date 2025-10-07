import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from flask import Flask, request, render_template_string

# Create a Flask application instance
app = Flask(__name__)


# =============== Data Fetching ===============
def fetch_stock_data(symbol, period="5y"):
    """Fetch historical stock data from Yahoo Finance"""
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    data = data.sort_index(ascending=True)
    return data


# =============== Technical Indicators ===============
def calculate_moving_averages(data):
    data["50MA"] = data["Close"].rolling(window=50).mean()
    data["200MA"] = data["Close"].rolling(window=200).mean()
    return data


def determin_trend(data):
    if (
        data["50MA"].iloc[-1] > data["200MA"].iloc[-1]
        and data["Close"].iloc[-1] > data["50MA"].iloc[-1]
    ):
        trend = "UPTREND (bullish for next 1-3 months)"
    elif (
        data["50MA"].iloc[-1] < data["200MA"].iloc[-1]
        and data["Close"].iloc[-1] < data["50MA"].iloc[-1]
    ):
        trend = "DOWNTREND (bearish for next 1-3 months)"
    else:
        trend = "SIDEWAYS (uncertain)"
    return trend


def calculate_RSI(data, window=14):
    delta = data["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data


def calculate_MACD(data, fast=12, slow=26, signal=9):
    data["EMA_fast"] = data["Close"].ewm(span=fast, adjust=False).mean()
    data["EMA_slow"] = data["Close"].ewm(span=slow, adjust=False).mean()
    data["MACD"] = data["EMA_fast"] - data["EMA_slow"]
    data["MACD_signal"] = data["MACD"].ewm(span=signal, adjust=False).mean()
    return data


# =============== Random Forest Forecast ===============
"""
This function uses a Random Forest ML model to learn from
historical stock indicators and predict stock prices for the next 30 days.
"""


def random_forest_forecast(data, days_ahead=30):
    """
    Predict future stock prices using Random Forest Regressor
    """
    df = data.copy()
    # next-day close as target
    df["Target"] = df["Close"].shift(-1)
    # Drop last row with NaN target
    df = df.dropna()
    # Features (you can add more indicators here)
    features = ["Close", "50MA", "200MA", "RSI", "MACD", "MACD_signal"]
    # drop rows with NaN from indicators
    df = df.dropna()
    # Feature matrix and target vector
    X = df[features]
    y = df["Target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Random Forest MAE: {mae:.2f}")

    # Forecast future price iteratively
    last_known = X.iloc[-1].values.reshape(1, -1)
    forecast_prices = []
    for _ in range(days_ahead):
        pred = model.predict(last_known)[0]
        forecast_prices.append(pred)
        # update only Close for simplicity
        last_known[0, 0] = pred

    return forecast_prices[-1], forecast_prices


# =============== Entry / Stoploss ===============
def calculate_entry_stoploss(data, trend, stoploss_percent=5):
    entry_price = None
    stop_loss = None
    if (
        trend.startswith("UPTREND")
        and data["RSI"].iloc[-1] < 70
        and data["MACD"].iloc[-1] > data["MACD_signal"].iloc[-1]
    ):
        entry_price = data["Close"].iloc[-1]
        stop_loss = entry_price * (1 - stoploss_percent / 100)
    return entry_price, stop_loss


# =============== Flask Routes ===============
@app.route("/", methods=["GET", "POST"])
def index():
    stocks = {
        "Reliance Industries": "RELIANCE.NS",
        "Infosys": "INFY.NS",
        "TCS": "TCS.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Ola Electric": "OLAELEC.NS",
    }

    result = None
    table_html = None

    if request.method == "POST":
        selected_symbol = request.form["symbol"]
        data = fetch_stock_data(selected_symbol)
        data = calculate_moving_averages(data)
        data = calculate_RSI(data)
        data = calculate_MACD(data)
        trend = determin_trend(data)
        # Random Forest Forecast
        predicted_price, _ = random_forest_forecast(data)
        # Entry & Stoploss
        entry_price, stop_loss = calculate_entry_stoploss(data, trend)
        current_price = data["Close"].iloc[-1]
        price_difference = predicted_price - current_price
        profit_or_loss = ((predicted_price - current_price) / current_price) * 100

        table_html = (
            data.tail(30)
            .reset_index()
            .to_html(classes="table table-striped table-bordered", index=False)
        )

        result = {
            "symbol": selected_symbol,
            "trend": trend,
            "current_price": f"{current_price:.2f}",
            "predicted_price": f"{predicted_price:.2f}",
            "price_difference": f"{price_difference:.2f}",
            "profit_or_loss": f"{profit_or_loss:.2f}%",
            "entry_price": f"{entry_price:.2f}" if entry_price else "No Entry Signal",
            "stop_loss": f"{stop_loss:.2f}" if stop_loss else "-",
            "date_now": data.index[-1].date(),
            "future_date": datetime.now().date() + timedelta(days=30),
        }

    return render_template_string(
        """
        <html>
            <head>
                <title>Stock Predictor</title>
                <link rel="stylesheet"
                      href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
                <style>
                    body { padding: 30px; }
                    th, td { text-align: center; }
                    .result-card { margin-top: 40px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h3 class="text-center mb-4">Stock Prediction Dashboard</h3>
                    <form method="POST" class="text-center mb-4">
                        <div class="row justify-content-center">
                            <div class="col-md-4">
                                <select name="symbol" class="form-select">
                                    {% for name, sym in stocks.items() %}
                                        <option value="{{ sym }}"
                                            {% if result and result.symbol == sym %}selected{% endif %}>
                                            {{ name }} ({{ sym }})
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>
                            <div class="col-md-2">
                                <button type="submit" class="btn btn-primary w-100">Predict</button>
                            </div>
                        </div>
                    </form>

                    {% if result %}
                    <div class="card result-card shadow">
                        <div class="card-body">
                            <h5 class="card-title text-center">Report for {{ result.symbol }}</h5>
                            <p><b>Trend:</b> {{ result.trend }}</p>
                            <p><b>Current Price:</b> ₹{{ result.current_price }} ({{ result.date_now }})</p>
                            <p><b>Predicted Price (Next 30 Days):</b> ₹{{ result.predicted_price }} ({{ result.future_date }})</p>
                            <p><b>Price Difference:</b> ₹{{ result.price_difference }}</p>
                            <p><b>Expected Return:</b> {{ result.profit_or_loss }}</p>
                            <p><b>Entry Price:</b> {{ result.entry_price }}</p>
                            <p><b>Stop Loss:</b> {{ result.stop_loss }}</p>
                        </div>
                    </div>

                    <div class="mt-4">
                        <h5>Last 30 Days Data</h5>
                        {{ table_html | safe }}
                    </div>
                    {% endif %}
                </div>
            </body>
        </html>
    """,
        stocks=stocks,
        result=result,
        table_html=table_html,
    )


# Run the app in debug mode
if __name__ == "__main__":
    # Run on local host
    # app.run(debug=True)

    # Run using public IP
    app.run(host="0.0.0.0", port=5000, debug=True)

    # Hugging Face uses port 7860 by default
    # app.run(host="0.0.0.0", port=7860)
