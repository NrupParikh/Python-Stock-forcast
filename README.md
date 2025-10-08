---
title: Stock Forecast App
emoji: 📈
colorFrom: green
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: Stock Prediction using Flask framework
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

---
title: Stock Forecast App
emoji: 📈
colorFrom: indigo
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: Stock Prediction using Flask framework
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🧠 Stock Price Predictor — Flask + Random Forest

A machine learning web app built using Flask and deployed on Hugging Face Spaces with Docker.  
It predicts future stock prices using historical Yahoo Finance data and Random Forest regression.

---

## 🚀 Tech Stack
- **Flask** — Web framework to build the backend logic and user interface.
- **Gunicorn** — WSGI production server for running Flask apps.
- **YFinance** — Fetches live and historical stock data.
- **Scikit-learn** — Implements the Random Forest prediction model.
- **Pandas & NumPy** — For data manipulation and mathematical operations.

---

## ⚙️ Features
✅ Fetch 5 years of stock data from Yahoo Finance  
✅ Compute indicators (RSI, MACD, Moving Averages)  
✅ Predict next 30-day prices using Random Forest  
✅ Detect trend (Bullish / Bearish / Sideways)  
✅ Suggest entry and stop-loss points  
✅ Display detailed report and 30-day recent table  

---

## 🐳 Deployment on Hugging Face
This Space uses **Docker runtime**.

### Files used:
- `Dockerfile` — Defines the container environment
- `requirements.txt` — Lists Python dependencies
- `app.py` — Contains Flask application logic

To deploy:
1. Create a new **Hugging Face Space**.
2. Choose **Docker** as the SDK.
3. Upload `app.py`, `requirements.txt`, and `Dockerfile`.
4. Click **Deploy Space** — your Flask app will run automatically on port **7860**.