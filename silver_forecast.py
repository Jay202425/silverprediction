import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── 1. Download 5 years of daily silver data ────────────────────────────
end_date = datetime.now()
start_date = end_date - timedelta(days=int(5 * 365.25))

print("Downloading silver price data...")
raw = yf.download("SI=F", start=start_date, end=end_date, interval="1d")

if raw.empty:
    print("No data returned."); exit(1)

# Flatten MultiIndex columns if present
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.get_level_values(0)

df = raw[["Close"]].dropna().copy()
df = df.reset_index()
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

print(f"Total records: {len(df)}  |  {df['ds'].iloc[0].date()} → {df['ds'].iloc[-1].date()}")

# ── 2. Train / Test split ───────────────────────────────────────────────
cutoff = df["ds"].iloc[-1] - timedelta(days=183)  # ~6 months
train = df[df["ds"] <= cutoff].copy()
test  = df[df["ds"] >  cutoff].copy()
print(f"Training: {len(train)} days  |  Test: {len(test)} days")

# ── 3. Preprocessing – MinMaxScaler ─────────────────────────────────────
scaler = MinMaxScaler()
train["y_scaled"] = scaler.fit_transform(train[["y"]])
test["y_scaled"]  = scaler.transform(test[["y"]])

# Prepare Prophet frames (Prophet needs columns named ds, y)
train_prophet = train[["ds", "y_scaled"]].rename(columns={"y_scaled": "y"})
test_prophet  = test[["ds", "y_scaled"]].rename(columns={"y_scaled": "y"})

# ── 4. Fit Prophet with tuned hyperparameters for lower RMSE ────────────
print("Fitting Prophet model (tuned hyperparameters)...")
model = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.15,   # more flexible trend
    seasonality_prior_scale=10,
    seasonality_mode="multiplicative",
    changepoint_range=0.9,
)
model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
model.fit(train_prophet)

# ── 5. Predict on test period ───────────────────────────────────────────
test_future = model.make_future_dataframe(periods=0)
# We need dates covering the test period too
all_dates = pd.DataFrame({"ds": pd.concat([train["ds"], test["ds"]]).values})
pred_all = model.predict(all_dates)

# Extract test predictions
pred_test = pred_all[pred_all["ds"].isin(test["ds"])].copy()
pred_test = pred_test.sort_values("ds").reset_index(drop=True)
test_sorted = test.sort_values("ds").reset_index(drop=True)

# Inverse-transform predictions back to original price scale
pred_test_price = scaler.inverse_transform(pred_test[["yhat"]].values)
pred_test_lower = scaler.inverse_transform(pred_test[["yhat_lower"]].values)
pred_test_upper = scaler.inverse_transform(pred_test[["yhat_upper"]].values)

actual_test_price = test_sorted["y"].values

rmse = np.sqrt(mean_squared_error(actual_test_price, pred_test_price.flatten()))
print(f"\n*** Test RMSE: ${rmse:.2f} ***\n")

# ── 6. Forecast next 1 year ────────────────────────────────────────────
# Refit on ALL data for the final forecast
full_prophet = df[["ds", "y"]].copy()
full_prophet["y"] = scaler.fit_transform(full_prophet[["y"]])

model_full = Prophet(
    daily_seasonality=False,
    weekly_seasonality=True,
    yearly_seasonality=True,
    changepoint_prior_scale=0.15,
    seasonality_prior_scale=10,
    seasonality_mode="multiplicative",
    changepoint_range=0.9,
)
model_full.add_seasonality(name="monthly", period=30.5, fourier_order=5)
model_full.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
model_full.fit(full_prophet)

future = model_full.make_future_dataframe(periods=365)
forecast = model_full.predict(future)

# Inverse-transform everything back to USD
forecast_price = scaler.inverse_transform(forecast[["yhat"]].values).flatten()
forecast_lower = scaler.inverse_transform(forecast[["yhat_lower"]].values).flatten()
forecast_upper = scaler.inverse_transform(forecast[["yhat_upper"]].values).flatten()

# ── 7. Plot everything ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))

# Training data
ax.plot(train["ds"], train["y"], color="steelblue", linewidth=0.8,
        label="Training Data (4.5 yrs)")

# Test data (actual)
ax.plot(test_sorted["ds"], actual_test_price, color="green", linewidth=1.2,
        label="Test Data — Actual (6 mo)")

# Test predictions
ax.plot(pred_test["ds"], pred_test_price, color="red", linewidth=1.2,
        linestyle="--", label=f"Test Prediction (RMSE=${rmse:.2f})")

# Future forecast
future_mask = forecast["ds"] > df["ds"].iloc[-1]
ax.plot(forecast["ds"][future_mask], forecast_price[future_mask.values],
        color="darkorange", linewidth=1.5, label="1-Year Forecast")
ax.fill_between(forecast["ds"][future_mask],
                forecast_lower[future_mask.values],
                forecast_upper[future_mask.values],
                color="orange", alpha=0.2, label="Forecast Confidence Interval")

# Cutoff line
ax.axvline(x=cutoff, color="gray", linestyle=":", linewidth=1, label="Train/Test Split")

ax.set_title("Silver Price — Prophet Forecast (Scaled + Tuned)", fontsize=16, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD / Troy Ounce)")
ax.legend(loc="upper left", fontsize=9)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()

plt.savefig("silver_forecast.png", dpi=150)
print("Chart saved to silver_forecast.png")
plt.show()
