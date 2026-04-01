import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Silver Price Forecast", layout="wide")
st.title("🪙 Silver Price — Prophet Forecast")
st.markdown(
    "**Training:** last 4.5 years · **Test:** last 6 months · "
    "**Forecast:** next 1 year · **Regressors:** Gold + USD Index · **Transform:** Log"
)

# ── 1. Download silver + gold + USD index ───────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(5 * 365.25))
    tickers = {"SI=F": "silver", "GC=F": "gold", "DX-Y.NYB": "usd"}
    frames = {}
    for ticker, name in tickers.items():
        raw = yf.download(ticker, start=start_date, end=end_date,
                          interval="1d", progress=False)
        if raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        s = raw[["Close"]].dropna().reset_index()
        s.columns = ["ds", name]
        s["ds"] = pd.to_datetime(s["ds"]).dt.tz_localize(None)
        frames[name] = s
    df = frames["silver"].merge(frames["gold"], on="ds").merge(frames["usd"], on="ds")
    return df.dropna().reset_index(drop=True)

with st.spinner("Downloading silver, gold & USD data..."):
    df = load_data()

if df is None or df.empty:
    st.error("No data returned from Yahoo Finance. Check your internet connection.")
    st.stop()

st.success(f"Loaded **{len(df)}** daily records: {df['ds'].iloc[0].date()} → {df['ds'].iloc[-1].date()}")

# Log-transform all series
df["y"]        = np.log(df["silver"])
df["gold_log"] = np.log(df["gold"])
df["usd_log"]  = np.log(df["usd"])

# ── 2. Train / Test split ───────────────────────────────────────────────
cutoff = df["ds"].iloc[-1] - timedelta(days=183)
train  = df[df["ds"] <= cutoff].copy()
test   = df[df["ds"] >  cutoff].copy()

col1, col2, col3 = st.columns(3)
col1.metric("Training days", len(train))
col2.metric("Test days",     len(test))
col3.metric("Total days",    len(df))

# ── 3. Fit Prophet with Gold + USD regressors ───────────────────────────
@st.cache_data(ttl=3600)
def fit_and_predict(_train, _test):
    train_p = _train[["ds", "y", "gold_log", "usd_log"]].copy()
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=15,
        seasonality_mode="additive",
        changepoint_range=0.95,
        n_changepoints=50,
    )
    model.add_regressor("gold_log", standardize=True)
    model.add_regressor("usd_log",  standardize=True)
    model.add_seasonality(name="monthly",   period=30.5,  fourier_order=8)
    model.add_seasonality(name="quarterly", period=91.25, fourier_order=8)
    model.add_seasonality(name="biannual",  period=182.5, fourier_order=5)
    model.fit(train_p)
    # Use actual gold/usd on test dates — valid backtest evaluation
    all_df = pd.concat([_train, _test])[["ds", "y", "gold_log", "usd_log"]].copy()
    return model.predict(all_df)

@st.cache_data(ttl=3600)
def forecast_future(_df):
    def forecast_regressor(df_in, col):
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.3,
            seasonality_mode="additive",
            changepoint_range=0.95,
        )
        m.fit(df_in[["ds", col]].rename(columns={col: "y"}))
        fut = m.make_future_dataframe(periods=365)
        fc  = m.predict(fut)
        return fc[["ds", "yhat"]].rename(columns={"yhat": col})

    gold_fc = forecast_regressor(_df, "gold_log")
    usd_fc  = forecast_regressor(_df, "usd_log")

    full_p = _df[["ds", "y", "gold_log", "usd_log"]].copy()
    model_full = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.5,
        seasonality_prior_scale=15,
        seasonality_mode="additive",
        changepoint_range=0.95,
        n_changepoints=50,
    )
    model_full.add_regressor("gold_log", standardize=True)
    model_full.add_regressor("usd_log",  standardize=True)
    model_full.add_seasonality(name="monthly",   period=30.5,  fourier_order=8)
    model_full.add_seasonality(name="quarterly", period=91.25, fourier_order=8)
    model_full.add_seasonality(name="biannual",  period=182.5, fourier_order=5)
    model_full.fit(full_p)

    future = model_full.make_future_dataframe(periods=365)
    future = future.merge(gold_fc, on="ds", how="left")
    future = future.merge(usd_fc,  on="ds", how="left")
    future["gold_log"] = future["gold_log"].ffill()
    future["usd_log"]  = future["usd_log"].ffill()
    return model_full.predict(future)

with st.spinner("Fitting Prophet model with Gold & USD regressors (~30 s)..."):
    pred_all = fit_and_predict(train, test)

# ── 4. Evaluate on test set ────────────────────────────────────────────
pred_test = pred_all[pred_all["ds"].isin(test["ds"])].copy()
pred_test = pred_test.sort_values("ds").reset_index(drop=True)
test_sorted = test.sort_values("ds").reset_index(drop=True)

pred_test_price = np.exp(pred_test["yhat"].values)
pred_test_lower = np.exp(pred_test["yhat_lower"].values)
pred_test_upper = np.exp(pred_test["yhat_upper"].values)
actual_test_price = test_sorted["silver"].values

rmse = np.sqrt(mean_squared_error(actual_test_price, pred_test_price))
mape = np.mean(np.abs((actual_test_price - pred_test_price) / actual_test_price)) * 100

st.markdown("---")
col1, col2 = st.columns(2)
col1.metric("Test RMSE", f"${rmse:.2f}")
col2.metric("Test MAPE", f"{mape:.2f}%")

# ── 5. Forecast next 1 year ────────────────────────────────────────────
with st.spinner("Forecasting next 1 year with regressor projections..."):
    forecast = forecast_future(df.copy())

forecast_price = np.exp(forecast["yhat"].values)
forecast_lower = np.exp(forecast["yhat_lower"].values)
forecast_upper = np.exp(forecast["yhat_upper"].values)

# ── 6. Interactive Plotly chart ─────────────────────────────────────────
st.markdown("---")
st.subheader("Silver Price: Train · Test · Forecast")

fig = go.Figure()

# Training data
fig.add_trace(go.Scatter(
    x=train["ds"], y=np.exp(train["y"]),
    mode="lines", name="Training Data (4.5 yrs)",
    line=dict(color="steelblue", width=1),
))

# Test actual
fig.add_trace(go.Scatter(
    x=test_sorted["ds"], y=actual_test_price,
    mode="lines", name="Test — Actual (6 mo)",
    line=dict(color="green", width=2),
))

# Test prediction
fig.add_trace(go.Scatter(
    x=pred_test["ds"], y=pred_test_price,
    mode="lines", name=f"Test — Predicted (RMSE ${rmse:.2f})",
    line=dict(color="red", width=2, dash="dash"),
))

# Future forecast
future_mask = forecast["ds"] > df["ds"].iloc[-1]
future_dates = forecast["ds"][future_mask]
future_prices = forecast_price[future_mask.values]
future_lo = forecast_lower[future_mask.values]
future_hi = forecast_upper[future_mask.values]

fig.add_trace(go.Scatter(
    x=future_dates, y=future_hi,
    mode="lines", line=dict(width=0), showlegend=False,
))
fig.add_trace(go.Scatter(
    x=future_dates, y=future_lo,
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(255,165,0,0.2)",
    name="Forecast Confidence",
))
fig.add_trace(go.Scatter(
    x=future_dates, y=future_prices,
    mode="lines", name="1-Year Forecast",
    line=dict(color="darkorange", width=2),
))

# Train/Test split line
fig.add_shape(type="line", x0=cutoff, x1=cutoff, y0=0, y1=1,
              yref="paper", line=dict(color="gray", dash="dot", width=1))
fig.add_annotation(x=cutoff, y=1, yref="paper",
                   text="Train / Test Split", showarrow=False,
                   yshift=10, font=dict(color="gray"))

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD / Troy Ounce)",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.15),
    height=600,
    margin=dict(l=60, r=30, t=40, b=80),
)

st.plotly_chart(fig, width='stretch')

# ── 7. Forecast table ──────────────────────────────────────────────────
st.subheader("Forecast Data (Next 12 Months)")
forecast_table = pd.DataFrame({
    "Date": future_dates.values,
    "Predicted Price ($)": np.round(future_prices, 2),
    "Lower Bound ($)": np.round(future_lo, 2),
    "Upper Bound ($)": np.round(future_hi, 2),
})
st.dataframe(forecast_table, width='stretch', hide_index=True)
