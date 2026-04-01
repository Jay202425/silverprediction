import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Download 10 years of daily silver futures data
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 10)

print("Downloading silver price data from Yahoo Finance...")
silver = yf.download("SI=F", start=start_date, end=end_date, interval="1d")

if silver.empty:
    print("No data returned. Please check your internet connection.")
    exit(1)

print(f"Downloaded {len(silver)} daily records from {silver.index[0].date()} to {silver.index[-1].date()}")

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(silver.index, silver["Close"], linewidth=0.8, color="#6A0DAD")
ax.set_title("Silver Price — Last 10 Years (Daily Close)", fontsize=16, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD per Troy Ounce)")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()

plt.savefig("silver_price_chart.png", dpi=150)
print("Chart saved to silver_price_chart.png")
plt.show()
