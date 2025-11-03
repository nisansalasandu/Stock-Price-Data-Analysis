# Level 2 - Task 2

# import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the cleaned dataset
df = pd.read_csv('Stock_Price_Cleaned.csv')
print("Cleaned dataset loaded successfully.")

# Show first five rows of the dataset
print("First five rows of the cleaned dataset:")
print(df.head())

# Convert date column to datetime type
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df.sort_values('date', inplace=True)

# Set date as index
df.set_index('date', inplace=True)

# Time serues plot of closing prices
plt.figure(figsize=(12, 6))
plt.plot(df['close'], label='Close Price')
plt.title('Time Series of Close Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Save the plot
plt.savefig('time_series_close_prices.png')

plt.show()

# Check missing values of close prices
missing_values = df['close'].isnull().sum()
print(f"Missing values in 'close' column: {missing_values}")

# No any missing values found, proceed to decomposition

# Decompose the time series(close prices)
decomposition = seasonal_decompose(df['close'], model='additive', period=30)

# Plot the decomposed components
decomposition.plot()

# Save the decomposition plot
plt.savefig('time_series_decomposition.png')
print("Time series decomposition plot saved as 'time_series_decomposition.png'")

plt.show()

# Calculate moving average with a window size of 30 days
df['30day_moving_avg'] = df['close'].rolling(window=30).mean()

# Plot original close prices and moving average
plt.figure(figsize=(12, 6))
plt.plot(df['close'], label='Close Price', alpha=0.5)
plt.plot(df['30day_moving_avg'], label='30-Day Moving Average', color='orange')
plt.title("Moving Average Smoothing of Close Prices")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# Save the moving average plot
plt.savefig('moving_average_smoothing.png')
print("✅ Moving average smoothing plot saved as 'moving_average_smoothing.png'")   
plt.show()

# Display dataset info after time series processing
print("Dataset info after time series processing:")
print(df.info(), "\n")

# Save the processed dataset
df.to_csv("Stock_Price_Time_Series_Processed.csv")
print("✅ Processed dataset saved as 'Stock_Price_Time_Series_Processed.csv'\n")







