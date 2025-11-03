# Level 2 - Task 1

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('2) Stock Prices Data Set.csv')
print("Dataset loaded successfully.")   

print("First five rows of the dataset:")
print(df.head())

print(df.info(), "\n")

# Data Cleaning
#Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Drop missing rows if any
df.dropna(inplace=True)
print("Missing values after dropping rows:")
print(df.isnull().sum())    

# Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows before removal: {duplicates}")

# Remove duplicate rows
df.drop_duplicates(inplace=True)
print(f"Number of duplicate rows after removal: {df.duplicated().sum()}\n")


# Convert date column to datetime type
df['date'] = pd.to_datetime(df['date'])
print("Date column converted to datetime type.\n")

#  Display dataset info after cleaning
print("Dataset info after cleaning:")
print(df.info(), "\n")

# Save the cleaned dataset

df.to_csv("Stock_Price_Cleaned.csv", index=False)
print("✅ Cleaned dataset saved as 'Stock_Price_Cleaned.csv'\n")


# Regression Analysis

# Predict closing price based on other features
X = df[['open']] # Independent variable
y = df['close']  # Dependent variable


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dataset split into training and testing sets.\n")

print("Training sample:", len(X_train))
print("Testing sample:", len(X_test), "\n")

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully.\n")

#Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Model Evaluation Metrics:")
print(f"R-squared: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}\n")

# Display model parameters
print("Model Coefficients:")
print(f"Intercept (b0): {model.intercept_:.4f}")
print(f"Coefficient for 'open' (b1): {model.coef_[0]:.4f}\n")

# Visualization - Regression Line
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Prices')
plt.title('Actual vs Predicted Stock Closing Prices')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.legend()

plt.savefig('regression_plot.png')
print("Regression plot saved as 'regression_plot.png'.\n")

plt.show()

