# Stock Price Data Analysis

## Overview

This project analyzes **stock price data** using Python.
It includes **data cleaning**, **regression analysis**, **time series analysis**, and **K-Means clustering** to understand patterns and trends in stock prices.


## Details

**Prepared by:** Ruwan Pathiranage Sanduni Nisansala
**Project:** Data Analysis using Python


## Tools Used

* Python
* pandas
* scikit-learn
* statsmodels
* matplotlib
* seaborn
* VS Code


## Tasks

### Task 1 – Data Cleaning & Regression Analysis

* Cleaned missing and duplicate values.
* Built a **Linear Regression** model to predict closing prices based on opening prices.
* Model achieved **R² = 0.9997**, showing a very strong relationship.

### Task 2 – Time Series Analysis

* Analyzed stock price trends over time.
* Used **seasonal decomposition** and **moving averages** to identify patterns.
* Found no strong seasonality but clear long-term trends.

### Task 3 – Clustering Analysis (K-Means)

* Standardized data using **StandardScaler**.
* Determined optimal clusters using the **Elbow Method (K=3)**.
* Grouped stocks into 3 clusters based on price and volume similarities.


## Key Findings

* Strong positive correlation between opening and closing prices.
* Clear price trends identified using time series analysis.
* Three main stock behavior clusters found using K-Means.


## Visuals

* `regression_plot.png` – Regression Line (Actual vs Predicted)
* `time_series_decomposition.png` – Trend & Residuals
* `moving_average_smoothing.png` – 30-day Moving Average
* `elbow_method.png` – Finding Optimal K
* `kmeans_clusters.png` – Open vs Close Clusters
* `kmeans_high_low_clusters.png` – High vs Low Clusters
* `kmeans_volume_close_clusters.png` – Volume vs Close Clusters
* `kmeans_open_volume_clusters.png` – Open vs Volume Clusters


## Conclusion

This project demonstrates how to apply **data analysis and machine learning** to real-world stock data.
It successfully covered:

* Cleaning and preparing data
* Predicting stock prices using regression
* Exploring time-based patterns
* Grouping similar stocks with clustering
