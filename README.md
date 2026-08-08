# Solar Energy Prediction System

A machine learning web application that forecasts short-term solar energy generation for tropical Net Zero Energy Buildings (NZEBs), built for Sunway University's campus solar installation.

## Overview

This project trains and benchmarks five regression models on historical solar generation data, integrates a live weather forecast API for real-time predictions, and delivers an interactive dashboard for visualizing model performance and future generation estimates.

## Features

- **5 ML models benchmarked:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- **Live weather API integration:** Fetches real-time 15-minute interval meteorological data from Open-Meteo (no API key required)
- **Interactive dashboard:** Built with Streamlit and Plotly — model comparison, forecast generation, and climate analysis views
- **End-to-end pipeline:** Raw data ingestion → preprocessing → model training → evaluation → deployment

## Model Results

| Model | Test R² | Test RMSE | Test MAE |
| Random Forest | **0.842** | 136.61 | 46.21 |
| XGBoost | 0.842 | 136.85 | 48.76 |
| Gradient Boosting | 0.840 | 137.38 | 50.23 |
| Decision Tree | 0.839 | 137.92 | 46.94 |
| Linear Regression | 0.805 | 152.06 | 83.55 |

Random Forest achieved the best performance with an R² of **0.842** on the test set.

## Tech Stack

- **Language:** Python 3.x
- **Dashboard:** Streamlit, Plotly
- **ML:** scikit-learn, XGBoost
- **Data:** pandas, NumPy
- **API:** Open-Meteo REST API (requests)
- **Visualization:** Matplotlib, Seaborn, Plotly

## Project Structure

```
solar-energy-prediction/
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Dependencies
├── data/
│   ├── raw/                # Original solar generation & weather sensor data
│   └── processed/          # Train/validation/test splits
├── models/                 # Trained model files (.pkl) and metrics
├── notebooks/              # Jupyter notebooks for each model (EDA → training)
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_linear_regression.ipynb
│   ├── 04_decision_tree.ipynb
│   ├── 05_random_forest.ipynb
│   ├── 06_gradient_boost.ipynb
│   └── 07_xgboost.ipynb
└── ratio_analysis.py       # Supplementary analysis scripts
```

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/solar-energy-prediction.git
cd solar-energy-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the dashboard**
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`. No API key needed — weather data is fetched live from Open-Meteo's free endpoint.

## Dashboard Views

- **Model Validation** - Compare all 5 models on test set metrics (R², RMSE, MAE) with interactive charts
- **Forecast Generation** - Generate short-term solar output predictions using live 15-minute weather data
- **Climate Analysis** - Explore historical generation patterns and weather correlations

## Dataset

Solar generation and weather sensor data from two plant sites (Plant 1 and Plant 2), recorded at 15-minute intervals. Raw data sourced from Kaggle's Solar Power Generation dataset, adapted for tropical NZEB context.