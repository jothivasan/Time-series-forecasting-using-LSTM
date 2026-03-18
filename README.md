# Multi-Domain LSTM Time Series Forecasting

> A Flask and TensorFlow application for exploring, training, evaluating, and forecasting time-series data across multiple domains.

**Status:** Public · **License:** MIT

## Overview

This project provides a configurable forecasting workflow for both univariate and multivariate datasets. Users can upload a CSV file, inspect its quality, preprocess the data, train an LSTM model, generate forecasts, and review evaluation metrics through a browser dashboard.

The system is designed to work across domains such as agriculture, energy, retail, finance, and climate data rather than being tied to one dataset or business case.

## Features

- CSV upload with automatic date-column and target-column detection
- Dataset preview, statistics, and data-quality reporting
- Missing-value handling with forward fill, backward fill, interpolation, mean, and median strategies
- Min-max, standard, and robust scaling
- Lag, rolling-statistics, and date-based feature engineering
- Configurable LSTM training with units, dropout, learning rate, epochs, and batch size
- Train, validation, and test data splitting
- Short-, mid-, long-, and custom-horizon forecasting
- Forecast date generation for daily, weekly, monthly, and hourly data
- Confidence-interval estimation for forecasts
- Evaluation with MSE, MAE, RMSE, MAPE, R², sMAPE, NRMSE, and directional accuracy
- Training-history and forecast visualizations
- Sample datasets for common forecasting scenarios

## Supported Domains

| Domain | Example use case |
| --- | --- |
| Agriculture | Crop-yield and milk-production forecasting |
| Energy | Consumption and demand forecasting |
| Retail | Sales and inventory planning |
| Finance | Stock-price trend analysis |
| Climate | Weather and environmental metrics |
| Custom | Upload any compatible time-series CSV |

## Technology

- Python 3.8+
- Flask
- TensorFlow / Keras
- pandas and NumPy
- scikit-learn
- Matplotlib
- HTML, CSS, and JavaScript dashboard

## Project Structure

```text
.
├── app.py                  # Flask application and API routes
├── config/
│   └── domains.json        # Domain and model configuration
├── core/
│   ├── data_processor.py   # Loading, cleaning, scaling, and feature engineering
│   ├── evaluator.py        # Forecast evaluation metrics and reports
│   ├── forecaster.py       # Multi-horizon forecasting utilities
│   └── lstm_model.py       # Generic TensorFlow/Keras LSTM model
├── datasets/               # Included domain datasets
├── input/                  # Upload examples and CSV guidance
├── sample_user_data/       # Sample input data
├── static/                 # Dashboard CSS and JavaScript
├── templates/              # Flask dashboard templates
├── requirements.txt        # Python dependencies
└── README.md
```

## Getting Started

### Requirements

- Python 3.8 or newer
- pip
- TensorFlow-compatible Python environment

### Installation

```bash
git clone https://github.com/jothivasan/Time-series-forecasting-using-LSTM.git
cd Time-series-forecasting-using-LSTM
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Run the dashboard

```bash
python app.py
```

Open [http://localhost:5001](http://localhost:5001) in your browser.

## Typical Workflow

1. Upload a CSV dataset from the dashboard.
2. Review the detected date column, target column, statistics, and quality report.
3. Configure missing-value handling, scaling, and optional feature engineering.
4. Configure and train the LSTM model.
5. Generate a forecast for the required horizon.
6. Review evaluation metrics and visualizations.

## API Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/upload_dataset` | `POST` | Upload and analyze a CSV dataset |
| `/api/preprocess` | `POST` | Apply preprocessing and feature engineering |
| `/api/train_model` | `POST` | Train and evaluate an LSTM model |
| `/api/forecast` | `POST` | Generate a forecast |
| `/api/evaluate` | `POST` | Generate evaluation metrics and reports |
| `/api/visualize` | `POST` | Generate training or forecast plots |

## Sample Data

The repository includes examples for retail sales, weather, production, energy, crypto prices, website traffic, air quality, and restaurant bookings. See [`input/README.md`](input/README.md) for file requirements and testing guidance.

## Notes

- Included datasets are for demonstration and educational use.
- Forecast quality depends on data volume, data quality, feature selection, and model configuration.
- Uploaded files and generated models/results are handled locally by the Flask application.

## License

This project is available under the [MIT License](LICENSE).
