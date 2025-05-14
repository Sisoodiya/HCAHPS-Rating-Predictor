# Hospital Rating Predictor

This application uses machine learning to predict hospital ratings based on patient survey data. It features an enhanced model with high accuracy (98.42%) for predicting hospital star ratings from 1 to 5 stars.

## Models

The application includes two models:

1. **Basic Model (Model1.py)**: A simpler model using Random Forest and XGBoost classifiers.
   - Accuracy: 95.45%
   - F1 Score: 95.45%
   - MAE: 0.0455

2. **Enhanced Model (Model2.py)**: A sophisticated ensemble model combining XGBoost, LightGBM, and Random Forest with advanced feature engineering.
   - Accuracy: 98.42%
   - F1 Score: 98.41%
   - MAE: 0.0158
   - Cross-Validation Accuracy: 97.87%

## Features

- Predicts hospital ratings on a 1-5 star scale
- Intuitive user interface for entering survey data
- Detailed explanation of predictions
- Visual representation of results

## Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
# Run the Streamlit app
streamlit run App.py
```

## Model Details

The enhanced model (Model2) uses the following key features for prediction:

1. Average "Sometimes/Never" response percentage
2. Weighted rating (patient survey star rating weighted by response rate)
3. Average "Always" response percentage
4. Hospital Survey Linear Mean Values
5. Nurse and Doctor communication metrics

## Data

The model was trained on the Hospital HCAHPS Survey Dataset, which includes patient experience ratings and survey data from hospitals across the United States.

## Requirements

See `requirements.txt` for a full list of dependencies.
