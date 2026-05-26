# Netflix User Churn Prediction Model

## About
This project is a machine learning model that predicts whether a user will churn based on their viewing behavior, subscription details, and engagement metrics. It uses an XGBoost classifier inside a Scikit-learn pipeline with preprocessing for categorical and numerical features.

The output is a simple prediction: **Yes (churn)** or **No (no churn)**.

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```
### Train the model
```bash
python src/train.py
```
### Make predictions
```bash
python src/predict.py
```
