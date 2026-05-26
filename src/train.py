import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

import joblib

df = pd.read_csv("../data/netflix_user_behavior_dataset.csv")
df

from sklearn.utils import resample

df_majority = df[df["churned"] == "No"]
df_minority = df[df["churned"] == "Yes"]

df_majority_downsampled = resample(
    df_majority,
    replace=False,
    n_samples=len(df_minority),
    random_state=42
)

df_balanced = pd.concat([df_majority_downsampled, df_minority])

X = df_balanced[["age", "gender", "country", "primary_device", "devices_used", "account_age_months", "subscription_type", "monthly_fee", "payment_method"]]
Y = df_balanced["churned"].map({
    "Yes":1,
    "No":0
})

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2, random_state=100)

categorical = [
    "gender",
    "country",
    "primary_device",
    "subscription_type",
    "payment_method"
]

numeric = [
    "age",
    "devices_used",
    "account_age_months",
    "monthly_fee"
]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ]
)

model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("classifier", XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric="logloss"
        )
        )
    ]
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, "../models/churn_model.pkl")