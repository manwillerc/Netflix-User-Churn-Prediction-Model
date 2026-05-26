import joblib
import pandas as pd

model = joblib.load("../models/churn_model.pkl")

df = pd.read_csv("../data/new_user_data.csv")

prediction = model.predict(df)

df["churned"] = pd.Series(prediction).map({1: "Yes", 0: "No"})

df.to_csv("../data/predicted_data.csv", index=False)

