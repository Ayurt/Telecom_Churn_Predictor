import pandas as pd

df = pd.read_csv("data/telco_churn.csv")

df = df.drop(columns=["customerID"])

df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

print("Shape after cleaning:", df.shape)
print("\nDtypes after cleaning:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())

df.to_csv("data/telco_churn_cleaned.csv", index=False)