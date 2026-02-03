import pandas as pd 

path = "data/telco_churn.csv"
df = pd.read_csv(path)

print("Shape: ", df.shape)
print("\nColumns: ", list(df.columns))
print("\nTarget value counts (Churn): ")
print(df["Churn"].value_counts(dropna=False))

print("\nDtypes (top 10): ")
print(df.dtypes.head(10))

print("\nTotal Charges sample (10): ")
print(df["TotalCharges"].head(10))

print("\nInformation about dataset: ")
print(df.info())