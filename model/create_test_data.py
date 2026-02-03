import pandas as pd

df = pd.read_csv("data/telco_churn_cleaned.csv")

X = df.drop(columns = ["Churn"])

X_test_sample = X.sample(n=200, random_state = 25)

X_test_sample.to_csv("data/telco_test_data.csv", index = False)

print("Saved test data: ", X_test_sample.shape)