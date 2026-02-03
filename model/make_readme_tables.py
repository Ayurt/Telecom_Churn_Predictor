import pandas as pd

def main():
    df = pd.read_csv("model/metrics_table.csv")

    df_rounded = df.copy()
    for col in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        df_rounded[col] = df_rounded[col].round(4)

    df_rounded = df_rounded.sort_values("AUC", ascending=False)

    print("\n=== README: Metrics Comparison Table (Markdown) ===\n")
    print(df_rounded.to_markdown(index=False))

    with open("model/metrics_table.md", "w", encoding="utf-8") as f:
        f.write(df_rounded.to_markdown(index=False))

    print("\nSaved: model/metrics_table.md")

if __name__ == "__main__":
    main()
