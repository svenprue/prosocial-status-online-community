import pandas as pd

df = pd.read_parquet(r"C:\Users\svenp\PycharmProjects\prosocial-status-online-community\02_raw_datasets\user_answers_bounty_dataset.parquet")
df = df[df["is_history"]==0]
print(df.head(100))