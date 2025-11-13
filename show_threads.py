# show_threads.py
import pandas as pd
from collections import Counter

df = pd.read_csv("data/sample_enron_slice.csv")
print("Total messages:", len(df))

thread_counts = Counter(df["thread_id"])
print("\nTop 10 threads by message count:")
for t_id, count in thread_counts.most_common(10):
    subj = df[df["thread_id"] == t_id]["subject"].iloc[0]
    print(f"- Thread ID: {t_id} | Messages: {count} | Subject: {subj}")